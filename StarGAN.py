from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
from glob import glob

class StarGAN(object) :
    def __init__(self, sess, args):
        self.model_name = 'StarGAN'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.dataset_path = os.path.join('./dataset', self.dataset_name)
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch
        self.selected_attrs = args.selected_attrs
        self.custom_label = np.expand_dims(args.custom_label, axis=0)
        self.c_dim = len(self.selected_attrs)

        """ Weight """
        self.adv_weight = args.adv_weight
        self.rec_weight = args.rec_weight
        self.cls_weight = args.cls_weight
        self.ld = args.ld

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# selected_attrs : ", self.selected_attrs)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, c, reuse=False, scope="generator"):
        channel = self.ch
        c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
        c = tf.tile(c, [1, x_init.shape[1], x_init.shape[2], 1])
        x = tf.concat([x_init, c], axis=-1)

        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, use_bias=False, scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, use_bias=False, scope='conv_'+str(i))
                x = instance_norm(x, scope='down_ins_norm_'+str(i))
                x = relu(x)

                channel = channel * 2

            # Bottleneck
            for i in range(self.n_res):
                x = resblock(x, channel, use_bias=False, scope='resblock_' + str(i))

            # Up-Sampling
            for i in range(2) :
                x = deconv(x, channel//2, kernel=4, stride=2, use_bias=False, scope='deconv_'+str(i))
                x = instance_norm(x, scope='up_ins_norm'+str(i))
                x = relu(x)

                channel = channel // 2


            x = conv(x, channels=3, kernel=7, stride=1, pad=3, use_bias=False, scope='G_logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_0')
            x = lrelu(x, 0.01)

            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_' + str(i))
                x = lrelu(x, 0.01)

                channel = channel * 2

            c_kernel = int(self.img_size / np.power(2, self.n_dis))

            logit = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, scope='D_logit')
            c = conv(x, channels=self.c_dim, kernel=c_kernel, stride=1, use_bias=False, scope='D_label')
            c = tf.reshape(c, shape=[-1, self.c_dim])

            return logit, c

    ##################################################################################
    # Model
    ##################################################################################

    def gradient_panalty(self, real, fake, scope="discriminator"):
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit, _ = self.discriminator(interpolated, reuse=True, scope=scope)


        GP = 0

        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp' :
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_data_class = ImageData(load_size=self.img_size, channels=self.img_ch, data_path=self.dataset_path, selected_attrs=self.selected_attrs, augment_flag=self.augment_flag)
        Image_data_class.preprocess()

        train_dataset_num = len(Image_data_class.train_dataset)
        test_dataset_num = len(Image_data_class.test_dataset)

        train_dataset = tf.data.Dataset.from_tensor_slices((Image_data_class.train_dataset, Image_data_class.train_dataset_label, Image_data_class.train_dataset_fix_label))
        test_dataset = tf.data.Dataset.from_tensor_slices((Image_data_class.test_dataset, Image_data_class.test_dataset_label, Image_data_class.test_dataset_fix_label))

        gpu_device = '/gpu:0'
        train_dataset = train_dataset.\
            apply(shuffle_and_repeat(train_dataset_num)).\
            apply(map_and_batch(Image_data_class.image_processing, self.batch_size, num_parallel_batches=8, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))

        test_dataset = test_dataset.\
            apply(shuffle_and_repeat(test_dataset_num)).\
            apply(map_and_batch(Image_data_class.image_processing, self.batch_size, num_parallel_batches=8, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))

        train_dataset_iterator = train_dataset.make_one_shot_iterator()
        test_dataset_iterator = test_dataset.make_one_shot_iterator()


        self.x_real, label_org, label_fix_list = train_dataset_iterator.get_next() # Input image / Original domain labels
        label_trg = tf.random_shuffle(label_org) # Target domain labels
        label_fix_list = tf.transpose(label_fix_list, perm=[1, 0, 2])

        self.x_test, test_label_org, test_label_fix_list = test_dataset_iterator.get_next()  # Input image / Original domain labels
        test_label_fix_list = tf.transpose(test_label_fix_list, perm=[1, 0, 2])

        self.custom_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='custom_image') # Custom Image
        custom_label_fix_list = tf.transpose(create_labels(self.custom_label, self.selected_attrs), perm=[1, 0, 2])

        """ Define Generator, Discriminator """
        x_fake = self.generator(self.x_real, label_trg) # real a
        x_recon = self.generator(x_fake, label_org, reuse=True) # real b

        real_logit, real_cls = self.discriminator(self.x_real)
        fake_logit, fake_cls = self.discriminator(x_fake, reuse=True)


        """ Define Loss """
        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
            GP = self.gradient_panalty(real=self.x_real, fake=x_fake)
        else :
            GP = 0

        g_adv_loss = generator_loss(loss_func=self.gan_type, fake=fake_logit)
        g_cls_loss = classification_loss(logit=fake_cls, label=label_trg)
        g_rec_loss = L1_loss(self.x_real, x_recon)

        d_adv_loss = discriminator_loss(loss_func=self.gan_type, real=real_logit, fake=fake_logit) + GP
        d_cls_loss = classification_loss(logit=real_cls, label=label_org)

        self.d_loss = self.adv_weight * d_adv_loss + self.cls_weight * d_cls_loss
        self.g_loss = self.adv_weight * g_adv_loss + self.cls_weight * g_cls_loss + self.rec_weight * g_rec_loss


        """ Result Image """
        self.x_fake_list = tf.map_fn(lambda x : self.generator(self.x_real, x, reuse=True), label_fix_list, dtype=tf.float32)


        """ Test Image """
        self.x_test_fake_list = tf.map_fn(lambda x : self.generator(self.x_test, x, reuse=True), test_label_fix_list, dtype=tf.float32)
        self.custom_fake_image = tf.map_fn(lambda x : self.generator(self.custom_image, x, reuse=True), custom_label_fix_list, dtype=tf.float32)


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=G_vars)
        self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=D_vars)


        """" Summary """
        self.Generator_loss = tf.summary.scalar("Generator_loss", self.g_loss)
        self.Discriminator_loss = tf.summary.scalar("Discriminator_loss", self.d_loss)

        self.g_adv_loss = tf.summary.scalar("g_adv_loss", g_adv_loss)
        self.g_cls_loss = tf.summary.scalar("g_cls_loss", g_cls_loss)
        self.g_rec_loss = tf.summary.scalar("g_rec_loss", g_rec_loss)

        self.d_adv_loss = tf.summary.scalar("d_adv_loss", d_adv_loss)
        self.d_cls_loss = tf.summary.scalar("d_cls_loss", d_cls_loss)

        self.g_summary_loss = tf.summary.merge([self.Generator_loss, self.g_adv_loss, self.g_cls_loss, self.g_rec_loss])
        self.d_summary_loss = tf.summary.merge([self.Discriminator_loss, self.d_adv_loss, self.d_cls_loss])


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag :
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay

            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.d_optimizer, self.d_loss, self.d_summary_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                g_loss = None
                if (counter - 1) % self.n_critic == 0 :
                    real_images, fake_images, _, g_loss, summary_str = self.sess.run([self.x_real, self.x_fake_list, self.g_optimizer, self.g_loss, self.g_summary_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss

                print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    real_image = np.expand_dims(real_images[0], axis=0)
                    fake_image = np.transpose(fake_images, axes=[1, 0, 2, 3, 4])[0] # [bs, c_dim, h, w, ch]

                    save_images(real_image, [1, 1],
                                './{}/real_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    save_images(fake_image, [1, self.c_dim],
                                './{}/fake_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'

        return "{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                       self.gan_type,
                                       n_res, n_dis)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_path = os.path.join(self.dataset_path, 'test')
        check_folder(test_path)
        test_files = glob(os.path.join(test_path, '*.*'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        image_folder = os.path.join(self.result_dir, 'images')
        check_folder(image_folder)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        # Custom Image
        for sample_file in test_files:
            print("Processing image: " + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(image_folder, '{}'.format(os.path.basename(sample_file)))

            fake_image = self.sess.run(self.custom_fake_image, feed_dict = {self.custom_image : sample_image})
            fake_image = np.transpose(fake_image, axes=[1, 0, 2, 3, 4])[0]
            save_images(fake_image, [1, self.c_dim], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_size, self.img_size))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_size * self.c_dim, self.img_size))
            index.write("</tr>")

        # CelebA
        real_images, fake_images = self.sess.run([self.x_test, self.x_test_fake_list])
        fake_images = np.transpose(fake_images, axes=[1, 0, 2, 3, 4])

        for i in range(len(real_images)) :
            print("{} / {}".format(i, len(real_images)))
            real_path = os.path.join(image_folder, 'real_{}.png'.format(i))
            fake_path = os.path.join(image_folder, 'fake_{}.png'.format(i))

            real_image = np.expand_dims(real_images[i], axis=0)
            fake_image = fake_images[i]
            save_images(real_image, [1, 1], real_path)
            save_images(fake_image, [1, self.c_dim], fake_path)

            index.write("<td>%s</td>" % os.path.basename(real_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (real_path if os.path.isabs(real_path) else (
                '../..' + os.path.sep + real_path), self.img_size, self.img_size))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (fake_path if os.path.isabs(fake_path) else (
                '../..' + os.path.sep + fake_path), self.img_size * self.c_dim, self.img_size))
            index.write("</tr>")

        index.close()