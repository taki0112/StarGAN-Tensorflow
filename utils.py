import scipy.misc
import numpy as np
import os
from scipy import misc

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

class ImageData:

    def __init__(self, load_size, channels, data_path, selected_attrs, augment_flag=False):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag
        self.selected_attrs = selected_attrs

        self.data_path = os.path.join(data_path, 'train')
        check_folder(self.data_path)
        self.lines = open(os.path.join(data_path, 'list_attr_celeba.txt'), 'r').readlines()

        self.train_dataset = []
        self.train_dataset_label = []
        self.train_dataset_fix_label = []

        self.test_dataset = []
        self.test_dataset_label = []
        self.test_dataset_fix_label = []

        self.attr2idx = {}
        self.idx2attr = {}

    def image_processing(self, filename, label, fix_label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()

            if p > 0.5 :
                img = augmentation(img, augment_size)


        return img, label, fix_label

    def preprocess(self) :
        all_attr_names = self.lines[1].split()
        for i, attr_name in enumerate(all_attr_names) :
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name


        lines = self.lines[2:]
        random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(lines) :
            split = line.split()
            filename = os.path.join(self.data_path, split[0])
            values = split[1:]

            label = []

            for attr_name in self.selected_attrs :
                idx = self.attr2idx[attr_name]

                if values[idx] == '1' :
                    label.append(1.0)
                else :
                    label.append(0.0)

            if i < 2000 :
                self.test_dataset.append(filename)
                self.test_dataset_label.append(label)
            else :
                self.train_dataset.append(filename)
                self.train_dataset_label.append(label)
            # ['./dataset/celebA/train/019932.jpg', [1, 0, 0, 0, 1]]

        self.test_dataset_fix_label = create_labels(self.test_dataset_label, self.selected_attrs)
        self.train_dataset_fix_label = create_labels(self.train_dataset_label, self.selected_attrs)

        print('\n Finished preprocessing the CelebA dataset...')

def load_test_data(image_path, size=128):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = normalize(img)

    return img

def augmentation(image, aug_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [aug_size, aug_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img

    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def create_labels(c_org, selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    c_org = np.asarray(c_org)
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []

    for i in range(len(selected_attrs)):
        c_trg = c_org.copy()

        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1.0
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0.0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

        c_trg_list.append(c_trg)

    c_trg_list = np.transpose(c_trg_list, axes=[1, 0, 2]) # [c_dim, bs, ch]

    return c_trg_list