import os
import zipfile
import argparse
import requests

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download dataset for StarGAN')
parser.add_argument('dataset', metavar='N', type=str, nargs='+', choices=['celebA'],
                    help='name of dataset to download [celebA]')


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, chunk_size=32 * 1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                          unit='B', unit_scale=True, desc=destination):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_celeb_a(dirpath):
    data_dir = 'celebA'
    celebA_dir = os.path.join(dirpath, data_dir)
    prepare_data_dir(celebA_dir)

    file_name, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    txt_name, txt_drive_id = "list_attr_celeba.txt", "0B7EVK8r0v71pblRyaVFSWGxPY0U"

    save_path = os.path.join(dirpath, file_name)
    txt_save_path = os.path.join(celebA_dir, txt_name)

    if os.path.exists(txt_save_path):
        print('[*] {} already exists'.format(txt_save_path))
    else:
        download_file_from_google_drive(drive_id, txt_save_path)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(celebA_dir)

    # os.remove(save_path)
    os.rename(os.path.join(celebA_dir, 'img_align_celeba'), os.path.join(celebA_dir, 'train'))

    custom_data_dir = os.path.join(celebA_dir, 'test')
    prepare_data_dir(custom_data_dir)


def prepare_data_dir(path='./dataset'):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    args = parser.parse_args()
    prepare_data_dir()

    if any(name in args.dataset for name in ['CelebA', 'celebA', 'celebA']):
        download_celeb_a('./dataset')
