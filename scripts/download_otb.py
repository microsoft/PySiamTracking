import os
import json
import wget
import argparse
import zipfile

from tqdm import tqdm
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Download official OTB benchmark.')
    parser.add_argument('--root_dir',
                        type=str,
                        default='data/benchmark/otb/',
                        help='directory path that used to save otb benchmark.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    url = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/{seq_name}.zip'
    args = parse_args()
    if not os.path.isdir(args.root_dir):
        os.makedirs(args.root_dir)

    otb_meta_file_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        '../siam_tracker/benchmarks/data/meta_info/otb.json'
    )

    with open(otb_meta_file_path, 'r') as f:
        seq_info_list = json.load(f)

    file_name_dict = OrderedDict()
    for seq_info in seq_info_list:
        file_name = seq_info['path'].split('/')[0]
        if file_name not in file_name_dict:
            file_name_dict[file_name] = url.format(seq_name=file_name)

    print("Downloading OTB benchmark...")
    print("All {} files...".format(len(file_name_dict)))

    for file_name, download_link in tqdm(file_name_dict.items()):
        save_path = os.path.join(args.root_dir, '{}.zip'.format(file_name))
        if not os.path.exists(save_path):
            wget.download(download_link, save_path, bar=None)
        with zipfile.ZipFile(save_path, 'r') as ref:
            ref.extractall(args.root_dir)

    print("Done! OTB benchmark has been saved to '{}'".format(args.root_dir))

    # clean files...
    for file_name, download_link in tqdm(file_name_dict.items()):
        save_path = os.path.join(args.root_dir, '{}.zip'.format(file_name))
        if os.path.exists(save_path):
            os.remove(save_path)
    print("Clean cached files successfully.")
