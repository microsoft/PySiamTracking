import os
import wget
import shutil
import argparse
import zipfile
import requests

from tqdm import tqdm
from collections import OrderedDict


DESCRIPTION_URL = 'http://data.votchallenge.net/vot{year}/main/description.json'
MAIN_URL = 'http://data.votchallenge.net/vot{year}/main/'


def parse_args():
    parser = argparse.ArgumentParser(description='Download official VOT benchmark.')
    parser.add_argument('--root_dir',
                        type=str,
                        default='data/benchmark/vot2017',
                        help='directory path that used to save otb benchmark.')
    parser.add_argument('--year',
                        type=str,
                        default='2017',
                        help='VOT dataset years')
    args = parser.parse_args()

    return args


def get_description(year):
    url = DESCRIPTION_URL.format(year=year)
    res = requests.get(url)
    assert res.status_code == 200, 'Cannot load description file. Please download VOT manually.'
    seq_info_list = res.json()['sequences']
    return seq_info_list


if __name__ == '__main__':

    args = parse_args()
    assert args.year in ['2016', '2017', '2019'], 'Support VOT-16 / VOT-17 only.'
    seq_info_list = get_description(args.year)

    if not os.path.isdir(args.root_dir):
        os.makedirs(args.root_dir)
    tmp_dir = os.path.join(args.root_dir, 'tmp')
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    print("Downloading VOT{} benchmarks.".format(args.year))
    print("Save to {}".format(args.root_dir))

    for seq_info in tqdm(seq_info_list):
        seq_name = seq_info['name']
        anno_zip_url = MAIN_URL.format(year=args.year) + seq_info['annotations']['url']
        frame_zip_url = MAIN_URL.format(year=args.year) + seq_info['channels']['color']['url']

        # download to tmp_dir
        anno_path = os.path.join(tmp_dir, '{}_anno.zip'.format(seq_name))
        if not os.path.exists(anno_path):
            wget.download(anno_zip_url, anno_path, bar=None)
        frame_path = os.path.join(tmp_dir, '{}_frame.zip'.format(seq_name))
        if not os.path.exists(frame_path):
            wget.download(frame_zip_url, frame_path, bar=None)

        # unzip file to target directory
        output_dir = os.path.join(args.root_dir, seq_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with zipfile.ZipFile(anno_path) as ref:
            ref.extractall(output_dir)

        output_dir = os.path.join(args.root_dir, seq_name, 'color')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with zipfile.ZipFile(frame_path) as ref:
            ref.extractall(output_dir)

    # clean tmp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done!")
