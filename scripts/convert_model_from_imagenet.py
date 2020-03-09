import _init_paths
import torch
import os
import argparse
from collections import OrderedDict

from torchvision.models import alexnet, resnet18, resnet34, resnet50, mobilenet_v2


NETWORKS = dict(
    alexnet=dict(
        func_name=alexnet,
        map_file='data/pretrained_models/alexnet_map.csv',
        output_file='data/pretrained_models/alexnet.pth',
    ),
    resnet18=dict(
        func_name=resnet18,
        map_file='data/pretrained_models/resnet18_map.csv',
        output_file='data/pretrained_models/resnet18.pth',
    ),
    resnet34=dict(
        func_name=resnet34,
        map_file='data/pretrained_models/resnet34_map.csv',
        output_file='data/pretrained_models/resnet34.pth',
    ),
    resnet50=dict(
        func_name=resnet50,
        map_file='data/pretrained_models/resnet50_map.csv',
        output_file='data/pretrained_models/resnet50.pth',
    ),
    mobilenetv2=dict(
        func_name=mobilenet_v2,
        map_file='data/pretrained_models/mobilenet_map.csv',
        output_file='data/pretrained_models/mobilenet.pth',
    )
)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert a ImageNet-pretrained model into our implementation')
    parser.add_argument('--net', default='all', type=str, help='network type.')
    args = parser.parse_args()
    return args


def load_tensor_map(map_path):
    with open(map_path, 'r') as f:
        lines = f.read().splitlines()
    new_from_old = OrderedDict()
    for line in lines:
        if line.strip() == '':
            continue
        sp = line.split('\t')
        new_from_old[sp[1]] = sp[0]
    return new_from_old


if __name__ == '__main__':
    args = parse_args()
    if args.net == 'all':
        network_list = list(NETWORKS.keys())
    else:
        network_list = [args.net]

    for network_name in network_list:
        print("Process {}".format(network_name))
        model = NETWORKS[network_name]['func_name'](pretrained=True, progress=True)
        tensor_map = load_tensor_map(NETWORKS[network_name]['map_file'])
        new_tensors = OrderedDict()
        for new_name, old_name in tensor_map.items():
            new_tensors[new_name] = model.state_dict()[old_name]
            print("Loading {} from {}".format(new_name, old_name))
        output_path = NETWORKS[network_name]['output_file']
        output_dir = os.path.dirname(output_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        torch.save(new_tensors, output_path)
        print("Finish.")
        print("*"*10)
