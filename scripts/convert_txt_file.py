import sys, os
import random
import argparse
import itertools
import numpy as np
import ast

random.seed(0)


def str2array(attr_bins):
    assert (isinstance(attr_bins, str))
    attr_bins = attr_bins.strip()
    if attr_bins.endswith(('.npy', '.npz')):
        attr_bins = np.load(attr_bins)
    else:
        assert (attr_bins.startswith('[') and attr_bins.endswith(']'))
        attr_bins = ast.literal_eval(attr_bins)
    return attr_bins


def get_attr(fname):
    if len(fname.split()) > 1:
        attr = float(fname.split()[1])
    else:
        attr = float(fname.split('_')[0])
    return attr


def get_attr_label(attr, bins):
    L = None
    for L in range(len(bins) - 1):
        if (attr >= bins[L]) and (attr < bins[L + 1]):
            break
    return L


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input.txt', help='txt file')
parser.add_argument('--output', type=str, default='output.txt', help='base name for output file')
parser.add_argument('--bins', type=str, default='[]')
opt = parser.parse_args()
print(opt.bins)
opt.bins = str2array(opt.bins) + [float('inf')]
print(opt.bins)

with open(opt.input, 'r') as f_in, open(opt.output, 'w') as f_out:
    for row in f_in.readlines():
        row_ = row.strip('\n').split(' ')
        filename = row_[0]
        label = get_attr_label(get_attr(row_[0]), opt.bins)
        f_out.write(f'{filename} {label}\n')
