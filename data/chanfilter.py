#!/bin/env python
import argparse
from sys import stdin, stderr

parser = argparse.ArgumentParser()
parser.add_argument("ch_num", type=int)

args = parser.parse_args()

for line in stdin:
    tokens = list(map(lambda x: x.rstrip(), line.rstrip().split(',')))
    if len(tokens) > 0:
        if args.ch_num == int(tokens[0]):
            first = True
            for t in tokens[1:]:
                if first:
                    first = False
                else:
                    print(',', end='')
                print(t, end='')
            print()
