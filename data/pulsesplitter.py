#!/bin/env python
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("fmt", help="% sign will be replaced with pulse number")

npulse = 1
for pulse in pulses(Text(sys.stdin.read())):
    pulse.print()

class Text:
    def __init__(self, src):
        self.src = list(map(lambda x: x.rstrip().split(','), src.rstrip().splitlines()))

class Pulse:
    def __init__(self, lo, hi, text):
        self.lo = lo
        self.hi = hi
        self.text = text

    def print(skip=0, file=sys.stdout):
        for i in range(self.lo, self.hi):
            print(*tuple(self.text[i][skip:]), sep='', file=file)

def pulses(text):
    pass
