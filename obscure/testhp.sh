#!/bin/sh
printf "known\n2,3\n4,5\n5,6\n\nunknown\n2,2\n4,4\n" | ./hp.py temp_offset
