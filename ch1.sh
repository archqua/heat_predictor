#!/usr/bin/bash
for i in {0..3};
do
(cat try1/heater.csv; echo ""; head -n 101 try1/channel"${i}".csv) | ./hp.py --mode stiff_offset > try1/new"${i}"stiffoffset
(cat try1/heater.csv; echo ""; head -n 101 try1/channel"${i}".csv) | ./hp.py --mode stiff_penalty_offset > try1/new"${i}"stiffpenaltyoffset
done
