#!/usr/bin/env bash

for task in crash adult property person society
do
    for alpha in 0.01 0.05 0.1
    do
        python real_data_exp.py $task $alpha
        read -p "Press enter to continue"
    done
done
