#!/usr/bin/env bash

# Make dirs
mkdir ./data
mkdir ./data/preprocessed

# Preprocess
python ./data/adult_preprocess.py
python ./data/crash_preprocess.py
python ./data/crime_preprocess.py
