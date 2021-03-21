#!/usr/bin/env bash
cd ..

xvfb-run python3 train_program.py --config configs/configs_PS/train_program_shapebd_hidden256.yaml --gpu 0,1,2,3,4,5,6,7 --seed 123
# xvfb-run python3 train_program.py --config configs/configs_PS/train_program_shapebd_hidden256.yaml --gpu 0,1,2,3,4,5,6,7 --seed 124
# xvfb-run python3 train_program.py --config configs/configs_PS/train_program_shapebd_hidden256.yaml --gpu 0,1,2,3,4,5,6,7 --seed 125