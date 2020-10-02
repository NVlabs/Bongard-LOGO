# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
import glob
import json
from multiprocessing import Pool


def run_process(process):
    os.system('python3 {}'.format(process))


def merge_action_programs(problem_type, save_dir):
    jsons_dir = '{}/{}'.format(save_dir, problem_type)
    json_files = sorted(glob.glob(os.path.join(jsons_dir, '*.json')))
    dict_merged = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            dict_merged.update(json.load(f))

    out_file = os.path.join(jsons_dir, '{}_action_programs.json'.format(problem_type))
    with open(out_file, "w") as f:
        json.dump(dict_merged, f)


# ####################################################
# ----------------- abstract shape -------------------
# ####################################################

max_ntasks = 20  # ntasks per attribute (20*220)
n_processes = 20
seed = 123
save_dir = 'images'
processes = (
    'generate_abstract_problems.py --ntasks {} --process_id {} --seed {} --save_dir {}'.format(
        max_ntasks // n_processes, i, seed + i, save_dir)
    for i in range(n_processes))

pool = Pool(processes=n_processes)
pool.map(run_process, processes)

pool.close()
pool.join()

# merge action programs
merge_action_programs(problem_type='hd', save_dir=save_dir)

# ####################################################
# ----------------- basic shape -------------------
# ####################################################

max_ntasks = 4000  # total problems (4000*1)
n_processes = 100
seed = 123
save_dir = 'images'
processes = (
    'generate_basic_problems.py --ntasks {} --process_id {} --seed {} --save_dir {}'.format(
        max_ntasks // n_processes, i, seed, save_dir)
    for i in range(n_processes))

pool = Pool(processes=n_processes)
pool.map(run_process, processes)

pool.close()
pool.join()

# merge action programs
merge_action_programs(problem_type='bd', save_dir=save_dir)

# ####################################################
# ----------------- free-form shape -------------------
# ####################################################

max_ntasks = 300  # ntasks per No. of actions (300*12)
n_processes = 100
seed = 123
save_dir = 'images'
processes = (
    'generate_ff_problems.py --ntasks {} --process_id {} --seed {} --save_dir {}'.format(
        max_ntasks // n_processes, i, seed + i, save_dir)
    for i in range(n_processes))

pool = Pool(processes=n_processes)
pool.map(run_process, processes)

pool.close()
pool.join()

# merge action programs
merge_action_programs(problem_type='ff', save_dir=save_dir)
