# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from bongard import BongardProblemPainter
from bongard.sampler import FreeformSampler
import numpy as np
import os, time
import json


def gen_ff_problems(ntasks, save_dir='', process_id=0,
                    random_state=np.random.RandomState(123)):
    num_act_lsts = [[i] for i in range(4, 10)] + [[3, 3], [2, 5], [3, 4], [3, 5], [4, 4], [4, 5]]
    save_ff_dir = os.path.join(save_dir, 'ff')
    os.makedirs(save_ff_dir, exist_ok=True)

    freeform_sampler = FreeformSampler(num_positive_examples=7, num_negative_examples=7,
                                       random_state=random_state)
    problem_painter = BongardProblemPainter(scaling_factors_range=(80, 140), magnifier_effect=True, random_seed=0)

    action_programs = {}
    count = 0
    start_time = time.time()
    for num_act_lst in num_act_lsts:
        for task_id in range(ntasks):
            task_process_id = task_id + process_id * ntasks
            bongard_ff_problem = freeform_sampler.sample(num_act_lst, task_process_id)
            ff_problem_name = bongard_ff_problem.get_problem_name()
            action_program = bongard_ff_problem.get_action_string_list()
            problem_painter.create_bongard_problem(
                bongard_ff_problem,
                bongard_problem_ps_dir=os.path.join(save_ff_dir, 'ps', ff_problem_name),
                bongard_problem_png_dir=os.path.join(save_ff_dir, 'png', ff_problem_name))
            action_programs[ff_problem_name] = action_program

            count += 1
            if count % 10 == 0:
                print('generate {} concepts, time elapsed {}, current in nact{}'.format(
                    count, time.time() - start_time, '_'.join([str(x) for x in num_act_lst])))

    action_program_filepath = os.path.join(save_ff_dir, 'ff_action_program_{:03d}.json'.format(process_id))
    with open(action_program_filepath, "w") as fhand:
        json.dump(action_programs, fhand)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ntasks', type=int, default=1, help='number of tasks per action setting')
    parser.add_argument('--process_id', type=int, default=0, help='which process that runs the job')
    parser.add_argument('--seed', type=int, default=123, help='seed to reproduce results')
    parser.add_argument('--save_dir', type=str, default='images', help='dir to save results')
    args = parser.parse_args()

    seed = args.seed
    random_state = np.random.RandomState(seed)
    ntasks = args.ntasks
    process_id = args.process_id
    save_dir = args.save_dir
    gen_ff_problems(ntasks, save_dir, process_id, random_state)
