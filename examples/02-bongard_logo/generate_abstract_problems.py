# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from bongard import BongardProblemPainter
from bongard.util_funcs import get_attribute_sampling_candidates
from bongard.sampler import AbstractSampler
import numpy as np
import os, time
import json


def gen_abstrct_problems(ntasks, save_dir='', process_id=0,
                         random_state=np.random.RandomState(123)):
    shape_actions_filepath = "../../data/human_designed_shapes.tsv"
    shape_attributes_filepath = "../../data/human_designed_shapes_attributes.tsv"
    shape_attributes_dict = get_attribute_sampling_candidates(shape_attributes_filepath)

    attribute_list = list(shape_attributes_dict.keys())
    num_attributes = len(attribute_list)

    bongard_attributes = []
    for i in range(num_attributes):
        bongard_attributes.append([attribute_list[i]])
    for i in range(num_attributes):
        for j in range(i + 1, num_attributes):
            bongard_attributes.append([attribute_list[i], attribute_list[j]])

    print('total number of bongard_attributes: ', len(bongard_attributes))

    save_abs_dir = os.path.join(save_dir, 'hd')
    os.makedirs(save_abs_dir, exist_ok=True)

    abstract_sampler = AbstractSampler(shape_actions_filepath, shape_attributes_filepath,
                                       num_positive_examples=7, num_negative_examples=7,
                                       random_state=random_state)
    problem_painter = BongardProblemPainter(scaling_factors_range=(150, 320), magnifier_effect=True, random_seed=0)

    action_programs = {}
    count = 0
    start_time = time.time()
    for attributes in bongard_attributes:
        for task_id in range(ntasks):
            task_process_id = task_id + process_id * ntasks
            bongard_abs_problem = abstract_sampler.sample(attributes, task_process_id)

            # decide if there is enough samples for the given attributes
            if bongard_abs_problem is None:
                break

            abs_problem_name = bongard_abs_problem.get_problem_name()
            action_program = bongard_abs_problem.get_action_string_list()
            problem_painter.create_bongard_problem(
                bongard_abs_problem,
                bongard_problem_ps_dir=os.path.join(save_abs_dir, 'ps', abs_problem_name),
                bongard_problem_png_dir=os.path.join(save_abs_dir, 'png', abs_problem_name))
            action_programs[abs_problem_name] = action_program

        count += 1
        if count % 10 == 0:
            print('generate {} concepts, time elapsed {}, current in {}'.format(
                count, time.time() - start_time, '-'.join(attributes)))

    action_program_filepath = os.path.join(save_abs_dir, 'hd_action_program_{:03d}.json'.format(process_id))
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
    gen_abstrct_problems(ntasks, save_dir, process_id, random_state)
