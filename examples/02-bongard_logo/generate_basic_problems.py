# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from bongard import BongardProblemPainter
from bongard.util_funcs import get_shape_super_classes
from bongard.sampler import BasicSampler
import numpy as np
import os, time
import json


def gen_basic_problems(ntasks, save_dir='', process_id=0, max_num_all_tasks=5000,
                       random_state=np.random.RandomState(123)):
    shape_actions_filepath = "../../data/human_designed_shapes.tsv"
    shape_attributes_filepath = "../../data/human_designed_shapes_attributes.tsv"
    shape_sup_class_dict = get_shape_super_classes(shape_actions_filepath)

    num_tests_per_bongard_problem = 1  # fixed for now

    shape_list = list(shape_sup_class_dict.keys())
    num_shapes = len(shape_list)
    print('num_shapes: ', num_shapes)

    basic_shape_list = []
    for i in range(num_shapes):
        basic_shape_list.append([shape_list[i]])

    basic_two_shapes_list = []
    for i in range(num_shapes):
        for j in range(i + 1, num_shapes):
            basic_two_shapes_list.append([shape_list[i], shape_list[j]])

    assert max_num_all_tasks > len(basic_shape_list), len(basic_shape_list)
    idx_basic_two_shapes = random_state.choice(range(len(basic_two_shapes_list)),
                                               size=max_num_all_tasks - len(basic_shape_list), replace=False)
    for i in idx_basic_two_shapes:
        basic_shape_list.append(basic_two_shapes_list[i])

    assert max_num_all_tasks >= (process_id + 1) * ntasks, (process_id + 1) * ntasks
    basic_shape_list = basic_shape_list[process_id * ntasks: (process_id + 1) * ntasks]

    save_basic_dir = os.path.join(save_dir, 'bd')
    os.makedirs(save_basic_dir, exist_ok=True)

    basic_sampler = BasicSampler(shape_actions_filepath, shape_attributes_filepath,
                                 num_positive_examples=7, num_negative_examples=7,
                                 random_state=random_state)
    problem_painter = BongardProblemPainter(scaling_factors_range=(150, 260), magnifier_effect=True, random_seed=0)

    action_programs = {}
    count = 0
    start_time = time.time()
    for shapes in basic_shape_list:
        for task_id in range(num_tests_per_bongard_problem):
            bongard_basic_problem = basic_sampler.sample(shapes, task_id)
            basic_problem_name = bongard_basic_problem.get_problem_name()
            action_program = bongard_basic_problem.get_action_string_list()
            problem_painter.create_bongard_problem(
                bongard_basic_problem,
                bongard_problem_ps_dir=os.path.join(save_basic_dir, 'ps', basic_problem_name),
                bongard_problem_png_dir=os.path.join(save_basic_dir, 'png', basic_problem_name))
            action_programs[basic_problem_name] = action_program

            count += 1
            if count % 10 == 0:
                print('generate {} concepts, time elapsed {}, current in {}'.format(
                    count, time.time() - start_time, '-'.join(shapes)))

    action_program_filepath = os.path.join(save_basic_dir, 'bd_action_program_{:03d}.json'.format(process_id))
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
    max_num_all_tasks = 5000
    gen_basic_problems(ntasks, save_dir, process_id, max_num_all_tasks, random_state)
