# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT
import matplotlib
matplotlib.use('Agg')
print("here")

from bongard import BongardProblemPainter
from bongard.util_funcs import get_shape_super_classes
from bongard.sampler import BasicSampler
from bongard.bongard import *
import numpy as np
import os, time
import json

def parse_action_string(action_string, line_length_normalization_factor=None):
    """
    Parse an line action_string.
    For example, "line_straight_0.7905-0.7500"
    """

    movement, turn_angle = action_string.split("-")
    turn_angle = float(turn_angle)
    action_name, line_type, line_length = movement.split("_")
    line_length = float(line_length)

    if action_name != "line":
        raise Exception("The action string imported is not a line action string!")

    if line_length_normalization_factor is not None:
        denormalized_line_length = BasicAction.denormalize_line_length(
            normalized_line_length=line_length,
            line_length_max=line_length_normalization_factor)
    else:
        denormalized_line_length = line_length

    turn_direction, denormalized_turn_angle = BasicAction.denormalize_turn_angle(normalized_turn_angle=turn_angle)

    return denormalized_line_length, line_type, turn_direction, denormalized_turn_angle

def gen_basic_problems(ntasks, save_dir='', process_id=0, max_num_all_tasks=5000,
                       random_state=np.random.RandomState(123), probe_lengths=False, probe_angles=False):
    shape_actions_filepath = "./data/human_designed_shapes.tsv"
    shape_attributes_filepath = "./data/human_designed_shapes_attributes.tsv"
    shape_sup_class_dict = get_shape_super_classes(shape_actions_filepath)

    num_tests_per_bongard_problem = 1  # fixed for now

    shape_list = list(shape_sup_class_dict.keys())[:8]
    # print(shape_list)
    num_shapes = len(shape_list)
    print('num_shapes: ', num_shapes)

    basic_shape_list = []
    for i in range(num_shapes):
        basic_shape_list.append([shape_list[i]])

    save_basic_dir = os.path.join(save_dir, 'bd')
    os.makedirs(save_basic_dir, exist_ok=True)

    basic_sampler = BasicSampler(shape_actions_filepath, shape_attributes_filepath,
                                 num_positive_examples=7, num_negative_examples=7,
                                 random_state=random_state, line_types=["normal"])
    problem_painter = BongardProblemPainter(scaling_factors_range=(100, 100), magnifier_effect=False, random_seed=0)

    action_programs = {}
    count = 0
    start_time = time.time()
    # print(basic_shape_list)
    for shapes in basic_shape_list:
        # print(shapes)
        for task_id in range(num_tests_per_bongard_problem):
            bongard_basic_problem = basic_sampler.sample(shapes, task_id)
            basic_problem_name = bongard_basic_problem.get_problem_name()
            action_program = bongard_basic_problem.get_action_string_list()
            p = action_program[0][0][0]

            pos_shapes = []
            neg_shapes = []
            p = action_program[0][0][0]
            if probe_lengths:
                for _ in range(20):
                    actions = []
                    e = np.random.uniform(0,0.4,3)
                    e.sort()
                    idx=2
                    for program in p: 
                        l, lt, td, ta = parse_action_string(program)
                        l += e[idx]
                        print(l, lt, td, ta)
                        la = LineAction(line_length=l, line_type=lt, turn_direction=td, turn_angle=ta)
                        actions.append(la)
                        idx-=1

                    shape = OneStrokeShape(basic_actions=actions)
                    shape = BongardImage(one_stroke_shapes=[shape])
                    pos_shapes.append(shape)

            if probe_angles:
                for _ in range(20):
                    actions = []
                    e = np.random.uniform(0,5,3)
                    e.sort()
                    idx=2
                    for program in p: 
                        l, lt, td, ta = parse_action_string(program)
                        ta += e[idx]
                        print(l, lt, td, ta)
                        la = LineAction(line_length=l, line_type=lt, turn_direction=td, turn_angle=ta)
                        actions.append(la)
                        idx-=1

                    shape = OneStrokeShape(basic_actions=actions)
                    shape = BongardImage(one_stroke_shapes=[shape])
                    pos_shapes.append(shape)
            
            # for ap in action_program[1]:
            #     ap = ap[0]
            #     action = []
            #     e = np.random.uniform(0,0.4,3)
            #     e.sort()
            #     idx=2
            #     for program in ap:
            #         l, lt, td, ta = parse_action_string(program)
            #         l += e[idx]
            #         print(l, lt, td, ta)
            #         la = LineAction(line_length=l, line_type=lt, turn_direction=td, turn_angle=ta)
            #         action.append(la)
            #         idx-=1

            #     shape = OneStrokeShape(basic_actions=action)
            #     shape = BongardImage(one_stroke_shapes=[shape])
            #     neg_shapes.append(shape)
            # print(action_program)
            # print()
            # print(p)
            # exit()

            bongard_problem = BongardProblem(positive_bongard_images=pos_shapes, negative_bongard_images=pos_shapes, problem_name=basic_problem_name)
            bongard_problem_painter = BongardProblemPainter(scaling_factors_range=(100,100), random_seed=121)
            # `BongardProblemPainter` will generate `ps` and `png` format images for the `BongardProblem` provided.
            bongard_problem_painter.create_bongard_problem(bongard_problem=bongard_problem, 
                                                           bongard_problem_ps_dir=os.path.join(save_basic_dir, 'ps', basic_problem_name), 
                                                           bongard_problem_png_dir=os.path.join(save_basic_dir, 'png', basic_problem_name))
            action_programs[basic_problem_name] = action_program

            count += 1
            if count % 10 == 0:
                print('generate {} concepts, time elapsed {}, current in {}'.format(
                    count, time.time() - start_time, '-'.join(shapes)))

        # break

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
    parser.add_argument("--probe_angle",type=int, default=0, help="wheter to probe angle")
    parser.add_argument("--probe_length", type=int, default=0, help="wheter to probe length")
    args = parser.parse_args()

    seed = args.seed
    random_state = np.random.RandomState(seed)
    ntasks = args.ntasks
    process_id = args.process_id
    save_dir = args.save_dir
    max_num_all_tasks = 5000
    probe_length = True if args.probe_length == 1 else False
    probe_angle = True if args.probe_angle == 1 else False
    gen_basic_problems(ntasks, save_dir, process_id, max_num_all_tasks, random_state, probe_length, probe_angle)
