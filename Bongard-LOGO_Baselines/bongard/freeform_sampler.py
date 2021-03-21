# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import numpy as np
import turtle
import copy

from bongard.bongard_painter import BongardShapePainter
from bongard.bongard_sampler import BongardProblemSampler, OneStrokeShape, BongardImage, BongardProblem
from bongard import ArcAction, LineAction

MIN_MOVE = 80.
MAX_MOVE = 140.
MAX_COR = 360.

direct_angles = [0., 30., -30., 60., -60., 90., -90., 120., -120., 150., -150.]

move_types = ["normal", "zigzag", "circle", "square", "triangle"]

distances = {
    'lg': MAX_MOVE,
    'md': (MIN_MOVE + MAX_MOVE) // 2,
    'sm': MIN_MOVE,
}

move_lst = {
    'line': [['line'],
             [[1.0]],
             [None],
             [None]],
    'arc': [['arc'],
            [[0.5, 90]],
            [None],
            [None]],
    'triangle': [['line', 'line', 'line'],
                 [[1.0], [1.0], [1.0]],
                 [None, 'L', 'L'],
                 [None, 120, 120]],
    'square': [['line', 'line', 'line', 'line'],
               [[1.0], [1.0], [1.0], [1.0]],
               [None, 'L', 'L', 'L'],
               [None, 90, 90, 90]],
    'semicircle': [['arc', 'arc'],
                  [[0.5, 90], [0.5, 90]],
                  [None, 'L'],
                  [None, 0]],
    'circle': [['arc', 'arc', 'arc', 'arc'],
              [[0.5, 90], [0.5, 90], [0.5, 90], [0.5, 90]],
              [None, 'L', 'L', 'L'],
              [None, 0, 0, 0]],
    'rightangle': [['line', 'line'],
                   [[1.0], [1.0]],
                   [None, 'L'],
                   [None, 90]],
}


def sample_single_ff_action(num_rest, move_lst, move_name_lst=None, random_state=np.random.RandomState(123)):
    if move_name_lst is None:
        # There is num_actions_per_move for each candidate action in moves
        # If num_actions_per_move > num_rest
        # We skip the candidate action
        move_name_lst = [x for x in move_lst.keys() if len(move_lst[x][0]) <= num_rest]

    move_name = random_state.choice(move_name_lst)
    ff_move_annotation = move_lst[move_name].copy()

    ff_move_annotation[2][0] = 'L'  # initial direction
    ff_move_annotation[3][0] = 0.  # initial angle

    base_actions_single = []
    for action_name, action_args, turn_direct, turn_angle in zip(*ff_move_annotation):
        if action_name == "line":
            sampled_line_type = random_state.choice(move_types)
            action = LineAction(line_length=action_args[0], line_type=sampled_line_type,
                                turn_direction=turn_direct, turn_angle=turn_angle)
        elif action_name == "arc":
            sampled_arc_type = random_state.choice(move_types)
            action = ArcAction(arc_angle=action_args[1], arc_type=sampled_arc_type,
                               turn_direction=turn_direct, turn_angle=turn_angle,
                               arc_radius=action_args[0])
        else:
            raise Exception("Unsupported action name {}!".format(action_name))

        base_actions_single.append(action)

    return base_actions_single


def sample_pos_ff_actions(num_actions=1, move_lst=None, move_name_lst=None, random_state=np.random.RandomState(123)):
    base_actions = []
    num_chosen = 0
    while num_chosen != num_actions:
        base_actions_single = sample_single_ff_action(
            num_actions - num_chosen, move_lst, move_name_lst, random_state=random_state)
        base_actions.extend(base_actions_single)
        num_chosen += len(base_actions_single)

    assert num_actions == len(base_actions), base_actions
    return base_actions


def sample_neg_ff_actions(base_actions=None, move_lst=None, num_action_diffs=1, num_type_diffs=1,
                          random_state=np.random.RandomState(123)):
    neg_base_actions = base_actions.copy()
    diff_types = ['move_name', 'direct_angle', 'move_type']
    action_diff_indices = random_state.choice(range(len(base_actions)), num_action_diffs, replace=False)
    type_diffs_selected = random_state.choice(diff_types, num_type_diffs, replace=False)

    for i in action_diff_indices:
        if i == 0 and ('direct_angle' in type_diffs_selected):
            i = random_state.choice(range(1, len(base_actions)))  # no effect for the angle of the first action
            assert i > 0
        base_action = copy.deepcopy(base_actions[i])

        if 'move_name' in type_diffs_selected:
            if base_action.name == 'line':
                arc_args = move_lst['arc'][1][0]
                base_action = ArcAction(arc_angle=arc_args[1], arc_type=base_action.line_type,
                                        turn_direction=base_action.turn_direction,
                                        turn_angle=base_action.turn_angle,
                                        arc_radius=arc_args[0])
            else:
                line_args = move_lst['line'][1][0]
                base_action = LineAction(line_length=line_args[0], line_type=base_action.arc_type,
                                         turn_direction=base_action.turn_direction,
                                         turn_angle=base_action.turn_angle)

        if 'direct_angle' in type_diffs_selected:
            direct_angle = base_action.turn_angle if base_action.turn_direction == 'L' \
                else -base_action.turn_angle
            other_lst = [x for x in direct_angles if abs(x - direct_angle) > 45]
            assert other_lst is not None
            other_direct_angle = random_state.choice(other_lst)
            base_action.turn_angle = abs(other_direct_angle)
            base_action.turn_direction = 'L' if other_direct_angle >= 0 else 'R'

        if 'move_type' in type_diffs_selected:
            if base_action.name == 'line':
                other_lst = [x for x in move_types if x != base_action.line_type]
                assert other_lst is not None
                base_action.line_type = random_state.choice(other_lst)
            else:
                other_lst = [x for x in move_types if x != base_action.arc_type]
                assert other_lst is not None
                base_action.arc_type = random_state.choice(other_lst)

        neg_base_actions[i] = base_action

    return neg_base_actions


def sample_with_overlap_checking(shape_painter, sample_fn, **kwargs):
    max_iters = 100

    shape_start_coordinates = (0, 0)
    shape_start_orientation = 0
    shape_action_scaling_factor = 200

    min_move_dists = {2: 0.5, 3: 0.85, 4: 0.95, 5: 0.95, 6: 1.15, 7: 1.35, 8: 1.55, 9: 1.75}

    for _ in range(max_iters):
        base_actions = sample_fn(**kwargs)
        shape_painter.draw(base_actions, [shape_action_scaling_factor] * len(base_actions),
                           shape_start_coordinates, shape_start_orientation)

        min_move_dist = min_move_dists[len(base_actions)]
        x_range_max = abs(shape_painter.x_range_accumulated[1] - shape_painter.x_range_accumulated[0])
        y_range_max = abs(shape_painter.y_range_accumulated[1] - shape_painter.y_range_accumulated[0])
        if x_range_max > min_move_dist * shape_action_scaling_factor \
                or y_range_max > min_move_dist * shape_action_scaling_factor:
            break

    return base_actions


class FreeformSampler(BongardProblemSampler):

    def __init__(self, num_positive_examples=7, num_negative_examples=7,
                 random_state=np.random.RandomState(123)):
        super().__init__(num_positive_examples, num_negative_examples)

        screen = turtle.Screen()
        width, height = (800, 800)
        screen.setup(width=width, height=width)
        screen.screensize(width, width)
        print('Screen size: ({}, {})'.format(screen.window_height(), screen.window_width()))
        screen.bgcolor("lightgrey")

        self.wn = turtle.Turtle()
        screen.tracer(0, 0)

        self.random_state = random_state
        self.num_act_lsts = [[i] for i in range(4, 10)] + [[3, 3], [2, 5], [3, 4], [3, 5], [4, 4], [4, 5]]
        self.name = 'ff'
        self.zf_num = 4

        self.shape_painter = BongardShapePainter(screen=screen, wn=self.wn)

    def sample(self, num_act_lst, task_id):
        assert num_act_lst in self.num_act_lsts, num_act_lst
        num_shapes_per_image = len(num_act_lst)
        bongard_problem_positive_images = []
        bongard_problem_negative_images = []

        # Positive samples
        bongard_pos_image_shapes = []
        for num_actions in num_act_lst:
            base_actions = sample_with_overlap_checking(self.shape_painter, sample_pos_ff_actions,
                                                        num_actions=num_actions, move_lst=move_lst,
                                                        random_state=self.random_state)

            shape = OneStrokeShape(basic_actions=base_actions,
                                   start_coordinates=None, start_orientation=None)
            bongard_pos_image_shapes.append(shape)

        for _ in range(self.num_positive_examples):
            bongard_image = BongardImage(one_stroke_shapes=bongard_pos_image_shapes)
            bongard_problem_positive_images.append(bongard_image)

        # Negative samples
        for i in range(self.num_negative_examples):
            bongard_neg_image_shapes = []
            # decide which shape to be perturbed into the negative
            idx_neg = self.random_state.choice(num_shapes_per_image)

            pos_base_actions_lst = bongard_problem_positive_images[i].get_actions()
            neg_base_actions = sample_with_overlap_checking(self.shape_painter, sample_neg_ff_actions,
                                                            base_actions=pos_base_actions_lst[idx_neg],
                                                            move_lst=move_lst, random_state=self.random_state)

            for j in range(num_shapes_per_image):
                if j == idx_neg:
                    shape = OneStrokeShape(basic_actions=neg_base_actions,
                                           start_coordinates=None, start_orientation=None)
                    bongard_neg_image_shapes.append(shape)
                else:
                    shape = OneStrokeShape(basic_actions=pos_base_actions_lst[j],
                                           start_coordinates=None, start_orientation=None)
                    bongard_neg_image_shapes.append(shape)

            bongard_image = BongardImage(one_stroke_shapes=bongard_neg_image_shapes)
            bongard_problem_negative_images.append(bongard_image)

        bongard_problem_name = '{}_nact{}_{}'.format(
            self.name, '_'.join([str(x) for x in num_act_lst]), str(task_id).zfill(self.zf_num))

        bongard_problem = BongardProblem(positive_bongard_images=bongard_problem_positive_images,
                                         negative_bongard_images=bongard_problem_negative_images,
                                         problem_name=bongard_problem_name, positive_rules=None,
                                         negative_rules=None)
        self.wn.clear()

        return bongard_problem
