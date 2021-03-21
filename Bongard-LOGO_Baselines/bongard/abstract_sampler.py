# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import numpy as np

from bongard.bongard_sampler import BongardProblemSampler, OneStrokeShape, \
    BongardImage, BongardProblem
from bongard import ArcAction, LineAction
from bongard.util_funcs import get_human_designed_shape_annotations, \
    get_attribute_sampling_candidates, get_stampable_shapes


class AbstractSampler(BongardProblemSampler):

    def __init__(self, shape_actions_filepath, shape_attributes_filepath,
                 num_positive_examples=7, num_negative_examples=7, min_sampling_num=10,
                 random_state=np.random.RandomState(123)):
        super().__init__(num_positive_examples, num_negative_examples)

        self.random_state = random_state
        self.min_sampling_num = min_sampling_num

        self.name = 'hd'
        self.zf_num = 4

        self.line_types = ["normal", "zigzag", "circle", "square", "triangle"]
        self.arc_types = ["normal", "zigzag", "circle", "square", "triangle"]

        self.nonstampable_line_types = ["normal", "zigzag"]
        self.nonstampable_arc_types = ["normal", "zigzag"]

        self.shape_actions_dict = get_human_designed_shape_annotations(shape_actions_filepath)
        self.shape_attributes_dict = get_attribute_sampling_candidates(shape_attributes_filepath)
        self.shape_stampable_dict = get_stampable_shapes(shape_attributes_filepath)

        self.attribute_list = list(self.shape_attributes_dict.keys())
        self.shapes_list = list(self.shape_actions_dict.keys())

        self.nonstampable_attr_list = ['exist_regular', 'exist_triangle', 'exist_quadrangle']  # Maybe be updated later

    def sample(self, attributes, task_id):
        any_attr_stampable = True
        for attr in attributes:
            assert attr in self.attribute_list, attr
            if attr in self.nonstampable_attr_list:
                any_attr_stampable = False

        nonstampable_line_types = ["normal", "zigzag"]
        nonstampable_arc_types = ["normal", "zigzag"]

        # update nonstampable_line_types as needed for some nonstampable attributes
        if 'exist_triangle' in attributes and 'exist_quadrangle' not in attributes:
            nonstampable_line_types.extend(["circle", "square"])
            nonstampable_arc_types.extend(["circle", "square"])
        if 'exist_quadrangle' in attributes and 'exist_triangle' not in attributes:
            nonstampable_line_types.extend(["circle", "triangle"])
            nonstampable_arc_types.extend(["circle", "triangle"])
        if 'exist_triangle' in attributes and 'exist_quadrangle' in attributes:
            nonstampable_line_types.extend(["circle"])
            nonstampable_arc_types.extend(["circle"])

        positive_funcs_list, negative_funcs_list = self.sample_pos_neg_funcs(attributes)
        if len(positive_funcs_list) == 0 or len(negative_funcs_list) == 0:
            return None

        # Positive
        bongard_problem_positive_images = []
        for i in range(self.num_positive_examples):
            sampled_shapes = positive_funcs_list[i]
            bongard_image = self.sampled_shapes_to_bonagard_image(sampled_shapes, any_attr_stampable,
                                                                  nonstampable_line_types, nonstampable_arc_types)
            bongard_problem_positive_images.append(bongard_image)

        # Negative
        bongard_problem_negative_images = []
        for i in range(self.num_negative_examples):
            sampled_shapes = negative_funcs_list[i]
            bongard_image = self.sampled_shapes_to_bonagard_image(sampled_shapes, any_attr_stampable,
                                                                  nonstampable_line_types, nonstampable_arc_types)
            bongard_problem_negative_images.append(bongard_image)

        bongard_problem_name = '{}_{}_{}'.format(
            self.name, "-".join(attributes), str(task_id).zfill(self.zf_num))

        bongard_problem = BongardProblem(positive_bongard_images=bongard_problem_positive_images,
                                         negative_bongard_images=bongard_problem_negative_images,
                                         problem_name=bongard_problem_name, positive_rules=None,
                                         negative_rules=None)
        return bongard_problem

    def sample_pos_neg_funcs(self, attributes):
        num_attrs_per_image = len(attributes)
        pos0_func_names, neg0_func_names = self.shape_attributes_dict[attributes[0]]
        if num_attrs_per_image == 2:
            pos1_func_names, neg1_func_names = self.shape_attributes_dict[attributes[1]]
            pos_pos_func_names = [name for name in pos0_func_names if name in pos1_func_names]
            pos_neg_func_names = [name for name in pos0_func_names if name in neg1_func_names]
            neg_pos_func_names = [name for name in neg0_func_names if name in pos1_func_names]

            if len(pos_pos_func_names) < self.min_sampling_num or \
                    len(pos_neg_func_names) < self.min_sampling_num or \
                    len(neg_pos_func_names) < self.min_sampling_num:
                print('No enough samples in the combination: ', attributes)
                return [], []

        positive_labels_list = [[1] * num_attrs_per_image] * self.num_positive_examples
        negative_labels_list = []
        k_prime = self.num_negative_examples - self.num_negative_examples % num_attrs_per_image
        for i in range(k_prime):
            temp = [1] * num_attrs_per_image
            temp[i // (k_prime // num_attrs_per_image)] = 0
            negative_labels_list.append(temp)

        for i in range(k_prime, self.num_negative_examples):
            temp = [1] * num_attrs_per_image
            idx = self.random_state.randint(num_attrs_per_image)
            temp[idx] = 0
            negative_labels_list.append(temp)

        positive_funcs_list = []
        negative_funcs_list = []

        if num_attrs_per_image == 1:
            num_neg = negative_labels_list.count([0])
            assert num_neg == self.num_negative_examples
            negative_funcs_list.extend(
                self.random_state.choice(neg0_func_names, size=num_neg, replace=False))
        elif num_attrs_per_image == 2:
            num_neg01 = negative_labels_list.count([0, 1])
            num_neg10 = negative_labels_list.count([1, 0])
            assert num_neg01 + num_neg10 == self.num_negative_examples
            negative_funcs_list.extend(
                self.random_state.choice(neg_pos_func_names, size=num_neg01, replace=False))
            negative_funcs_list.extend(
                self.random_state.choice(pos_neg_func_names, size=num_neg10, replace=False))
        else:
            raise Exception('Unsupported num_attrs_per_image')
        negative_funcs_list = [[x] for x in negative_funcs_list]

        if num_attrs_per_image == 1:
            num_pos = positive_labels_list.count([1])
            assert num_pos == self.num_positive_examples
            positive_funcs_list.extend(
                self.random_state.choice(pos0_func_names, size=num_pos, replace=False))
        elif num_attrs_per_image == 2:
            num_pos = positive_labels_list.count([1, 1])
            assert num_pos == self.num_positive_examples
            positive_funcs_list.extend(
                self.random_state.choice(pos_pos_func_names, size=num_pos, replace=False))
        else:
            raise Exception("Positive label should not have zero!")
        positive_funcs_list = [[x] for x in positive_funcs_list]

        self.random_state.shuffle(positive_funcs_list)
        self.random_state.shuffle(negative_funcs_list)

        return positive_funcs_list, negative_funcs_list

    def sampled_shapes_to_bonagard_image(self, sampled_shapes, any_attr_stampable,
                                         nonstampable_line_types, nonstampable_arc_types):
        bongard_image_shapes = []
        for sampled_shape in sampled_shapes:
            basic_actions = []

            shape_annotation = self.shape_actions_dict[sampled_shape]
            action_name_sequence = shape_annotation[3]
            action_arguments_sequence = shape_annotation[4]
            turn_direction_sequence = shape_annotation[5]
            turn_angle_sequence = shape_annotation[6]


            candidate_line_types = self.line_types
            candidate_arc_types = self.arc_types
            if not any_attr_stampable:
                candidate_line_types = nonstampable_line_types
                candidate_arc_types = nonstampable_arc_types

            if not self.shape_stampable_dict[sampled_shape]:
                candidate_line_types = ['normal']
                candidate_arc_types = ['normal']

            for action_name, action_arguments, turn_direction, turn_angle in zip(
                    action_name_sequence, action_arguments_sequence,
                    turn_direction_sequence, turn_angle_sequence):
                if action_name == "line":
                    sampled_line_type = self.random_state.choice(candidate_line_types)
                    action = LineAction(line_length=action_arguments[0], line_type=sampled_line_type,
                                        turn_direction=turn_direction, turn_angle=turn_angle)
                elif action_name == "arc":
                    sampled_arc_type = self.random_state.choice(candidate_arc_types)
                    action = ArcAction(arc_angle=action_arguments[1], arc_type=sampled_arc_type,
                                       turn_direction=turn_direction, turn_angle=turn_angle,
                                       arc_radius=action_arguments[0])
                else:
                    raise Exception("Unsupported action name {}!".format(action_name))

                basic_actions.append(action)
            shape = OneStrokeShape(basic_actions=basic_actions, start_coordinates=None, start_orientation=None)
            bongard_image_shapes.append(shape)
        bongard_image = BongardImage(one_stroke_shapes=bongard_image_shapes)

        return bongard_image
