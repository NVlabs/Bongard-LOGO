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


class BasicSampler(BongardProblemSampler):

    def __init__(self, shape_actions_filepath, shape_attributes_filepath,
                 num_positive_examples=7, num_negative_examples=7,
                 random_state=np.random.RandomState(123)):
        super().__init__(num_positive_examples, num_negative_examples)

        self.random_state = random_state

        self.name = 'bd'
        self.zf_num = 4

        self.line_types = ["normal", "zigzag", "circle", "square", "triangle"]
        self.arc_types = ["normal", "zigzag", "circle", "square", "triangle"]

        self.shape_actions_dict = get_human_designed_shape_annotations(shape_actions_filepath)
        self.shape_sup_class_dict = get_shape_super_classes(shape_actions_filepath)
        self.shape_stampable_dict = get_stampable_shapes(shape_attributes_filepath)

        self.shapes_list = list(self.shape_actions_dict.keys())

    def sample(self, shapes, task_id):
        shapes_stampable = [True, True]
        for i, shape in enumerate(shapes):
            assert shape in self.shapes_list, shape
            if not self.shape_stampable_dict[shape]:
                shapes_stampable[i] = False
        positive_funcs_list, negative_funcs_list = self.sample_pos_neg_funcs(shapes)

        # Positive
        bongard_problem_positive_images = []
        for i in range(self.num_positive_examples):
            sampled_shapes = positive_funcs_list[i]
            bongard_image = self.sampled_shapes_to_bonagard_image(sampled_shapes, shapes_stampable)
            bongard_problem_positive_images.append(bongard_image)

        # Negative
        bongard_problem_negative_images = []
        for i in range(self.num_negative_examples):
            sampled_shapes = negative_funcs_list[i]
            bongard_image = self.sampled_shapes_to_bonagard_image(sampled_shapes, shapes_stampable)
            bongard_problem_negative_images.append(bongard_image)

        bongard_problem_name = '{}_{}_{}'.format(
            self.name, "-".join(shapes), str(task_id).zfill(self.zf_num))

        bongard_problem = BongardProblem(positive_bongard_images=bongard_problem_positive_images,
                                         negative_bongard_images=bongard_problem_negative_images,
                                         problem_name=bongard_problem_name, positive_rules=None,
                                         negative_rules=None)
        return bongard_problem

    def sample_pos_neg_funcs(self, shapes):
        num_attrs_per_image = len(shapes)

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

        for negative_labels in negative_labels_list:
            negative_funcs = []
            for pos_shape, label in zip(shapes, negative_labels):
                if label == 1:
                    negative_funcs.append(pos_shape)
                elif label == 0:
                    sup_class = self.shape_sup_class_dict[pos_shape]
                    neg_shape_list = [k for k, v in self.shape_sup_class_dict.items()
                                      if v == sup_class and k != pos_shape]
                    neg_shape = self.random_state.choice(neg_shape_list)
                    negative_funcs.append(neg_shape)
                else:
                    raise Exception("Unknown label!")

            negative_funcs_list.append(negative_funcs)

        for positive_labels in positive_labels_list:
            positive_funcs = []
            for pos_shape, label in zip(shapes, positive_labels):
                if label == 1:
                    positive_funcs.append(pos_shape)
                elif label == 0:
                    raise Exception("Positive label should not have zero!")
                else:
                    raise Exception("Unknown label!")
            positive_funcs_list.append(positive_funcs)

            positive_funcs_list.append(positive_funcs)

        self.random_state.shuffle(positive_funcs_list)
        self.random_state.shuffle(negative_funcs_list)

        return positive_funcs_list, negative_funcs_list

    def sampled_shapes_to_bonagard_image(self, sampled_shapes, shapes_stampable):
        bongard_image_shapes = []
        for i, sampled_shape in enumerate(sampled_shapes):
            basic_actions = []

            shape_annotation = self.shape_actions_dict[sampled_shape]
            action_name_sequence = shape_annotation[3]
            action_arguments_sequence = shape_annotation[4]
            turn_direction_sequence = shape_annotation[5]
            turn_angle_sequence = shape_annotation[6]

            candicate_line_types = self.line_types
            candicate_arc_types = self.arc_types
            if not shapes_stampable[i] or not self.shape_stampable_dict[sampled_shape]:
                candicate_line_types = ['normal']
                candicate_arc_types = ['normal']

            for action_name, action_arguments, turn_direction, turn_angle in zip(
                    action_name_sequence, action_arguments_sequence,
                    turn_direction_sequence, turn_angle_sequence):
                if action_name == "line":
                    sampled_line_type = self.random_state.choice(candicate_line_types)
                    action = LineAction(line_length=action_arguments[0], line_type=sampled_line_type,
                                        turn_direction=turn_direction, turn_angle=turn_angle)
                elif action_name == "arc":
                    sampled_arc_type = self.random_state.choice(candicate_arc_types)
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
