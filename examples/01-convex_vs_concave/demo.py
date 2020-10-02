# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from bongard import LineAction, ArcAction, OneStrokeShape, BongardImage, BongardProblem, BongardImagePainter, \
    BongardProblemPainter
from bongard.util_funcs import get_human_designed_shape_annotations, get_attribute_sampling_candidates
from bongard.plot import create_visualized_bongard_problem

import numpy as np
import os


def create_shape(shape_annotation):
    # shape_annotation: (shape_name, super_class, num_actions_computed, base_action_func_names,
    #                    base_action_func_parameters, directions, angles)

    line_arc_types = ["normal", "zigzag", "circle", "triangle"]

    base_action_func_names = shape_annotation[3]
    base_action_func_parameters = shape_annotation[4]
    directions = shape_annotation[5]
    angles = shape_annotation[6]

    base_actions = []

    for base_action_func_name, base_action_func_parameter, direction, angle in zip(
            base_action_func_names, base_action_func_parameters, directions, angles):
        if base_action_func_name == "line":
            action = LineAction(line_length=base_action_func_parameter[0],
                                line_type=np.random.choice(line_arc_types),
                                turn_direction=direction, turn_angle=angle)
        elif base_action_func_name == "arc":
            action = ArcAction(arc_angle=base_action_func_parameter[1],
                               arc_type=np.random.choice(line_arc_types),
                               turn_direction=direction, turn_angle=angle,
                               arc_radius=base_action_func_parameter[0])
        else:
            raise Exception("Unknown action type!")
        base_actions.append(action)

    shape = OneStrokeShape(basic_actions=base_actions, start_coordinates=None, start_orientation=None)

    return shape


def create_bongard_problem(positive_shapes, negative_shapes, shape_annocation_dict):
    bongard_problem_name = "Convex VS Concave"

    # Typically Bongard program consists of seven images for positive images and negative images, respectively.
    # The first six images would be used for "training", and the last image would be reserved for "test"

    bongard_problem_positive_images = []
    bongard_problem_negative_images = []

    for positive_shape in positive_shapes:
        shape_annotation = shape_annocation_dict[positive_shape]
        shape = create_shape(shape_annotation=shape_annotation)
        bongard_image = BongardImage(one_stroke_shapes=[shape])
        bongard_problem_positive_images.append(bongard_image)

    for negative_shape in negative_shapes:
        shape_annotation = shape_annocation_dict[negative_shape]
        shape = create_shape(shape_annotation=shape_annotation)
        bongard_image = BongardImage(one_stroke_shapes=[shape])
        bongard_problem_negative_images.append(bongard_image)

    bongard_problem = BongardProblem(positive_bongard_images=bongard_problem_positive_images,
                                     negative_bongard_images=bongard_problem_negative_images,
                                     problem_name=bongard_problem_name, positive_rules=None,
                                     negative_rules=None)

    return bongard_problem


if __name__ == "__main__":

    random_seed = 0
    bongard_problem_ps_dir = "./demo/ps"
    bongard_problem_png_dir = "./demo/png"
    bongard_problem_vis_filepath = "./demo/bongard_demo.png"

    annotated_shape_table_filepath = "../../data/human_designed_shapes.tsv"
    attribute_table_filepath = "../../data/human_designed_shapes_attributes.tsv"

    if not os.path.exists(bongard_problem_ps_dir):
        os.makedirs(bongard_problem_ps_dir)
    if not os.path.exists(bongard_problem_png_dir):
        os.makedirs(bongard_problem_png_dir)

    np.random.seed(random_seed)

    shape_annocation_dict = get_human_designed_shape_annotations(
        annotated_shape_table_filepath=annotated_shape_table_filepath)
    attribute_candidates = get_attribute_sampling_candidates(attribute_table_filepath=attribute_table_filepath,
                                                             min_num_positives=7, min_num_negatives=7)

    convex_shapes = attribute_candidates["convex"][0]
    concave_shapes = attribute_candidates["convex"][1]

    positive_shapes = np.random.choice(convex_shapes, size=7, replace=False)
    negative_shapes = np.random.choice(concave_shapes, size=7, replace=False)

    # Create an instance of Bongard problem based our design.
    bongard_problem = create_bongard_problem(positive_shapes=positive_shapes, negative_shapes=negative_shapes,
                                             shape_annocation_dict=shape_annocation_dict)
    # Use Bongard problem painter to draw Bongard problems.
    # The Bongard problem painter supports creating Bongard problems whose image has at most two shapes.
    bongard_problem_painter = BongardProblemPainter(random_seed=random_seed)
    # The Bongard painter will automatically create Bongard problems in the specified directories.
    # The Bongard images created will be save to hard drive
    bongard_problem_painter.create_bongard_problem(bongard_problem=bongard_problem,
                                                   bongard_problem_ps_dir=bongard_problem_ps_dir,
                                                   bongard_problem_png_dir=bongard_problem_png_dir)
    # Create a merged image for Bongard problem human-readable visualization, using the helper function.
    create_visualized_bongard_problem(bongard_problem_dir=bongard_problem_png_dir,
                                      bongard_problem_visualized_filepath=bongard_problem_vis_filepath)
