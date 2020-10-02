# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

"""
In this example, we are going to create a Bongard problem where the positive images are all rectangles,
whereas the negative images are all circles, which violate the concept in the positive images. 
"""

from bongard import LineAction, ArcAction, OneStrokeShape, BongardImage, BongardProblem, BongardImagePainter, \
    BongardProblemPainter
from bongard.plot import create_visualized_bongard_problem
import numpy as np
import os


def create_random_rectangle():
    line_arc_types = ["normal", "zigzag", "circle", "triangle"]

    rectangle_actions = []

    w = np.random.uniform(low=0.3, high=1.0)
    h = np.random.uniform(low=0.3, high=1.0)
    # line_length is in [0, 1]
    action_0 = LineAction(line_length=w, line_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=90)
    action_1 = LineAction(line_length=h, line_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=90)
    action_2 = LineAction(line_length=w, line_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=90)
    action_3 = LineAction(line_length=h, line_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=90)

    rectangle_actions = [action_0, action_1, action_2, action_3]

    shape = OneStrokeShape(basic_actions=rectangle_actions, start_coordinates=None, start_orientation=None)

    return shape


def create_random_circle():
    line_arc_types = ["normal", "zigzag", "circle", "triangle"]
    # arc_angle is in [-360, 360]
    arc_radius = np.random.uniform(low=0.2, high=0.5)
    action_0 = ArcAction(arc_angle=90, arc_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=0,
                         arc_radius=arc_radius)
    action_1 = ArcAction(arc_angle=90, arc_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=0,
                         arc_radius=arc_radius)
    action_2 = ArcAction(arc_angle=90, arc_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=0,
                         arc_radius=arc_radius)
    action_3 = ArcAction(arc_angle=90, arc_type=np.random.choice(line_arc_types), turn_direction="R", turn_angle=0,
                         arc_radius=arc_radius)

    circle_actions = [action_0, action_1, action_2, action_3]

    shape = OneStrokeShape(basic_actions=circle_actions, start_coordinates=None, start_orientation=None)

    return shape


def create_random_rectangle_image():
    num_shapes_candidate = [1, 2]
    shapes = [create_random_rectangle() for _ in range(np.random.choice(num_shapes_candidate))]
    bongard_image = BongardImage(one_stroke_shapes=shapes)

    return bongard_image


def create_random_circle_image():
    num_shapes_candidate = [1, 2]
    shapes = [create_random_circle() for _ in range(np.random.choice(num_shapes_candidate))]
    bongard_image = BongardImage(one_stroke_shapes=shapes)

    return bongard_image


def create_bongard_problem():
    bongard_problem_name = "Rectangle VS Circle"

    # Typically Bongard program consists of seven images for positive images and negative images, respectively.
    # The first six images would be used for "training", and the last image would be reserved for "test"
    bongard_problem_positive_images = [create_random_rectangle_image() for _ in range(7)]
    bongard_problem_negative_images = [create_random_circle_image() for _ in range(7)]

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

    if not os.path.exists(bongard_problem_ps_dir):
        os.makedirs(bongard_problem_ps_dir)
    if not os.path.exists(bongard_problem_png_dir):
        os.makedirs(bongard_problem_png_dir)

    np.random.seed(random_seed)

    # Create an instance of Bongard problem based our design.
    bongard_problem = create_bongard_problem()
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
    # It is feasible to get the action program from Bongard problem
    # Note that the action program is a "normalized" action program and it does not contain the shape location info
    action_program = bongard_problem.get_action_string_list()
    print("=" * 75)
    print("Bongard problem action program:")
    print("-" * 75)
    print(action_program)
    print("=" * 75)
    # Optional
    # Dump action program to hard drive 
    # Optional
    # Read action program from hard drive 
    # It is also possible to reconstruct the Bongard problem from action program
    bongard_problem_recovered = BongardProblem.import_from_action_string_list(action_string_list=action_program)
    action_program_recovered = bongard_problem_recovered.get_action_string_list()
    print("Bongard problem recovered program:")
    print("-" * 75)
    print(action_program_recovered)
    print("=" * 75)
    print("Bongard problem successfully recovered: {}".format(action_program == action_program_recovered))
    print("=" * 75)
