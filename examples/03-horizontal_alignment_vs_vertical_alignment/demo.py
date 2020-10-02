# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from bongard import LineAction, ArcAction, OneStrokeShape, BongardImage, BongardProblem, BongardImagePainter, \
    BongardProblemPainter
from bongard.plot import create_visualized_bongard_problem
import numpy as np
import os


def create_triangle(start_coordinates, start_orientation):
    triangle_actions = []

    l = np.random.uniform(low=0.1, high=0.15)
    # line_length is in [0, 1]
    action_0 = LineAction(line_length=l, line_type="normal", turn_direction="R", turn_angle=120)
    action_1 = LineAction(line_length=l, line_type="normal", turn_direction="R", turn_angle=120)
    action_2 = LineAction(line_length=l, line_type="normal", turn_direction="R", turn_angle=120)

    triangle_actions = [action_0, action_1, action_2]
    scaling_factors = [np.random.uniform(280, 300) for _ in range(len(triangle_actions))]

    shape = OneStrokeShape(basic_actions=triangle_actions, start_coordinates=start_coordinates,
                           start_orientation=start_orientation, scaling_factors=scaling_factors)

    return shape


def create_circle(start_coordinates, start_orientation):
    # arc_angle is in [-360, 360]
    arc_radius = np.random.uniform(low=0.05, high=0.075)
    action_0 = ArcAction(arc_angle=360, arc_type="normal", turn_direction="R",
                         turn_angle=0, arc_radius=arc_radius)
    circle_actions = [action_0]
    scaling_factors = [np.random.uniform(240, 250) for _ in range(len(circle_actions))]

    shape = OneStrokeShape(basic_actions=circle_actions, start_coordinates=start_coordinates,
                           start_orientation=start_orientation, scaling_factors=scaling_factors)

    return shape


def create_random_positive_image():
    # In the positive images, all the triangles are horizontally aligned.

    shapes = []

    num_triangles = np.random.randint(4, 5)
    num_circles = np.random.randint(6, 7)

    x_mean = np.random.uniform(-300, 300)

    triangle_ys = [-300 + 600 / num_triangles * i + np.random.uniform(-5, 5) for i in range(num_triangles)]
    triangle_xs = [x_mean + np.random.uniform(-5, 5) for _ in range(num_triangles)]

    def sample_circle_x(triangle_x_mean):
        if np.random.uniform(-300, 300) < x_mean:
            x = np.random.uniform(-300, triangle_x_mean - 50)
        else:
            x = np.random.uniform(triangle_x_mean + 50, 300)
        return x

    circle_xs = [sample_circle_x(triangle_x_mean=x_mean) for _ in range(num_circles)]

    for i in range(num_triangles):
        triangle = create_triangle(start_coordinates=(triangle_ys[i], triangle_xs[i]), start_orientation=120)
        shapes.append(triangle)

    for i in range(num_circles):
        circle = create_circle(start_coordinates=(np.random.uniform(-300, 300), circle_xs[i]),
                               start_orientation=np.random.uniform(-360, 360))
        shapes.append(circle)

    bongard_image = BongardImage(one_stroke_shapes=shapes)

    return bongard_image


def create_random_negative_image():
    # In the negative images, all the triangles are vertically aligned.

    shapes = []

    num_triangles = np.random.randint(4, 5)
    num_circles = np.random.randint(6, 7)

    y_mean = np.random.uniform(-300, 300)

    triangle_xs = [-300 + 600 / num_triangles * i + np.random.uniform(-5, 5) for i in range(num_triangles)]
    triangle_ys = [y_mean + np.random.uniform(-5, 5) for _ in range(num_triangles)]

    def sample_circle_y(triangle_y_mean):
        if np.random.uniform(-300, 300) < y_mean:
            y = np.random.uniform(-300, triangle_y_mean - 50)
        else:
            y = np.random.uniform(triangle_y_mean + 50, 300)
        return y

    circle_ys = [sample_circle_y(triangle_y_mean=y_mean) for _ in range(num_circles)]

    for i in range(num_triangles):
        triangle = create_triangle(start_coordinates=(triangle_ys[i], triangle_xs[i]), start_orientation=120)
        shapes.append(triangle)

    for i in range(num_circles):
        circle = create_circle(start_coordinates=(circle_ys[i], np.random.uniform(-300, 300)),
                               start_orientation=np.random.uniform(-360, 360))
        shapes.append(circle)

    bongard_image = BongardImage(one_stroke_shapes=shapes)

    return bongard_image


def create_bongard_problem():
    bongard_problem_name = "Rectangle VS Circle"

    # Typically Bongard program consists of seven images for positive images and negative images, respectively.
    # The first six images would be used for "training", and the last image would be reserved for "test"
    bongard_problem_positive_images = [create_random_positive_image() for _ in range(7)]
    bongard_problem_negative_images = [create_random_negative_image() for _ in range(7)]

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
                                                   bongard_problem_png_dir=bongard_problem_png_dir,
                                                   auto_position=False)
    # Create a merged image for Bongard problem human-readable visualization, using the helper function.
    create_visualized_bongard_problem(bongard_problem_dir=bongard_problem_png_dir,
                                      bongard_problem_visualized_filepath=bongard_problem_vis_filepath)
