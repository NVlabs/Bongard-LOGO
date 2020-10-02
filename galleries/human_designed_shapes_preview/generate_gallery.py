# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from bongard import LineAction, ArcAction, OneStrokeShape, BongardImage, BongardImagePainter
from bongard.util_funcs import get_human_designed_shape_annotations

import turtle
import os
from collections import OrderedDict
from PIL import Image
from multiprocessing import Pool
import shutil

def generate_shape_collections(annotated_shape_table_filepath, line_or_arc_type="normal"):
    start_coordinates = (0, 0)
    start_orientation = 0
    action_scaling_factor = 200

    annotation_dict = get_human_designed_shape_annotations(
        annotated_shape_table_filepath=annotated_shape_table_filepath)

    bongard_images = OrderedDict()

    for shape_name in annotation_dict:

        basic_actions = []
        shape_annotation = annotation_dict[shape_name]
        action_name_sequence = shape_annotation[3]
        action_arguments_sequence = shape_annotation[4]
        turn_direction_sequence = shape_annotation[5]
        turn_angle_sequence = shape_annotation[6]
        for action_name, action_arguments, turn_direction, turn_angle in zip(action_name_sequence,
                                                                             action_arguments_sequence,
                                                                             turn_direction_sequence,
                                                                             turn_angle_sequence):
            if action_name == "line":
                action = LineAction(line_length=action_arguments[0], line_type=line_or_arc_type,
                                    turn_direction=turn_direction, turn_angle=turn_angle)
            elif action_name == "arc":
                action = ArcAction(arc_angle=action_arguments[1], arc_type=line_or_arc_type,
                                   turn_direction=turn_direction, turn_angle=turn_angle, arc_radius=action_arguments[0])
            else:
                raise Exception("Unsupported action name {}!".format(action_name))

            basic_actions.append(action)
        shape = OneStrokeShape(basic_actions=basic_actions, start_coordinates=start_coordinates,
                               start_orientation=start_orientation,
                               scaling_factors=[action_scaling_factor] * len(basic_actions))
        bongard_image = BongardImage(one_stroke_shapes=[shape])
        bongard_images[shape_name] = bongard_image

    return bongard_images


def save_shape_images(shape_images, screen, wn, shape_image_ps_dir, shape_image_png_dir):
    png_filepath_dict = OrderedDict()

    screen.tracer(0, 0)

    if not os.path.exists(shape_image_ps_dir):
        os.makedirs(shape_image_ps_dir)
    if not os.path.exists(shape_image_png_dir):
        os.makedirs(shape_image_png_dir)

    painter = BongardImagePainter(screen=screen, wn=wn)

    for shape_name in shape_images.keys():
        ps_filename = "{}.ps".format(shape_name)
        png_filename = "{}.png".format(shape_name)
        ps_filepath = os.path.join(shape_image_ps_dir, ps_filename)
        png_filepath = os.path.join(shape_image_png_dir, png_filename)

        shape_image = shape_images[shape_name]

        is_valid = painter.draw_bongard_image(bongard_image=shape_image)
        assert is_valid, "Shape image {} is not valid!".format(shape_name)

        screen.update()

        # Save image
        screen.getcanvas().postscript(file=ps_filepath)
        Image.open(ps_filepath).resize((128, 128)).save(png_filepath)

        png_filepath_dict[shape_name] = png_filepath

        wn.clear()

    return png_filepath_dict


def generate_multiview_markdown(shape_image_png_dir, shape_names, line_or_arc_types,
                                markdown_filepath="./human_designed_shape_collections.md"):
    filepath_dict = OrderedDict()
    # Create filepath dictionary
    for shape_name in shape_names:
        image_filename = "{}.png".format(shape_name)
        filepath_dict[shape_name] = [os.path.join(shape_image_png_dir, line_or_arc_type, image_filename) for
                                     line_or_arc_type in line_or_arc_types]

    num_types = len(line_or_arc_types)
    line_template = "|{}" + "|" + "|".join(["{}"] * num_types) + "|" + "\n"
    header_line = line_template.format("shape", *line_or_arc_types)
    separator_line = line_template.format(*([":-:"] * (num_types + 1)))
    url_template = "<img src=\"{}\" width=\"50%\"/>"

    with open(markdown_filepath, "w") as fhand:

        fhand.write(header_line)
        fhand.write(separator_line)

        for shape_name in filepath_dict.keys():
            filepaths = filepath_dict[shape_name]
            urls = [url_template.format(filepath) for filepath in filepaths]
            content = [shape_name] + urls
            line = line_template.format(*content)
            fhand.write(line)

    return


def generate_shape_images(annotated_shape_table_filepath, shape_image_ps_dir, shape_image_png_dir, line_or_arc_type):

    screen = turtle.Screen()
    width, height = (800, 800)
    screen.setup(width=width, height=height)
    screen.screensize(width, height)
    # print('Screen size: ({}, {})'.format(screen.window_height(), screen.window_width()))
    screen.bgcolor("lightgrey")

    wn = turtle.Turtle()
    wn.pen(fillcolor="white", pencolor="black", pendown=False, pensize=8, speed=0)

    shape_images = generate_shape_collections(annotated_shape_table_filepath=annotated_shape_table_filepath, line_or_arc_type=line_or_arc_type)
    save_shape_images(shape_images=shape_images, screen=screen, wn=wn, shape_image_ps_dir=os.path.join(shape_image_ps_dir, line_or_arc_type), shape_image_png_dir=os.path.join(shape_image_png_dir, line_or_arc_type))

    return 


def main():
    annotated_shape_table_filepath = "../../data/human_designed_shapes.tsv"
    shape_image_ps_dir = "./images/shapes/ps"
    shape_image_png_dir = "./images/shapes/png"
    markdown_filepath = "./gallery.md"
    line_or_arc_types = ["normal", "zigzag", "arrow", "circle", "square", "triangle", "classic"]
    pool = Pool(processes=len(line_or_arc_types))

    jobs = []
    for line_or_arc_type in line_or_arc_types:
        job = pool.apply_async(func=generate_shape_images, args=(annotated_shape_table_filepath, shape_image_ps_dir, shape_image_png_dir, line_or_arc_type))
        jobs.append(job)
    pool.close()
    for job in jobs:
        job.get()

    annotation_dict = get_human_designed_shape_annotations(
        annotated_shape_table_filepath=annotated_shape_table_filepath)
    shape_names = annotation_dict.keys()
    generate_multiview_markdown(shape_image_png_dir=shape_image_png_dir, shape_names=shape_names,
                                line_or_arc_types=line_or_arc_types, markdown_filepath=markdown_filepath)

    # We don't need ps for gallery.
    shutil.rmtree(shape_image_ps_dir)

if __name__ == "__main__":
    main()
