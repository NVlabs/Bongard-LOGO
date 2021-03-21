# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from bongard import LineAction, ArcAction, BongardImage, BongardProblem
import math
import numpy as np
import os
from PIL import Image
import turtle


class BongardShapePainter(object):

    def __init__(self, screen, wn, stamp_unit_distance=20, stamp_unit_arc_angle=15, zigzag_unit_distance=17,
                 zigzag_unit_arc_angle=12, x_range=(-360, 360), y_range=(-360, 360), base_scaling_factor=180,
                 magnifier_effect=False):

        """
        screen: turtle.Screen instance
        wn: turtle.tt.Turtle instance
        """

        self.magnifier_effect = magnifier_effect

        self.stamp_unit_distance = stamp_unit_distance
        self.stamp_unit_arc_angle = stamp_unit_arc_angle
        self.zigzag_unit_distance = zigzag_unit_distance
        self.zigzag_unit_arc_angle = zigzag_unit_arc_angle

        self.base_scaling_factor = base_scaling_factor
        self.base_length = self.base_scaling_factor
        self.base_radius = self.base_length / 2

        self.turtle_base_shape_size = {"arrow": 1.0, "circle": 0.75, "square": 0.65, "triangle": 0.8, "classic": 1.8}

        self.screen = screen
        self.wn = wn

        # We do not need to show turtle
        self.wn.hideturtle()

        self.x_range = x_range
        self.y_range = y_range
        self.x_range_accumulated = [None, None]
        self.y_range_accumulated = [None, None]

    def set_turtle_shape(self, turtle_shape):
        self.wn.shape(turtle_shape)

    def set_turtle_size(self, turtle_size=None):
        if turtle_size is None:
            turtle_size = self.turtle_base_shape_size[self.wn.shape()]
        self.wn.shapesize(turtle_size, outline=2.5)

    def set_turtle_locations(self, x, y, theta):
        self.wn.penup()
        self.wn.goto(x=x, y=y)
        self.wn.setheading(to_angle=theta)
        self.wn.pendown()

    def is_turtle_outside_range(self):
        """
        Judge if turtle has run out of range
        """

        cur_x, cur_y = self.wn.position()

        if cur_x < self.x_range[0] or cur_x > self.x_range[1]:
            return True
        if cur_y < self.y_range[0] or cur_y > self.y_range[1]:
            return True

        return False

    def set_accumulated_ranges(self):
        cur_x, cur_y = self.wn.position()

        if cur_x < self.x_range_accumulated[0]:
            self.x_range_accumulated[0] = cur_x
        if cur_x > self.x_range_accumulated[1]:
            self.x_range_accumulated[1] = cur_x

        if cur_y < self.y_range_accumulated[0]:
            self.y_range_accumulated[0] = cur_y
        if cur_y > self.y_range_accumulated[1]:
            self.y_range_accumulated[1] = cur_y

    def reset_accumulated_ranges(self, x_range_accumulated, y_range_accumulated):

        self.x_range_accumulated = x_range_accumulated
        self.y_range_accumulated = y_range_accumulated

    def draw(self, shape_actions, shape_action_scaling_factors, shape_start_coordinates, shape_start_orientation):
        """
        Return: Boolean value
        """
        self.reset_accumulated_ranges(x_range_accumulated=[shape_start_coordinates[0], shape_start_coordinates[0]],
                                      y_range_accumulated=[shape_start_coordinates[1], shape_start_coordinates[1]])

        self.set_turtle_locations(x=shape_start_coordinates[0], y=shape_start_coordinates[1],
                                  theta=shape_start_orientation)

        # print(len(shape_actions))
        for action, action_scaling_factor in zip(shape_actions, shape_action_scaling_factors):

            if action.turn_direction == "L":
                self.wn.left(action.turn_angle)
            elif action.turn_direction == "R":
                self.wn.right(action.turn_angle)
            else:
                print(action.turn_direction)
                raise Exception("Unsupported direction type!")

            if isinstance(action, LineAction):
                is_valid = self.draw_line(length=action.line_length * action_scaling_factor,
                                          line_type=action.line_type, scaling_factor=action_scaling_factor)
                if not is_valid:
                    return False
            elif isinstance(action, ArcAction):
                is_valid = self.draw_arc(angle=action.arc_angle, radius=action.arc_radius * action_scaling_factor,
                                         arc_type=action.arc_type, scaling_factor=action_scaling_factor)
                if not is_valid:
                    return False

        return True

    def draw_line(self, length, line_type, scaling_factor=None):

        dist = length

        # Zigzag
        if line_type == "zigzag":
            # Compute adjusted zigzag unit distance
            zigzag_angle = 60

            if self.magnifier_effect == True and scaling_factor is not None:
                # dist_per_step = self.zigzag_unit_distance / self.base_length * length
                dist_per_step = self.zigzag_unit_distance / self.base_scaling_factor * scaling_factor
                turtle_size = self.turtle_base_shape_size[self.wn.shape()] / self.base_scaling_factor * scaling_factor
                self.set_turtle_size(turtle_size=turtle_size)
            else:
                dist_per_step = self.zigzag_unit_distance
                self.set_turtle_size(turtle_size=None)

            num_zigzags = int(abs(dist) // dist_per_step)
            if abs(dist) / dist_per_step - abs(dist) // dist_per_step >= 0.5:
                num_zigzags += 1
            if num_zigzags == 0:
                num_zigzags = 1
            adjusted_zigzag_unit_distance = dist / num_zigzags
            for _ in range(num_zigzags):
                self.wn.left(zigzag_angle)
                self.wn.forward(adjusted_zigzag_unit_distance / 2)
                self.wn.right(zigzag_angle * 2)
                self.wn.forward(adjusted_zigzag_unit_distance)
                self.wn.left(zigzag_angle * 2)
                self.wn.forward(adjusted_zigzag_unit_distance / 2)
                self.wn.right(zigzag_angle)
        # Normal
        elif line_type == "normal":
            self.wn.forward(dist)
        # Stamp
        else:
            self.set_turtle_shape(turtle_shape=line_type)
            self.wn.penup()

            if self.magnifier_effect == True and scaling_factor is not None:
                # dist_per_step = self.stamp_unit_distance / self.base_length * length
                dist_per_step = self.stamp_unit_distance / self.base_scaling_factor * scaling_factor
                turtle_size = self.turtle_base_shape_size[self.wn.shape()] / self.base_scaling_factor * scaling_factor
                self.set_turtle_size(turtle_size=turtle_size)
            else:
                dist_per_step = self.stamp_unit_distance
                self.set_turtle_size(turtle_size=None)
            '''
            if abs(dist) > dist_per_step:
                num_stamps = int(abs(dist) // dist_per_step)
                arc_dist_per_step = abs(dist) / num_stamps
                self.wn.stamp()
                for _ in range(num_stamps):
                    self.wn.forward(arc_dist_per_step)
                    self.wn.stamp()
                # This is almost negligible
                self.wn.forward(dist - arc_dist_per_step * num_stamps)
                self.wn.stamp()
            else:
                self.wn.stamp()
                self.wn.forward(dist)
                self.wn.stamp()
            '''

            num_stamps = int(abs(dist) // dist_per_step)

            if abs(dist) / dist_per_step - abs(dist) // dist_per_step >= 0.5:
                num_stamps += 1
            if num_stamps == 0:
                num_stamps = 1

            arc_dist_per_step = abs(dist) / num_stamps
            self.wn.stamp()
            for _ in range(num_stamps):
                self.wn.forward(arc_dist_per_step)
                self.wn.stamp()
            # This is almost negligible
            self.wn.forward(dist - arc_dist_per_step * num_stamps)
            self.wn.stamp()

            self.wn.pendown()

        if self.is_turtle_outside_range():
            return False

        self.set_accumulated_ranges()

        return True

    def draw_small_arc(self, angle, radius, arc_type, scaling_factor=None):

        if abs(angle) > 90:
            raise Exception("Small arc is in a range of [-90, 90]!")

        if arc_type == "zigzag":

            # Compute adjusted zigzag unit arc
            zigzag_angle = 60

            '''
            if self.magnifier_effect == True:
                num_zigzags = int(abs(angle) // self.zigzag_unit_arc_angle)

                self.zigzag_unit_distance / (2 * math.pi * radius) * 360  / self.base_scaling_factor * scaling_factor
            else:
                num_zigzags = int(abs(angle) // self.zigzag_unit_arc_angle * scaling_factor // self.base_scaling_factor)

            #num_zigzags = int(abs(angle) // self.zigzag_unit_arc_angle)
            if abs(angle) / self.zigzag_unit_arc_angle - abs(angle) // self.zigzag_unit_arc_angle >= 0.5:
                num_zigzags += 1
            if num_zigzags == 0:
                num_zigzags = 1
            adjusted_zigzag_unit_angle = abs(angle) / num_zigzags
            '''

            if self.magnifier_effect:
                arc_per_step = self.zigzag_unit_distance / (
                            2 * math.pi * radius) * 360 / self.base_scaling_factor * scaling_factor
            else:
                arc_per_step = self.zigzag_unit_distance / (2 * math.pi * radius) * 360

            num_zigzags = int(abs(angle) // arc_per_step)

            if abs(angle) / arc_per_step - abs(angle) // arc_per_step >= 0.5:
                num_zigzags += 1
            if num_zigzags == 0:
                num_zigzags = 1
            adjusted_zigzag_unit_angle = abs(angle) / num_zigzags

            alpha = adjusted_zigzag_unit_angle / 2

            zigzag_side_length = 2 * radius * math.sin(alpha / 2 * math.pi / 180)

            if angle >= 0:
                for _ in range(num_zigzags):
                    self.wn.right(zigzag_angle - alpha / 2)
                    self.wn.forward(zigzag_side_length)
                    self.wn.left(180 - zigzag_angle)
                    self.wn.forward(zigzag_side_length)
                    self.wn.left(alpha)
                    self.wn.forward(zigzag_side_length)
                    self.wn.right(180 - zigzag_angle)
                    self.wn.forward(zigzag_side_length)
                    self.wn.left(zigzag_angle + alpha / 2)
            else:
                for _ in range(num_zigzags):
                    self.wn.right(zigzag_angle + alpha / 2)
                    self.wn.forward(-zigzag_side_length)
                    self.wn.left(180 - zigzag_angle)
                    self.wn.forward(-zigzag_side_length)
                    self.wn.right(alpha)
                    self.wn.forward(-zigzag_side_length)
                    self.wn.right(180 - zigzag_angle)
                    self.wn.forward(-zigzag_side_length)
                    self.wn.left(zigzag_angle - alpha / 2)

        elif arc_type == "normal":
            self.wn.circle(radius=radius, extent=angle)

        else:
            self.set_turtle_shape(turtle_shape=arc_type)
            self.wn.penup()

            if self.magnifier_effect and scaling_factor is not None:
                # arc_per_step = self.stamp_unit_arc_angle
                arc_per_step = self.stamp_unit_distance / (
                            2 * math.pi * radius) * 360 / self.base_scaling_factor * scaling_factor
                turtle_size = self.turtle_base_shape_size[self.wn.shape()] / self.base_scaling_factor * scaling_factor
                self.set_turtle_size(turtle_size=turtle_size)
            else:
                # arc_per_step = self.stamp_unit_arc_angle / (scaling_factor / self.base_scaling_factor)
                # arc_per_step = self.stamp_unit_arc_angle
                arc_per_step = self.stamp_unit_distance / (2 * math.pi * radius) * 360
                self.set_turtle_size(turtle_size=None)
            '''
            #num_stamps = int(abs(angle) // arc_per_step)
            if abs(angle) > arc_per_step:
                num_stamps = int(abs(angle) // arc_per_step)
                arc_ave = abs(angle) / num_stamps
                self.wn.stamp()
                if angle >= 0:
                    for _ in range(num_stamps):
                        self.wn.circle(radius, extent=arc_ave)
                        self.wn.stamp()
                    # This should be almost negligible
                    self.wn.circle(angle - arc_ave * num_stamps)
                    self.wn.stamp()
                else:
                    for _ in range(num_stamps):
                        self.wn.circle(radius, extent=-arc_ave)
                        self.wn.stamp()
                    # This should be almost negligible
                    self.wn.circle(angle + arc_ave * num_stamps)
                    self.wn.stamp()
            else:
                self.wn.stamp()
                self.wn.circle(radius, extent=angle)
                self.wn.stamp()
            '''

            num_stamps = int(abs(angle) // arc_per_step)

            if abs(angle) / arc_per_step - abs(angle) // arc_per_step >= 0.5:
                num_stamps += 1
            if num_stamps == 0:
                num_stamps = 1
            arc_ave = abs(angle) / num_stamps

            self.wn.stamp()
            if angle >= 0:
                for _ in range(num_stamps):
                    self.wn.circle(radius, extent=arc_ave)
                    self.wn.stamp()
                # This should be almost negligible
                self.wn.circle(angle - arc_ave * num_stamps)
                self.wn.stamp()
            else:
                for _ in range(num_stamps):
                    self.wn.circle(radius, extent=-arc_ave)
                    self.wn.stamp()
                # This should be almost negligible
                self.wn.circle(angle + arc_ave * num_stamps)
                self.wn.stamp()

            self.wn.pendown()

        if self.is_turtle_outside_range():
            return False

        self.set_accumulated_ranges()

        return True

    def draw_arc(self, angle, radius, arc_type, scaling_factor):

        if angle > 0:
            num_arc90 = int(angle // 90)
            angle_left = angle - num_arc90 * 90
            for _ in range(num_arc90):
                is_valid = self.draw_small_arc(angle=90, radius=radius, arc_type=arc_type,
                                               scaling_factor=scaling_factor)
                if not is_valid:
                    return False
            if angle_left > 0:
                is_valid = self.draw_small_arc(angle=angle_left, radius=radius, arc_type=arc_type,
                                               scaling_factor=scaling_factor)
                if not is_valid:
                    return False
        else:  # angle < 0
            num_arcn90 = int(abs(angle) // 90)
            angle_left = angle - num_arcn90 * (-90)
            for _ in range(num_arcn90):
                is_valid = self.draw_small_arc(angle=-90, radius=radius, arc_type=arc_type,
                                               scaling_factor=scaling_factor)
                if not is_valid:
                    return False
            if angle_left < 0:
                is_valid = self.draw_small_arc(angle=angle_left, radius=radius, arc_type=arc_type,
                                               scaling_factor=scaling_factor)
                if not is_valid:
                    return False

        return True


class BongardImagePainter(object):

    def __init__(self, screen, wn, stamp_unit_distance=25, stamp_unit_arc_angle=15, zigzag_unit_distance=20,
                 zigzag_unit_arc_angle=15, x_range=(-360, 360), y_range=(-360, 360), base_scaling_factor=180,
                 magnifier_effect=False):
        """
        screen: turtle.Screen instance
        wn: turtle.tt.Turtle instance
        """

        self.magnifier_effect = magnifier_effect

        self.stamp_unit_distance = stamp_unit_distance
        self.stamp_unit_arc_angle = stamp_unit_arc_angle
        self.zigzag_unit_distance = zigzag_unit_distance
        self.zigzag_unit_arc_angle = zigzag_unit_arc_angle

        self.base_scaling_factor = base_scaling_factor

        self.screen = screen
        self.wn = wn

        self.x_range = x_range
        self.y_range = y_range

        self.shapePainter = BongardShapePainter(self.screen, self.wn, self.stamp_unit_distance,
                                                self.stamp_unit_arc_angle, self.zigzag_unit_distance,
                                                self.zigzag_unit_arc_angle, self.x_range, self.y_range,
                                                base_scaling_factor=self.base_scaling_factor,
                                                magnifier_effect=self.magnifier_effect)

    def draw(self, actions, action_scaling_factors, start_coordinates, start_orientations):
        """
        Return: Boolean value
        """
        # Iterate over shapes
        for shape_actions, shape_action_scaling_factors, shape_start_coordinates, shape_start_orientation in zip(
                actions, action_scaling_factors, start_coordinates, start_orientations):
            is_valid = self.shapePainter.draw(shape_actions, shape_action_scaling_factors,
                                              shape_start_coordinates, shape_start_orientation)

            if not is_valid:
                return False

        return True

    def draw_bongard_image(self, bongard_image):

        assert isinstance(bongard_image, BongardImage), "bongard_image is not an instance of BongardImage!"

        is_valid = self.draw(actions=bongard_image.get_actions(),
                             action_scaling_factors=bongard_image.get_scaling_factors(),
                             start_coordinates=bongard_image.get_start_coordinates(),
                             start_orientations=bongard_image.get_start_orientations())

        return is_valid


class BongardProblemPainter(object):

    def __init__(self, stamp_unit_distance=20, stamp_unit_arc_angle=15, zigzag_unit_distance=17,
                 zigzag_unit_arc_angle=12, x_range=(-360, 360), y_range=(-360, 360), base_scaling_factor=180,
                 magnifier_effect=False, scaling_factors_range=(150, 220),
                 random_seed=0):

        self.screen = turtle.Screen()
        width, height = (800, 800)
        self.screen.setup(width=width, height=height)
        self.screen.screensize(width, height)
        # print('Screen size: ({}, {})'.format(screen.window_height(), screen.window_width()))
        self.screen.bgcolor("lightgrey")

        self.wn = turtle.Turtle()
        self.wn.pen(fillcolor="white", pencolor="black", pendown=False, pensize=8, speed=0)

        self.screen.tracer(0, 0)
        self.bongard_image_painter = BongardImagePainter(screen=self.screen, wn=self.wn,
                                                         stamp_unit_distance=stamp_unit_distance,
                                                         stamp_unit_arc_angle=stamp_unit_arc_angle,
                                                         zigzag_unit_distance=zigzag_unit_distance,
                                                         zigzag_unit_arc_angle=zigzag_unit_arc_angle, x_range=x_range,
                                                         y_range=y_range, base_scaling_factor=base_scaling_factor,
                                                         magnifier_effect=magnifier_effect)
        self.random_state = np.random.RandomState(random_seed)
        self.scaling_factors_range = scaling_factors_range

    def sample_start_coordinates_and_orientation(self, num_shapes=1, max_radius=250):
        # This method guarantees to produce valid coordinates and orientations

        assert num_shapes == 1 or num_shapes == 2, \
            "We only support one or two shapes in the image, but there are {} shapes!".format(num_shapes)

        start_coordinates_list = []
        start_orientation_list = []

        if num_shapes == 1:
            theta = self.random_state.uniform(0, 360)
            radius = self.random_state.uniform(max_radius)
            x = radius * math.cos(theta * np.pi / 180)
            y = radius * math.sin(theta * np.pi / 180)
            start_coordinates_list = [(x, y)]
            start_orientation_list = [theta]
        elif num_shapes == 2:
            # First shape
            theta_1 = self.random_state.uniform(0, 360)
            radius = self.random_state.uniform(0.5 * max_radius, max_radius)
            x_1 = radius * math.cos(theta_1 * np.pi / 180)
            y_1 = radius * math.sin(theta_1 * np.pi / 180)
            start_coordinates_list.append((x_1, y_1))
            start_orientation_list.append(theta_1)
            # Second shape
            # Monte Carlo sampling
            while True:
                x_2 = self.random_state.uniform(-max_radius * 1.2, max_radius * 1.2)
                y_2 = self.random_state.uniform(-max_radius * 1.2, max_radius * 1.2)
                if np.linalg.norm(np.array([x_2, y_2]) - np.array([x_1, y_1])) >= max_radius * 1.2 and abs(
                        x_2 - x_1) >= max_radius * 0.18 and abs(y_2 - y_1) >= max_radius * 0.18:
                    break
            theta_2 = self.random_state.uniform(theta_1 - 60, theta_1 + 60)

            start_coordinates_list = [(x_1, y_1), (x_2, y_2)]
            start_orientation_list = [theta_1, theta_2]

        return start_coordinates_list, start_orientation_list

    def save_bongard_images(self, bongard_images, image_label, bongard_problem_ps_dir, bongard_problem_png_dir):

        for i, bongard_image in enumerate(bongard_images):

            assert isinstance(bongard_image, BongardImage), "BongardImagePainter requires BongardImage for painting!"

            is_valid = False
            paint_trials = 0
            max_paint_trials = 300
            while not is_valid and paint_trials < max_paint_trials:
                self.wn.clear()
                num_shapes = bongard_image.get_num_shapes()
                start_coordinates, start_orientations = self.sample_start_coordinates_and_orientation(
                    num_shapes=num_shapes)
                bongard_image.set_start_coordinates(start_coordinates=start_coordinates)
                bongard_image.set_start_orientations(start_orientations=start_orientations)
                scaling_factors = [
                    self.random_state.uniform(low=self.scaling_factors_range[0], high=self.scaling_factors_range[1]) for
                    _ in range(num_shapes)]
                bongard_image.set_consistent_scaling_factors(scaling_factors=scaling_factors)
                is_valid = self.bongard_image_painter.draw_bongard_image(bongard_image=bongard_image)
                # assert is_valid == True, "Creating Bongard problem failed!"
                self.screen.update()
                paint_trials += 1

            if not is_valid:
                print('[warning] maximum paint trials have been reached but <is_valid=False>!')

            # Save image
            ps_filename = "{}.ps".format(i)
            png_filename = "{}.png".format(i)
            ps_dir = os.path.join(bongard_problem_ps_dir, image_label)
            png_dir = os.path.join(bongard_problem_png_dir, image_label)
            if not os.path.exists(ps_dir):
                os.makedirs(ps_dir)
            if not os.path.exists(png_dir):
                os.makedirs(png_dir)
            ps_filepath = os.path.join(ps_dir, ps_filename)
            png_filepath = os.path.join(png_dir, png_filename)
            self.screen.getcanvas().postscript(file=ps_filepath)
            Image.open(ps_filepath).resize((512, 512)).save(png_filepath)

            self.wn.clear()

    def create_bongard_problem(self, bongard_problem, bongard_problem_ps_dir, bongard_problem_png_dir):

        assert isinstance(bongard_problem, BongardProblem), "bongard_problem is not an instance of BongardProblem!"

        positive_bongard_images = bongard_problem.get_positive_bongard_images()
        negative_bongard_images = bongard_problem.get_negative_bongard_images()

        positive_label = "1"
        negative_label = "0"

        self.save_bongard_images(bongard_images=positive_bongard_images, image_label=positive_label,
                                 bongard_problem_ps_dir=bongard_problem_ps_dir,
                                 bongard_problem_png_dir=bongard_problem_png_dir)
        self.save_bongard_images(bongard_images=negative_bongard_images, image_label=negative_label,
                                 bongard_problem_ps_dir=bongard_problem_ps_dir,
                                 bongard_problem_png_dir=bongard_problem_png_dir)


'''

def main():

    import turtle

    screen = turtle.Screen()
    width, height = (800, 800)
    screen.setup(width=width, height=height)
    screen.screensize(width, height)
    print('Screen size: ({}, {})'.format(screen.window_height(), screen.window_width()))
    screen.bgcolor("lightgrey")

    screen.tracer(0, 0)

    wn = turtle.Turtle()
    wn.pen(fillcolor="white", pencolor='black', pendown=False, pensize=8, speed=10)

    l1 = LineAction(line_length=1.0, line_type="zigzag", turn_direction="R", turn_angle=90)
    l2 = LineAction(line_length=0.5, line_type="normal", turn_direction="R", turn_angle=90)
    l3 = LineAction(line_length=1.0, line_type="square", turn_direction="R", turn_angle=90)
    l4 = LineAction(line_length=0.5, line_type="circle", turn_direction="R", turn_angle=90)

    actions = [[l1, l2, l3, l4]]
    action_scaling_factors = [[200, 200, 200, 200]]
    start_coordinates = [(0,0)]
    start_orientations = [90]

    painter = BongardImagePainter(screen=screen, wn=wn)

    is_valid = painter.draw(actions=actions, action_scaling_factors=action_scaling_factors, 
                            start_coordinates=start_coordinates, start_orientations=start_orientations)

    print(is_valid)

    screen.update()

    turtle.done() 


if __name__ == "__main__":
    
    main()
'''
