# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import numpy as np


class BasicAction(object):

    @staticmethod
    def normalize_line_length(line_length, line_length_max):

        # [0,line_length_max] -> [0,1]

        assert line_length >= 0, "line_length should not be negative!"
        normalized_line_length = line_length / line_length_max

        return normalized_line_length

    @staticmethod
    def denormalize_line_length(normalized_line_length, line_length_max):

        # [0,1] -> [0,line_length_max]

        assert normalized_line_length >= 0 and normalized_line_length <= 1, "normalized_line_length should not be in [0, 1]!"
        line_length = normalized_line_length * line_length_max

        return line_length

    @staticmethod
    def normalize_turn_angle(turn_direction, turn_angle):

        # L180 -> 180
        # R180 -> -180
        # [-180,180] -> [0,1]

        assert turn_angle <= 180 and turn_angle >= 0, "angle should be in [0, 180]!"

        if turn_direction == "L":
            normalized_turn_angle = (turn_angle + 180) / 360
        elif turn_direction == "R":
            normalized_turn_angle = (180 - turn_angle) / 360
        else:
            raise Exception("Unsupported direction!")

        return normalized_turn_angle

    @staticmethod
    def denormalize_turn_angle(normalized_turn_angle):

        # L180 -> 180
        # R180 -> -180
        # [0,1] -> [-180,180]

        assert normalized_turn_angle >= 0 and normalized_turn_angle <= 1, "normalized_turn_angle should be in [0, 1]!"

        if normalized_turn_angle >= 0.5:
            direction = "L"
            angle = normalized_turn_angle * 360 - 180
        else:
            direction = "R"
            angle = 180 - normalized_turn_angle * 360

        return direction, angle

    '''
    @staticmethod
    def normalize_arc_angle(arc_angle):

        # [-90,90] -> [0,1]

        assert arc_angle >= -90 and arc_angle <= 90, "arc_angle should not be in [-90, 90]!"
        normalized_arc_angle = (arc_angle + 90) / 180

        return normalized_arc_angle

    @staticmethod
    def denormalize_arc_angle(normalized_arc_angle):

        # [0,1] -> [-90,90]

        assert normalized_arc_angle >= 0 and normalized_arc_angle <= 1, "normalized_arc_angle should not be in [0, 1]!"
        arc_angle = normalized_arc_angle * 180 - 90

        return arc_angle
    '''

    @staticmethod
    def normalize_arc_angle(arc_angle):

        # [-360,360] -> [0,1]

        assert arc_angle >= -360 and arc_angle <= 360, "arc_angle should not be in [-360, 360]!"
        normalized_arc_angle = (arc_angle + 360) / 720

        return normalized_arc_angle

    @staticmethod
    def denormalize_arc_angle(normalized_arc_angle):

        # [0,1] -> [-360,360]

        assert normalized_arc_angle >= 0 and normalized_arc_angle <= 1, "normalized_arc_angle should not be in [0, 1]!"
        arc_angle = normalized_arc_angle * 720 - 360

        return arc_angle

    @staticmethod
    def get_action_type(action_string):

        return action_string.split("_")[0]


class ArcAction(BasicAction):

    def __init__(self, arc_angle, arc_type, turn_direction, turn_angle, arc_radius=0.5):

        super(ArcAction, self).__init__()

        self.arc_radius = arc_radius
        self.arc_angle = arc_angle
        self.arc_type = arc_type
        self.turn_direction = turn_direction
        self.turn_angle = turn_angle
        self.name = "arc"

    def export_to_action_string(self, arc_radius_normalizaton_factor=None):

        if arc_radius_normalizaton_factor is not None:
            normalized_arc_radius = BasicAction.normalize_line_length(line_length=self.arc_radius,
                                                                      line_length_max=arc_radius_normalizaton_factor)
        else:
            normalized_arc_radius = self.arc_radius

        normalized_arc_angle = BasicAction.normalize_arc_angle(arc_angle=self.arc_angle)

        normalized_turn_angle = BasicAction.normalize_turn_angle(turn_direction=self.turn_direction,
                                                                 turn_angle=self.turn_angle)

        action_string = "{}_{}_{:.3f}_{:.3f}-{:.3f}".format(self.name, self.arc_type, normalized_arc_radius,
                                                            normalized_arc_angle, normalized_turn_angle)

        return action_string

    @classmethod
    def import_from_action_string(cls, action_string, arc_radius_normalizaton_factor=None):
        """
        Parse an line action_string.
        For example, "arc_zigzag_0.5_0.7905-0.7500"
        """

        movement, turn_angle = action_string.split("-")
        turn_angle = float(turn_angle)
        action_name, arc_type, arc_radius, arc_angle = movement.split("_")
        arc_radius = float(arc_radius)
        arc_angle = float(arc_angle)

        if action_name != "arc":
            raise Exception("The action string imported is not an arc action string!")

        if arc_radius_normalizaton_factor is not None:
            denormalized_arc_radius = BasicAction.denormalize_line_length(normalized_line_length=arc_radius,
                                                                          line_length_max=arc_radius_normalizaton_factor)
        else:
            denormalized_arc_radius = arc_radius

        denormalized_arc_angle = BasicAction.denormalize_arc_angle(normalized_arc_angle=arc_angle)

        turn_direction, denormalized_turn_angle = BasicAction.denormalize_turn_angle(normalized_turn_angle=turn_angle)

        return cls(arc_angle=denormalized_arc_angle, arc_type=arc_type, turn_direction=turn_direction,
                   turn_angle=denormalized_turn_angle, arc_radius=denormalized_arc_radius)

    def __str__(self):

        return self.export_to_action_string()


class LineAction(BasicAction):

    def __init__(self, line_length, line_type, turn_direction, turn_angle):

        super(LineAction, self).__init__()

        self.line_length = line_length
        self.line_type = line_type
        self.turn_direction = turn_direction
        self.turn_angle = turn_angle
        self.name = "line"

    def export_to_action_string(self, line_length_normalization_factor=None):

        if line_length_normalization_factor is not None:
            normalized_line_length = BasicAction.normalize_line_length(line_length=self.line_length,
                                                                       line_length_max=line_length_normalization_factor)
        else:
            normalized_line_length = self.line_length

        normalized_turn_angle = BasicAction.normalize_turn_angle(turn_direction=self.turn_direction,
                                                                 turn_angle=self.turn_angle)

        action_string = "{}_{}_{:.3f}-{:.3f}".format(self.name, self.line_type, normalized_line_length,
                                                     normalized_turn_angle)

        return action_string

    @classmethod
    def import_from_action_string(cls, action_string, line_length_normalization_factor=None):
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

        return cls(line_length=denormalized_line_length, line_type=line_type, turn_direction=turn_direction,
                   turn_angle=denormalized_turn_angle)

    def __str__(self):

        return self.export_to_action_string()


class OneStrokeShape(object):

    def __init__(self, basic_actions, start_coordinates=None, start_orientation=None, scaling_factors=None):
        self.basic_actions = basic_actions
        self.start_coordinates = start_coordinates
        self.start_orientation = start_orientation
        self.scaling_factors = scaling_factors

    def __str__(self):
        return "[" + ", ".join(self.basic_actions) + "]" + ", " + "{}, {}, {}".format(self.start_coordinates[0],
                                                                                      self.start_coordinates[1],
                                                                                      self.start_orientation)

    def get_num_actions(self):
        return len(self.basic_actions)

    def get_actions(self):
        return self.basic_actions

    def get_action_string_list(self):
        return [action.export_to_action_string() for action in self.basic_actions]

    def set_start_coordinates(self, start_coordinates):
        self.start_coordinates = start_coordinates

    def set_start_orientation(self, start_orientation):
        self.start_orientation = start_orientation

    def set_scaling_factors(self, scaling_factors):
        self.scaling_factors = scaling_factors

    def set_consistent_scaling_factors(self, scaling_factor):
        assert np.isscalar(scaling_factor) == True, "Setting consistent scaling factors only requires one scalar!"

        self.scaling_factors = [scaling_factor] * len(self.basic_actions)

    def get_start_coordinates(self):
        return self.start_coordinates

    def get_start_orientation(self):
        return self.start_orientation

    def get_scaling_factors(self):
        return self.scaling_factors


class BongardImage(object):

    def __init__(self, one_stroke_shapes):
        self.one_stroke_shapes = one_stroke_shapes

    def get_num_shapes(self):

        return len(self.one_stroke_shapes)

    def get_actions(self):
        return [one_stroke_shape.get_actions() for one_stroke_shape in self.one_stroke_shapes]

    def get_action_string_list(self):
        return [one_stroke_shape.get_action_string_list() for one_stroke_shape in self.one_stroke_shapes]

    def get_start_coordinates(self):
        return [one_stroke_shape.get_start_coordinates() for one_stroke_shape in self.one_stroke_shapes]

    def set_start_coordinates(self, start_coordinates):

        assert isinstance(start_coordinates, list), "start_coordinates should be a list!"
        assert len(start_coordinates) == len(
            self.one_stroke_shapes), "The number of start_coordinates should be the same as the number of one_stroke_shapes!"
        for i in range(len(start_coordinates)):
            self.one_stroke_shapes[i].set_start_coordinates(start_coordinates=start_coordinates[i])

    def get_start_orientations(self):

        return [one_stroke_shape.get_start_orientation() for one_stroke_shape in self.one_stroke_shapes]

    def set_start_orientations(self, start_orientations):

        assert isinstance(start_orientations, list), "start_orientations should be a list!"
        assert len(start_orientations) == len(
            self.one_stroke_shapes), "The number of start_orientations should be the same as the number of one_stroke_shapes!"
        for i in range(len(start_orientations)):
            self.one_stroke_shapes[i].set_start_orientation(start_orientation=start_orientations[i])

    def get_scaling_factors(self):
        return [one_stroke_shape.get_scaling_factors() for one_stroke_shape in self.one_stroke_shapes]

    def set_scaling_factors(self, scaling_factors):

        assert isinstance(scaling_factors, list), "scaling_factors should be a list!"
        assert len(scaling_factors) == len(
            self.one_stroke_shapes), "The number of scaling_factors should be the same as the number of one_stroke_shapes!"
        for i in range(len(scaling_factors)):
            self.one_stroke_shapes[i].set_scaling_factors(scaling_factors=scaling_factors[i])

    def set_consistent_scaling_factors(self, scaling_factors):
        # Set the scaling factors consistent for each action in each individual shapes 

        assert isinstance(scaling_factors, list), "scaling_factors should be a list!"
        assert len(scaling_factors) == len(
            self.one_stroke_shapes), "The number of scaling_factors should be the same as the number of one_stroke_shapes!"
        for i in range(len(scaling_factors)):
            self.one_stroke_shapes[i].set_consistent_scaling_factors(scaling_factor=scaling_factors[i])


class BongardProblem(object):
    def __init__(self, positive_bongard_images, negative_bongard_images, problem_name=None, problem_description=None,
                 positive_rules=None, negative_rules=None):
        self.positive_bongard_images = positive_bongard_images
        self.negative_bongard_images = negative_bongard_images
        self.problem_name = problem_name
        self.problem_description = problem_description
        self.positive_rules = positive_rules
        self.negative_rules = negative_rules

    def get_problem_name(self):
        return self.problem_name

    def get_problem_description(self):
        return self.problem_description

    def get_positive_rules(self):
        return self.positive_rules

    def get_negative_rules(self):
        return self.negative_rules

    def get_positive_bongard_images(self):
        return self.positive_bongard_images

    def get_negative_bongard_images(self):
        return self.negative_bongard_images

    def get_actions(self):
        return [[positive_bongard_image.get_actions() for positive_bongard_image in self.positive_bongard_images],
                [negative_bongard_image.get_actions() for negative_bongard_image in self.negative_bongard_images]]

    def get_action_string_list(self):
        return [[positive_bongard_image.get_action_string_list() for positive_bongard_image in
                 self.positive_bongard_images],
                [negative_bongard_image.get_action_string_list() for negative_bongard_image in
                 self.negative_bongard_images]]

    def get_start_coordinates(self):
        return [
            [positive_bongard_image.get_start_coordinates() for positive_bongard_image in self.positive_bongard_images],
            [negative_bongard_image.get_start_coordinates() for negative_bongard_image in self.negative_bongard_images]]

    def get_start_orientations(self):
        return [[positive_bongard_image.get_start_orientations() for positive_bongard_image in
                 self.positive_bongard_images],
                [negative_bongard_image.get_start_orientations() for negative_bongard_image in
                 self.negative_bongard_images]]

    def get_scaling_factors(self):
        return [
            [positive_bongard_image.get_scaling_factors() for positive_bongard_image in self.positive_bongard_images],
            [negative_bongard_image.get_scaling_factors() for negative_bongard_image in self.negative_bongard_images]]
