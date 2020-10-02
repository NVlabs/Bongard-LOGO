# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import pandas as pd
import collections


def get_human_designed_shape_annotations(annotated_shape_table_filepath):
    """
    Get a dictionary of action annotations for each human designed shape.
    """

    annotation_dict = collections.OrderedDict()

    df = pd.read_csv(annotated_shape_table_filepath, header=0, delimiter="\t")

    # Iterate over the rows of a small table

    for idx in range(len(df)):

        row = df.iloc[idx]
        shape_name = row["shape name"]
        shape_function_name = row["shape function name"]
        super_class = row["super class"]
        base_actions = row["set of base actions"]
        turn_angles = row["turn angles"]

        base_action_list = [base_action_string.strip() for base_action_string in base_actions.split(",")]
        turn_angle_list = [turn_angle_string.strip() for turn_angle_string in turn_angles.split("--")]

        assert len(base_action_list) == len(turn_angle_list), \
            "The number of base action and angles should be the same for shape {}!".format(shape_name)

        base_action_func_names = []
        base_action_func_parameters = []
        for base_action in base_action_list:

            base_action_components = base_action.split("_")
            base_action_name = base_action_components[0]
            base_action_arguments = base_action_components[1:]
            for i in range(len(base_action_arguments)):
                base_action_arguments[i] = float(base_action_arguments[i])

            if base_action_name.lower() == "line":
                base_action_func_names.append("line")
                base_action_func_parameters.append(base_action_arguments)
            elif base_action_name.lower() == "arc":
                base_action_func_names.append("arc")
                base_action_func_parameters.append(base_action_arguments)
            else:
                raise Exception("Unsupported base action {}!".format(base_action))

        directions = [angle.strip()[0] for angle in turn_angle_list]
        angles = [float(angle.strip()[1:]) for angle in turn_angle_list]

        assert len(base_action_func_names) == len(base_action_func_parameters) == len(angles) == len(directions)

        num_actions_computed = len(base_action_func_names)

        annotation_dict[shape_function_name] = (shape_name, super_class, num_actions_computed,
                                                base_action_func_names, base_action_func_parameters,
                                                directions, angles)

    return annotation_dict


def get_attribute_sampling_candidates(attribute_table_filepath, min_num_positives=0, min_num_negatives=0):
    """
    Key: primitive or advanced attribute, string
    Values: (positive_function_names, negative_function_names) (list of strings, list of strings)
    """

    df = pd.read_csv(attribute_table_filepath, header=0, delimiter="\t")

    # Convert values to integer
    for attribute in df.columns.values.tolist()[2:]:
        df[attribute] = df[attribute].astype(int)

    # Based on the primitive attributes, we create advanced attributes
    def label_symmetric_transposed(row):
        if row["symmetric"] == 1 and row["self_transposed"] == 1:
            return 1
        elif row["symmetric"] == -1 or row["self_transposed"] == -1:
            return -1
        elif row["symmetric"] == 0 or row["self_transposed"] == 0:
            return 0
        else:
            return -1

    df["symmetric_transposed"] = df.apply(lambda row: label_symmetric_transposed(row), axis=1)

    header_list = df.columns.values.tolist()
    shape_function_names = df["shape function name"]
    attribute_list = header_list[3:]
    num_attributes = len(attribute_list)
    num_shapes = df.shape[0]

    valid_attribute_list = []

    for attribute in attribute_list:
        value_counts = df[attribute].value_counts()

        if value_counts[1] >= min_num_positives and value_counts[0] >= min_num_negatives:
            valid_attribute_list.append(attribute)

    attribute_sampling_candidate_dict = {}

    for attribute in valid_attribute_list:
        positive_indices = df[df[attribute] == 1].index.tolist()
        negative_indices = df[df[attribute] == 0].index.tolist()
        positive_shape_function_names = shape_function_names[positive_indices].tolist()
        negative_shape_function_names = shape_function_names[negative_indices].tolist()
        attribute_sampling_candidate_dict[attribute] = (positive_shape_function_names, negative_shape_function_names)

    return attribute_sampling_candidate_dict


def get_shape_super_classes(annotated_shape_table_filepath):
    """
    Get a dictionary from the shape_actions csv table.
    Key: function name, string
    Values: whether a shape could be represented using stamps, bool
    """

    df = pd.read_csv(annotated_shape_table_filepath, header=0, delimiter="\t")

    shape_super_class_dict = pd.Series(df["super class"].values.astype(str),
                                       index=df["shape function name"]).to_dict()

    return shape_super_class_dict


def get_stampable_shapes(attribute_table_filepath):
    """
    Get a dictionary from the attribute csv table.
    Key: function name, string
    Values: whether a shape could be represented using stamps, bool
    """

    df = pd.read_csv(attribute_table_filepath, header=0, delimiter="\t")

    stampable_shape_dict = pd.Series(df["if stamp"].values.astype(bool),
                                     index=df["shape function name"]).to_dict()

    return stampable_shape_dict
