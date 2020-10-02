# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT


class BongardProblemSampler(object):

    def __init__(self, num_positive_examples=7, num_negative_examples=7):
        self.num_positive_examples = num_positive_examples
        self.num_negative_examples = num_negative_examples

    def sample(self, *args):
        raise NotImplementedError

