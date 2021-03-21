# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

from .bongard import LineAction, ArcAction, OneStrokeShape, BongardImage, BongardProblem
from .bongard_painter import BongardImagePainter, BongardProblemPainter
from .bongard_sampler import BongardProblemSampler
from .util_funcs import get_human_designed_shape_annotations, get_attribute_sampling_candidates, \
    get_shape_super_classes, get_stampable_shapes
