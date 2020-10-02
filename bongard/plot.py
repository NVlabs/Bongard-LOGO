# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
from tqdm import tqdm
from PIL import Image
import uuid
from shutil import copyfile
import json
import random
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_bongard_problem(class_a_images, class_b_images, test_a_image, test_b_image, filepath):
    assert len(class_a_images) == 6, "Class A must have 6 images."
    assert len(class_b_images) == 6, "Class B must have 6 images."

    # https://matplotlib.org/tutorials/intermediate/gridspec.html#sphx-glr-tutorials-intermediate-gridspec-py
    fig = plt.figure(figsize=(15.9, 10), constrained_layout=False)
    grid_specs = fig.add_gridspec(2, 3, wspace=0.1, hspace=0.0, height_ratios=[1, 9], width_ratios=[1, 1, 0.5])

    ax = fig.add_subplot(grid_specs[0, 0])
    ax.text(x=0.5, y=0.5, s="A", fontsize=20, fontweight='bold', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(grid_specs[0, 1])
    ax.text(x=0.5, y=0.5, s="B", fontsize=20, fontweight='bold', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(grid_specs[0, 2])
    ax.text(x=0.5, y=0.5, s="Test", fontsize=20, fontweight='bold', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])

    for col in range(2):
        ax = fig.add_subplot(grid_specs[1, col])
        ax.set_xticks([])
        ax.set_yticks([])

    # Class A
    inner_grid = grid_specs[1, 0].subgridspec(3, 2, wspace=0.0, hspace=0.0)
    for row in range(3):
        for col in range(2):
            img = class_a_images[row * 2 + col]
            ax = fig.add_subplot(inner_grid[row, col], aspect='equal')
            ax.imshow(img, cmap='gray', origin='upper')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.axis('off')

    # Class B
    inner_grid = grid_specs[1, 1].subgridspec(3, 2, wspace=0.0, hspace=0.0)
    for row in range(3):
        for col in range(2):
            img = class_b_images[row * 2 + col]
            ax = fig.add_subplot(inner_grid[row, col], aspect='equal')
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    # Test
    inner_grid = grid_specs[1, 2].subgridspec(3, 1, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(inner_grid[0, 0])
    ax.imshow(test_a_image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(inner_grid[1, 0])
    ax.imshow(test_b_image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(filepath, bbox_inches='tight')

    plt.close()


def create_visualized_bongard_problem(bongard_problem_dir, bongard_problem_visualized_filepath):
    """
    Positive images in "1" means set A and negative images in "0" means set B,
    as stated in original Bongard problems
    """
    positive_images = []
    negative_images = []
    for i in range(7):
        img = mpimg.imread(os.path.join(bongard_problem_dir, "1", "{}.png".format(i)))
        positive_images.append(img)
    for i in range(7):
        img = mpimg.imread(os.path.join(bongard_problem_dir, "0", "{}.png".format(i)))
        negative_images.append(img)

    plot_bongard_problem(class_a_images=positive_images[:6], class_b_images=negative_images[:6],
                         test_a_image=positive_images[6], test_b_image=negative_images[6],
                         filepath=bongard_problem_visualized_filepath)
