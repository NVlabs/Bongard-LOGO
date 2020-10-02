# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import setuptools

with open("README.md", "r") as fhand:
    long_description = fhand.read()

setuptools.setup(
    name="bongard",
    version="0.0.1",
    author="Lei Mao, Weili Nie",
    author_email="lmao@nvidia.com, wnie@nvidia.com",
    description="Bongard problem generation using Python turtle graphics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVlabs/Bongard-LOGO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
