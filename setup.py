# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import setuptools

with open("README.md", "r") as fhand:
    long_description = fhand.read()

setuptools.setup(
    name="bongard",
    version="0.0.2",
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
    install_requires=[
        "numpy>=1.18.3",
        "pillow>=7.1.2",
        "matplotlib>=3.2.1",
        "tqdm>=4.46.0",
        "pandas==1.0.5",
    ],
)
