#!/usr/bin/env python3

"""Setup."""

from setuptools import find_packages, setup

setup(
    name="seq_jax",
    version="0.1.0",
    description="Multimodal sequence modeling in JAX",
    packages=find_packages(),
    python_requires=">=3.12",
)
