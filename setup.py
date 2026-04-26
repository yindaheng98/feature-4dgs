from setuptools import find_packages, setup

import os


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()


pypi_build = os.environ.get("PYPI_BUILD", "").lower() in {"1", "true", "yes", "on"}


setup(
    name="feature_4dgs",
    version="1.0.1",
    author="yindaheng98",
    author_email="yindaheng98@gmail.com",
    url="https://github.com/yindaheng98/feature-4dgs",
    description="Packaged Python training code for sequence-aware Feature 3D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    install_requires=[
        "feature-3dgs",
    ] + ([
        "feature-3dgs @ git+https://github.com/yindaheng98/feature-3dgs.git@main",
    ] if not pypi_build else []),
)
