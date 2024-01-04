
from setuptools import setup, find_packages

## README
with open("README.md", "r") as fh:
    long_description = fh.read()

## Requirements
with open("requirements.txt", "r") as r:
    requirements = [i.strip() for i in r.readlines()]

## Run Setup
setup(
    name="cce",
    version="0.0.1",
    author="Keith Harrigian",
    author_email="kharrigian@jhu.edu",
    description="Clinical Concept Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kharrigian/ml4h-clinical-bert",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
)