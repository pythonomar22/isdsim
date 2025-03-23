from setuptools import setup, find_packages

setup(
    name="isd-lib",
    version="0.1.0",
    description="A library for simulating classical Information Set Decoding (ISD) algorithms",
    author="ISD Developer",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pytest",
        "pandas",
        "tqdm",
        "jupyter"
    ],
    python_requires=">=3.6",
) 