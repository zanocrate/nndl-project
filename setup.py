from setuptools import setup, find_packages

setup(
    name='pantheon',
    version='0.0.1',
    description='Package for NNDL project',
    packages=find_packages(include=['src', 'utils'])
)