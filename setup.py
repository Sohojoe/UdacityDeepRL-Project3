from setuptools import setup, find_packages
import sys

if not (sys.version.startswith('3.5') or sys.version.startswith('3.6')):
    raise Exception('Only Python 3.5 and 3.6 are supported')

setup(name='baby_rl',
      packages=[package for package in find_packages()
                if package.startswith('baby_rl')],
      install_requires=[],
      description="Reinforcement Learning Algorithms in PyTorch",
      author="Joe Booth",
      url='https://github.com/Sohojoe/baby_rl',
      author_email="joe@joebooth.com",
      version="0.1")