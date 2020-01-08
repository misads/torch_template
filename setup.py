import re
import setuptools
from setuptools import setup

with open('torch_template/version.py') as fid:
    try:
        __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
    except:
        raise ValueError("could not find version number")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torch-template',
    version=__version__,
    entry_points={
        'console_scripts': [
            'tt-new = torch_template.__main__:new_project',
        ]
    },
    description='Torch_Template - A PyTorch template with commonly used models and tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/misads/torch_template',
    author='Haoyu Xu',
    author_email='xuhaoyu@tju.edu.cn',
    license='MIT',
    install_requires=[
        "torch",
        "numpy",
        "torchvision",
        "tensorboardX",
    ],
    packages=setuptools.find_packages(exclude=["torch_template.templates"]),  # ['torch_template', 'torch_template/utils']
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.5',
)
