from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="transformermpc",
    version="0.1.6",
    author="Vrushabh Zinage, Ahmed Khalil, Efstathios Bakolas",
    author_email="vrushabh.zinage@gmail.com",  
    description="Accelerating Model Predictive Control via Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vrushabh27/transformermpc",  
    packages=[
        'transformermpc',
        'transformermpc.data',
        'transformermpc.models',
        'transformermpc.utils',
        'transformermpc.training',
        'transformermpc.demo',
    ],
    package_dir={'transformermpc': 'transformermpc'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
) 