from setuptools import setup, find_packages

setup(
    name='mnist_interactive',
    version='1.0.5',
    description='A library for experimenting with MNIST models using an interactive 28x28 grid.',
    author='Luca Lowndes',
    author_email='Luca@Lowndes.net',
    url='https://github.com/LucaLow/MNIST-Interactive-Model-Analyzer',
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    install_requires=[
        'tensorflow>=2.0',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)