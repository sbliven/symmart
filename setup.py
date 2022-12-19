#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', "numpy", "pillow", "matplotlib","scipy","sympy"]

setup_requirements = []

test_requirements = ["tox", "flake8", "coverage", "pytest>=3", "pytest-runner>=5", "black", "mypy", "isort",]

doc_requirements = ["Sphinx>1.8"]

dev_requirements = ["bump2version", "twine"]

setup(
    author="Spencer Bliven",
    author_email='spencer.bliven@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Art from symmetric transformations in the spirit of Frank A. Farris",
    entry_points={
        'console_scripts': [
            'symmart=symmart.cli:main',
        ],
    },
    install_requires=requirements,
    extras_require={
        "tests": test_requirements,
        "docs": doc_requirements,
        "dev": dev_requirements + test_requirements + doc_requirements,
    },
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='symmart',
    name='symmart',
    packages=find_packages(include=['symmart', 'symmart.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sbliven/symmart',
    version='0.1.0',
    zip_safe=False,
)
