from setuptools import setup, find_packages

setup(
    name='CodeInsight',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'networkx',
        'matplotlib',
        'requests'
    ],
    entry_points={
        'console_scripts': [
            'codeinsight=codeinsight.cli:main',
        ],
    },
)
