from setuptools import setup, find_packages

setup(
    name='patternlib',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'joblib==1.4.2',
        'matplotlib==3.10.3',
        'networkx==3.4.1',
        'numpy==2.2.6',
        'optuna==4.3.0',
        'pandas==2.0.3',
        'scikit_learn==1.6.1',
        'scipy==1.15.3',
        'seaborn==0.13.2',
        'statsmodels==0.14.4',
        'torch==2.7.0',
        'torch_geometric==2.6.1',
        'tqdm==4.66.5'
    ],
    entry_points={
        'console_scripts': [
            'pattern = pattern.main:main',
            'pattern-generate = pattern.scripts.generate_registries:main'
        ],
    },
    package_data={
        'pattern': [
            'config/registries/*.json'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)