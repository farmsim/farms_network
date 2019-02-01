import setuptools

setuptools.setup(
    name='farms_network_generator',
    version='0.1',
    description='Module to generate, develop and visualize neural networks',
    url='https://gitlab.com/FARMSIM/farms_network_generator.git',
    author='biorob-farms',
    author_email='biorob-farms@groupes.epfl.ch',
    license='MIT',
    packages=setuptools.find_packages(exclude=['tests*']),
    dependency_links=[
        'https://gitlab.com/FARMSIM/farms_pylog.git'],
    install_requires=[
        'numpy',
        'casadi',
        'farms_pylog',
        'matplotlib',
        'networkx',
        'collections',
    ],
    zip_safe=False
)
