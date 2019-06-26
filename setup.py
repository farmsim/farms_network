import setuptools
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import Cython

directive_defaults = Cython.Compiler.Options.get_directive_defaults()

extensions = [
    Extension("farms_network_generator.network_generator",
              ["farms_network_generator/network_generator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network_generator.oscillator",
              ["farms_network_generator/oscillator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network_generator.leaky_integrator",
              ["farms_network_generator/leaky_integrator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network_generator.neuron",
              ["farms_network_generator/neuron.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network_generator.lif_danner_nap",
              ["farms_network_generator/lif_danner_nap.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network_generator.lif_danner",
              ["farms_network_generator/lif_danner.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network_generator.lif_daun_interneuron",
              ["farms_network_generator/lif_daun_interneuron.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network_generator.hh_daun_motorneuron",
              ["farms_network_generator/hh_daun_motorneuron.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              )]

directive_defaults['linetrace'] = True,
directive_defaults['binding'] = True

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
        'https://gitlab.com/FARMSIM/farms_pylog.git',
        'https://gitlab.com/FARMSIM/farms_dae_generator.git'],
    install_requires=[
        'farms_dae_generator',
        'numpy',
        'casadi',
        'farms_pylog',
        'matplotlib',
        'networkx',
        'pydot'
    ],
    zip_safe=False,
    ext_modules=cythonize(extensions, annotate=True)
)
