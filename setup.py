import setuptools
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import Cython

directive_defaults = Cython.Compiler.Options.get_directive_defaults()

extensions = [
    Extension("farms_network.network_generator",
              ["farms_network/network_generator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.oscillator",
              ["farms_network/oscillator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.leaky_integrator",
              ["farms_network/leaky_integrator.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.neuron",
              ["farms_network/neuron.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    # Extension("farms_network.lif_danner_nap",
    #           ["farms_network/lif_danner_nap.pyx"],
    #           include_dirs=[numpy.get_include()],
    #           extra_compile_args=['-ffast-math', '-O3'],
    #           extra_link_args=['-O3']
    #           ),
    # Extension("farms_network.lif_danner",
    #           ["farms_network/lif_danner.pyx"],
    #           include_dirs=[numpy.get_include()],
    #           extra_compile_args=['-ffast-math', '-O3'],
    #           extra_link_args=['-O3']
    #           ),
    # Extension("farms_network.lif_daun_interneuron",
    #           ["farms_network/lif_daun_interneuron.pyx"],
    #           include_dirs=[numpy.get_include()],
    #           extra_compile_args=['-ffast-math', '-O3'],
    #           extra_link_args=['-O3']
    #           ),
    # Extension("farms_network.hh_daun_motorneuron",
    #           ["farms_network/hh_daun_motorneuron.pyx"],
    #           include_dirs=[numpy.get_include()],
    #           extra_compile_args=['-ffast-math', '-O3'],
    #           extra_link_args=['-O3']
    #           ),
    # Extension("farms_network.sensory_neuron",
    #           ["farms_network/sensory_neuron.pyx"],
    #           include_dirs=[numpy.get_include()],
    #           extra_compile_args=['-ffast-math', '-O3'],
    #           extra_link_args=['-O3']
    #           )
]

directive_defaults['linetrace'] = True,
directive_defaults['binding'] = True

setuptools.setup(
    name='farms_network',
    version='0.1',
    description='Module to generate, develop and visualize neural networks',
    url='https://gitlab.com/FARMSIM/farms_network.git',
    author='biorob-farms',
    author_email='biorob-farms@groupes.epfl.ch',
    license='MIT',
    packages=setuptools.find_packages(exclude=['tests*']),
    dependency_links=[
        'https://gitlab.com/FARMSIM/farms_pylog.git',
        'https://gitlab.com/FARMSIM/farms_dae.git'],
    install_requires=[
        'farms_dae',
        'numpy',
        'farms_pylog',
        'matplotlib',
        'networkx',
        'pydot'
    ],
    zip_safe=False,
    ext_modules=cythonize(extensions, annotate=True),
    package_data = {
        'farms_network': ['*.pxd'],
    },
)
