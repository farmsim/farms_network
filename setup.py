from setuptools import setup, dist, find_packages
from setuptools.extension import Extension

from farms_container import get_include

dist.Distribution().fetch_build_eggs(['numpy'])
import numpy

dist.Distribution().fetch_build_eggs(['Cython>=0.15.1'])
from Cython.Build import cythonize
from Cython.Compiler import Options


Options.docstrings = True
Options.fast_fail = True
Options.annotate = True
Options.warning_errors = True
Options.profile = False

# directive_defaults = Cython.Compiler.Options.get_directive_defaults()

extensions = [
    Extension("farms_network.network_generator",
              ["farms_network/network_generator.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.oscillator",
              ["farms_network/oscillator.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.hopf_oscillator",
              ["farms_network/hopf_oscillator.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.morphed_oscillator",
              ["farms_network/morphed_oscillator.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.leaky_integrator",
              ["farms_network/leaky_integrator.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.neuron",
              ["farms_network/neuron.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.lif_danner_nap",
              ["farms_network/lif_danner_nap.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.lif_danner",
              ["farms_network/lif_danner.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.lif_daun_interneuron",
              ["farms_network/lif_daun_interneuron.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.hh_daun_motorneuron",
              ["farms_network/hh_daun_motorneuron.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.sensory_neuron",
              ["farms_network/sensory_neuron.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.fitzhugh_nagumo",
              ["farms_network/fitzhugh_nagumo.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.matsuoka_neuron",
              ["farms_network/matsuoka_neuron.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              ),
    Extension("farms_network.morris_lecar",
              ["farms_network/morris_lecar.pyx"],
              include_dirs=[numpy.get_include(), get_include()],
              extra_compile_args=['-ffast-math', '-O3'],
              extra_link_args=['-O3']
              )
]

setup(
    name='farms_network',
    version='0.1',
    description='Module to generate, develop and visualize neural networks',
    url='https://gitlab.com/FARMSIM/farms_network.git',
    author="Jonathan Arreguit  & Shravan Tata Ramalingasetty",
    author_email='biorob-farms@groupes.epfl.ch',
    license='Apache-2.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'farms_pylog @ git+https://gitlab.com/FARMSIM/farms_pylog.git',
        'tqdm',
        'matplotlib',
        'networkx',
        'pydot',
        'scipy'
    ],
    zip_safe=False,
    ext_modules=cythonize(
        extensions, include_path=[numpy.get_include()] + [get_include()]
    ),
    package_data={
        'farms_network': ['*.pxd'],
        'farms_container': ['*.pxd'],
    },
)
