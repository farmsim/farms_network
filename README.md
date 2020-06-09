# FARMS NETWORK


## AUTHOR: Shravan Tata Ramalingasetty


## EMAIL: shravantr@gmail.com


## Description :

   This respository contains the necessary components to generate, integrate,
analyze and visualize neural network models. Currently the following neuron models are implemented :

-   lif\_danner
-   lif\_danner\_nap
-   lif\_daun\_interneuron
-   hh\_daun\_motorneuron
-   sensory\_neuron
-   leaky\_integrator
-   oscillator
-   morphed\_oscillator
-   fitzhugh\_nagumo
-   matsuoka\_neuron
-   morris\_lecar


# Installation


## Requirements

-   Python 2/3
-   Cython
-   pip
-   tqdm
-   numpy
-   matplotlib
-   networkx
-   pydot
-   ddt
-   scipy
-   farms\_pylog
-   farms\_container


## Steps for local install

The master branch is only supports Python 3. For Python 2 installation jump XXXXX


### For user installation

    pip install git+https://gitlab.com/farmsim/farms_network.git#egg=farms_network


### For developer installation

    git clone https://gitlab.com/farmsim/farms_network.git#egg=farms_network PATH_TO_THE_DIRECTORY
    cd PATH_TO_THE_DIRECTORY
    pip install -e . --user


### For Python 2 user installation

    pip install git+https://gitlab.com/farmsim/farms_network.git@python-2.7#egg=farms_network


### For Python 2 developer installation

    git clone https://gitlab.com/farmsim/farms_network.git@python-2.7#egg=farms_network PATH_TO_THE_DIRECTORY
    cd PATH_TO_THE_DIRECTORY
    pip install -e . --user


## Example

![img](./figures/danner_network.png "Danner mouse network")
