""" Run script """

from argparse import ArgumentParser

from farms_core.io.yaml import read_yaml
from farms_network.core.network import PyNetwork
from farms_network.core.options import NetworkOptions
from tqdm import tqdm


def run_network(network_options):

    network = PyNetwork.from_options(network_options)
    network.setup_integrator(network_options)

    # data.to_file("/tmp/sim.hdf5")

    # Integrate
    N_ITERATIONS = network_options.integration.n_iterations
    TIMESTEP = network_options.integration.timestep

    inputs_view = network.data.external_inputs.array
    for iteration in tqdm(range(0, N_ITERATIONS), colour="green", ascii=" >="):
        inputs_view[:] = (iteration / N_ITERATIONS) * 1.0
        network.step()
        network.data.times.array[iteration] = iteration*TIMESTEP


def main():
    """ Main """

    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", "-c", dest="config_path", type=str, required=True
    )
    clargs = parser.parse_args()
    # run network
    options = NetworkOptions.from_options(read_yaml(clargs.config_path))
    run_network(options)


if __name__ == '__main__':
    main()
