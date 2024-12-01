"""

-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------

Header for Node Base Struture.

"""

from .edge cimport Edge


cdef struct Node:
    # Generic parameters
    unsigned int nstates        # Number of state variables in the node.
    unsigned int nparameters    # Number of parameters for the node.
    unsigned int ninputs        # Number of inputs

    char* model_type            # Type of the model (e.g., "empty").
    char* name                  # Unique name of the node.

    bint is_statefull              # Flag indicating whether the node is stateful. (ODE)

    # Parameters
    void* parameters            # Pointer to the parameters of the node.

    # Functions
    void ode(
        double time,
        double* states,
        double* derivatives,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        double noise,
        Node* node,
        Edge** edges,
    ) noexcept

    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        Node* node,
        Edge** edges,
    ) noexcept


cdef:
    void ode(
        double time,
        double* states,
        double* derivatives,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        double noise,
        Node* node,
        Edge** edges,
    ) noexcept
    double output(
        double time,
        double* states,
        double external_input,
        double* network_outputs,
        unsigned int* inputs,
        double* weights,
        Node* node,
        Edge** edges,
    ) noexcept


cdef class PyNode:
    """ Python interface to Node C-Structure"""

    cdef:
        Node* node
