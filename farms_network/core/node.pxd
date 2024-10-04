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


cdef packed struct Node:
    # Generic parameters
    unsigned int nstates        # Number of state variables in the node.
    unsigned int nparameters    # Number of parameters for the node.
    unsigned int ninputs        # Number of inputs to the node.

    char* model_type            # Type of the model (e.g., "empty").
    char* name                  # Unique name of the node.

    bint statefull              # Flag indicating whether the node is stateful. (ODE)

    # Parameters
    void* parameters            # Pointer to the parameters of the node.

    # Functions
    void ode(
        double time,
        double[:] states,
        double[:] derivatives,
        double usr_input,
        double[:] inputs,
        double[:] weights,
        double[:] noise,
        Node node
    ) noexcept

    double output(
        double time,
        double[:] states,
        double usr_input,
        double[:] inputs,
        double[:] weights,
        double[:] noise,
        Node node
    ) noexcept


cdef:
    void ode(
        double time,
        double[:] states,
        double[:] derivatives,
        double usr_input,
        double[:] inputs,
        double[:] weights,
        double[:] noise,
        Node node
    ) noexcept
    double output(
        double time,
        double[:] states,
        double usr_input,
        double[:] inputs,
        double[:] weights,
        double[:] noise,
        Node node
    ) noexcept


cdef class PyNode:
    """ Python interface to Node C-Structure"""

    cdef:
        Node* node