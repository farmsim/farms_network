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

Header for Edge Base Struture.

"""


cdef struct EdgeCy:

    #
    char* source                # Source node
    char* target                # Target node
    char* type                  # Type of connection

    # Edge parameters
    unsigned int nparameters
    void* parameters



cdef class Edge:
    """ Python interface to Edge C-Structure"""

    cdef:
        EdgeCy* c_edge
