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
"""

from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from .options import EdgeOptions


cdef class PyEdge:
    """ Python interface to Edge C-Structure"""

    def __cinit__(self):
        self.edge = <Edge*>malloc(sizeof(Edge))
        if self.edge is NULL:
            raise MemoryError("Failed to allocate memory for Edge")
        self.edge.nparameters = 0

    def __dealloc__(self):
        if self.edge is not NULL:
            if self.edge.source is not NULL:
                free(self.edge.source)
            if self.edge.target is not NULL:
                free(self.edge.target)
            if self.edge.type is not NULL:
                free(self.edge.type)
            if self.edge.parameters is not NULL:
                free(self.edge.parameters)
            free(self.edge)

    def __init__(self, source: str, target: str, edge_type: str, **kwargs):
        self.edge.source = strdup(source.encode('UTF-8'))
        self.edge.target = strdup(source.encode('UTF-8'))
        self.edge.type = strdup(edge_type.encode('UTF-8'))

    @classmethod
    def from_options(cls, edge_options: EdgeOptions):
        """ From edge options """
        source: str = edge_options.source
        target: str = edge_options.target
        edge_type: str = edge_options.type

        return cls(source, target, edge_type, **edge_options.parameters)

    # Property methods for source
    @property
    def source(self):
        if self.edge.source is NULL:
            return None
        return self.edge.source.decode('UTF-8')

    @property
    def target(self):
        if self.edge.target is NULL:
            return None
        return self.edge.target.decode('UTF-8')

    @property
    def type(self):
        if self.edge.type is NULL:
            return None
        return self.edge.type.decode('UTF-8')

    # Property methods for nparameters
    @property
    def nparameters(self):
        return self.edge.nparameters

    @property
    def parameters(self):
        """Generic accessor for parameters."""
        if not self.edge.parameters:
            raise ValueError("Edge parameters are NULL")
        if self.edge.nparameters == 0:
            raise ValueError("No parameters available")

        # The derived class should override this method to provide specific behavior
        raise NotImplementedError("Base class does not define parameter handling")
