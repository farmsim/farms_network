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


from farms_core.array.array_cy cimport (DoubleArray1D, DoubleArray2D,
                                        IntegerArray1D)

include 'types.pxd'


cdef class NetworkDataCy:

    cdef:
        public NetworkStatesCy states
        public NetworkStatesCy derivatives
        public DoubleArray1D external_inputs
        public DoubleArray1D outputs
        public NetworkConnectivityCy connectivity
        public NetworkNoiseCy noise

        public NodeDataCy[:] nodes


cdef class NetworkStatesCy(DoubleArray1D):
    """ State array """

    cdef public UITYPEv1 indices


cdef class NetworkConnectivityCy:
    """ Network connectivity array """

    cdef:
        public DTYPEv1 weights
        public UITYPEv1 sources
        public UITYPEv1 indices


cdef class NetworkNoiseCy:
    """ Noise data array """

    cdef:
        public DTYPEv1 states
        public DTYPEv1 drift
        public DTYPEv1 diffusion
        public DTYPEv1 outputs


cdef class NodeDataCy:
    """ Node data """
    cdef:
        public DoubleArray2D states
        public DoubleArray2D derivatives
        public DoubleArray1D output
        public DoubleArray1D external_input
