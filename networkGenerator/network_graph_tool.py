"""Graph Tool Tut."""


from graph_tool.all import *

G = Graph()

v1 = G.add_vertex()
v2 = G.add_vertex()

e = G.add_edge(v1, v2)

# Arbitrary python object.
prop = G.new_vertex_property("object")

import numpy as np
s = {'hello': np.sin}
prop[G.vertex(0)] = s


G.save("my_graph.xml")
