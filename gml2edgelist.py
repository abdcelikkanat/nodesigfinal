import os
import sys
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.io as sio

dataset_name = "youtube_new"
base_folder = "../NodeSketch/graphs/"
input_path = os.path.join(base_folder, dataset_name + ".gml")
output_path = os.path.join(base_folder, dataset_name + ".edgelist")

g = nx.read_gml(input_path)
N = g.number_of_nodes()

with open(output_path, 'w') as fin:
	for edge in g.edges():
		fin.write("{} {}\n".format(edge[0], edge[1]))
		


