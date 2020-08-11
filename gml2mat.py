import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.io as sio

dataset_name = "./citeseer_undirected"
dataset_path = "./" + dataset_name + ".gml"

g = nx.read_gml(dataset_path)
N = g.number_of_nodes()


row = np.array([node for node in range(N) for nb in nx.neighbors(g, str(node))  ] )
col = np.array([nb for node in range(N) for nb in nx.neighbors(g, str(node)) ])
data = np.ones(row.shape)
M = csr_matrix((data, (row, col)), shape=(N, N)).toarray()


mdict={'network': M}

sio.savemat(file_name="./"+dataset_name+".mat", mdict=mdict, format='4')


