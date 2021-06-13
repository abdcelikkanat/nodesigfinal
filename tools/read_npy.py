import sys
import numpy as np

file_path = sys.argv[1]


data = np.load(file_path)
print(data)
