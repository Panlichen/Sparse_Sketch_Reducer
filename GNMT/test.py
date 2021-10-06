import numpy as np
import os 
from scipy import sparse
send_path = os.path.join(os.path.dirname(__file__), '1615892328.6030278_sendbuffer.npy')
recieve_path = os.path.join(os.path.dirname(__file__), '1615892330.9899693_grad.npy')

send = np.load(send_path)
recieve = np.load(recieve_path)
send_sparse = sparse.coo_matrix(send)
recieve_sparse = sparse.coo_matrix(recieve)
print(send_sparse.nnz, recieve_sparse.nnz)
print(set(send_sparse.row), set(recieve_sparse.row))