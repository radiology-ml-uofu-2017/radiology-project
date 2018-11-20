#!/home/sci/ricbl/Documents/virtualenvs/dgx_python2_pytorch0.4/bin/python
#SBATCH --time=0-30:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1 # number of cluster nodes, abbreviated by -N
#SBATCH --mincpus=8
#SBATCH -o dgx_log/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e dgx_log/slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --mem=215G

import h5py
import numpy as np
import os
from h5df import Store
import pandas as pd
try:
    import cPickle as pickle
except:
    import _pickle as pickle 

print('oi1')
listImage = pd.read_pickle('./images_2012-20162017_256_256_prot3.pkl')['preprocessed']
nimages = listImage.shape[0]
a = np.empty([nimages, 3, 256, 256])
for index in range(nimages):
    a[index, :, :,:] = listImage.iloc[index].numpy()

print(a.shape)
b = a.reshape((a.shape[0], a.shape[1]*a.shape[2]*a.shape[3]))
print(b.shape)

print('oi2')
store = Store('images_2012-20162017_256_256_prot3.h5df', mode="w")
index = ['ind'+str(i) for i in range(b.shape[0])]
columns = ['col'+str(i) for i in range(b.shape[1])]
mkdf = lambda: pd.DataFrame(b, index=index, columns=columns)
store.put("/frames/1", mkdf())
h5f = h5py.File('./images_2012-20162017_256_256_prot3.h5', 'w')
h5f.create_dataset('dataset_1', data=a)
print('oi3')
h5f.close()