import os
import pandas as pd

DATASET='modelnet10'

metadata = pd.DataFrame({'filename': pd.Series(dtype='str'),
                   'split': pd.Series(dtype='str'),
                   'path': pd.Series(dtype='str'),
                   'label': pd.Series(dtype='int'),
                   'label_str': pd.Series(dtype='str')},)

path = '/home/ubuntu/nndlproject/data/'+DATASET+'/ply/'

labels = {k:v for v,k in enumerate(os.listdir(path))}
i=0
for r, d, f in os.walk(path):
    for file in f:
        
        label,split = r.split('/')[-2:]
        new_row = {
            'filename' : file,
            'split' : split,
            'path' : r+'/'+file,
            'label' : labels[label],
            'label_str' : label
        }
        metadata.loc[i] = new_row

        i+=1
        
metadata.to_csv('/home/ubuntu/nndlproject/data/'+DATASET+'/metadata.csv')