DATASET='modelnet10'
ROOT_DIR='/home/ubuntu/nndl-project/data/' # where datasets are

import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.join(ROOT_DIR,DATASET)
os.chdir(ROOT_DIR)
file_formats = np.array(os.listdir())[[os.path.isdir(s) for s in os.listdir()]]

file_format_column = []
label_column = []
label_id_column = []
orientation_class_column = []
path_column = []
split_column = []

for file_format in file_formats:
    # go into file format dir
    os.chdir(os.path.join(ROOT_DIR,file_format))
    # list every label
    labels = os.listdir()
    for label in labels:
        if label == '.DS_Store': continue
        # go into the label dir
        os.chdir(os.path.join(ROOT_DIR,file_format,label)) 
        splits = os.listdir()
        for split in splits:
            if split == '.DS_Store': continue
            # go into split dir
            os.chdir(os.path.join(ROOT_DIR,file_format,label,split)) 
            
            # if not npy, no orientation class
            if file_format != 'npy': 
                orientation_class = None
                file_names = os.listdir()
                for file_name in file_names:
                    if file_name == '.DS_Store': continue
                    fname,ext = file_name.split('.')
                    label_id = fname.split('_')[1]
                    path = os.path.join(ROOT_DIR,file_format,label,split,file_name)

                    # add entry
                    file_format_column.append(file_format)
                    label_column.append(label)
                    label_id_column.append(label_id)
                    orientation_class_column.append(orientation_class)
                    path_column.append(path)
                    split_column.append(split)
            
            else:

                orientation_classes = os.listdir()

                for orientation_class in orientation_classes:
                    if orientation_class == '.DS_Store': continue
                    # go into orientation class dir
                    os.chdir(os.path.join(ROOT_DIR,file_format,label,split,orientation_class)) 

                    file_names = os.listdir()

                    for file_name in file_names:
                        if file_name == '.DS_Store': continue
                        fname,ext = file_name.split('.')
                        label_id = fname.split('_')[1]
                        path = os.path.join(ROOT_DIR,file_format,label,split,orientation_class,file_name)

                        # add entry
                        file_format_column.append(file_format)
                        label_column.append(label)
                        label_id_column.append(label_id)
                        orientation_class_column.append(orientation_class)
                        path_column.append(path)
                        split_column.append(split)

metadata = pd.DataFrame({
    'file_format' : file_format_column,
    'split' : split_column,
    'label' : label_column,
    'orientation_class' : orientation_class_column,
    'label_id' : label_id_column,
    'path' : path_column})

metadata.set_index(pd.RangeIndex(len(metadata),name='index'))
metadata['label_int']=pd.factorize(metadata['label'])[0]
metadata['orientation_class_id']=pd.factorize(metadata['label']+metadata['orientation_class'].astype(str))[0]
metadata.to_csv(os.path.join(ROOT_DIR,'metadata.csv'))

# rereading cause im stupid
metadata=pd.read_csv(os.path.join(ROOT_DIR,'metadata.csv'),index_col=0).set_index('file_format')
metadata['orientation_class']=metadata['orientation_class'].fillna(-1).astype(int)
orientation_classes=metadata.loc['npy'].groupby(['label','orientation_class']).count().reset_index()[['label','orientation_class']]
orientation_classes[['rot_x','rot_y']]=0
orientation_classes['rot_z'] = orientation_classes['orientation_class']*90

orientation_classes.to_csv(os.path.join(ROOT_DIR,'orientation_classes.csv'))