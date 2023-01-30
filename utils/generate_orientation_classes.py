# simple script to generate rotation classes for objects.
# as in the paper, rotations are only around the z axis
# this works by defining a number of orientation classes, 
# then genrating corresponding z axis rotation. ex: 4 classes -> 0,90,180,270 degrees 

import pandas as pd

destination_path = '/home/ubuntu/nndl-project/data/modelnet10/orientation_classes.csv'

# this part is for grabbing label->int association
metadata_path = '/home/ubuntu/nndl-project/data/modelnet10/metadata.csv'
metadata = pd.read_csv(metadata_path,index_col=0)
int_to_label = dict(metadata.groupby(['label','label_str']).groups.keys())

# example: 4 classes for each label
# edit this dictionary to initialize different number of orientaiton classes
orientation_classes = pd.DataFrame(
    {
        'label' : [k for k in int_to_label.keys()],
        'label_str' : [v for v in int_to_label.values()],
        'orientation_class' : [[i for i in range(4)] for j in int_to_label.keys()], # 4 for everyone!
        'n_orientation_classes' : [4 for i in int_to_label.keys()]

    }
).explode('orientation_class') # explode on the orientation class column, so we duplicate each row for every item in the 'orientation_class' list

# only z rotations
orientation_classes['rot_x'] = 0
orientation_classes['rot_y'] = 0
orientation_classes['rot_z'] = orientation_classes['orientation_class']*90

orientation_classes['orientation_class'] += orientation_classes['label']*4

# set index
orientation_classes.set_index('label')

orientation_classes.to_csv(destination_path)

