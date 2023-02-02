import pandas as pd
from torch.utils.data import Dataset
import torch
import pyvista as pv
import numpy as np
import torch.nn.functional as F

# BROKEN WITH SLICE INDEXING IDK WHY

# TO DO:

#     - simplify structure based on new metadata file
#     - implement one hot encoding as output


class ModelNetDataset(Dataset):

    def __init__(self,metadata_path,N,split='all', file_format='npy'):

        # load csv file
        self.metadata = pd.read_csv(metadata_path,index_col=0)

        # metadata should have the following columns
        expected_fields = ['label', 'orientation_class', 'label_id', 'path', 'label_int','orientation_class_id']
        for expected_field in expected_fields: assert(expected_field in self.metadata.columns)

        # subset entries
        self.metadata=self.metadata.set_index(['file_format','split'])

        # get the number of labels
        self.n_orientation_classes = self.metadata['orientation_class_id'].unique().size
        self.n_classes = self.metadata['label_int'].unique().size

        if file_format not in self.metadata.index.levels[0]: raise ValueError('File format not in metadata.')
        self.file_format = file_format

        if split not in ['train','test','all']: raise ValueError('Split must be "train","test" or "all"')
        if split == 'all' : self.split = ['train','test']
        else: self.split = split

        self.N = N

    
    def __len__(self):

        return len(self.metadata.loc[self.file_format].loc[self.split])

    def __getitem__(self, idx):

        # sample has orientation_class, path, orientation_class_id, label, label_int
        sample = self.metadata.loc[self.file_format].loc[self.split].iloc[idx]

        if sample['orientation_class'] == -1:
            rot_xyz = 360*np.random.rand(3)
            voxel_grid = load_voxel_grid(sample['path'],self.N,*rot_xyz)
            return voxel_grid, rot_xyz, sample['label_int']
        else:
            voxel_grid = np.load(sample['path'])
            F.one_hot(sample['orientation_class_id'],num_classes=N_ORIENTATION_CLASSES).float()
             # ONE HOT THESE
            sample['label_int']
            return voxel_grid, sample['orientation_class_id'], sample['label_int']
            



def load_voxel_grid(mesh_path : str,N : int,rot_x,rot_y,rot_z,pivot=None,add_channel_dim : bool =True):
    """
    Loads a mesh .ply file using pyvista, rotate it along the x,y,z axes 
    using rot_x,rot_y_rot_z arguments as degrees, and then voxelizes it in a NxNxN grid.

    Arguments:
    ------
        mesh_path : str, path to the mesh .ply file
        N : int, number of cells per dimension of the cubic uniform grid
        rot_x , rot_y , rot_z : floats, rotation degrees of the mesh along the x,y,z axis
        pivot : default 0,0,0; pivot for the rotation
        add_channel_dim
    
    Returns:
    ------
        voxel_grid : np.array of shape (N,N,N), of dtype int8, representing the occupancy voxel grid
    """

    mesh = pv.read(mesh_path)

    if pivot is None: pivot = np.zeros(3)

    mesh.rotate_x(rot_x,pivot,inplace=True)
    mesh.rotate_y(rot_y,pivot,inplace=True)
    mesh.rotate_z(rot_z,pivot,inplace=True)

    pivot=mesh.outline().center_of_mass()
    mesh.translate(-pivot,inplace=True)

    scaling_factor = 1/max(abs(np.array(mesh.bounds)))
    mesh.scale(scaling_factor,inplace=True)

    resolution = N*np.ones(3,dtype=int)
    grid = pv.UniformGrid()
    grid.dimensions = resolution+1 # the grid points are the edges of the cells: add 1 for last edge
    grid.spacing = 2/resolution # spacing of the grid
    grid.origin = (-1,-1,-1) # point from which the grid grows

    # now we grab the cell centers
    points = grid.cell_centers()

    # and compute the distance form the mesh
    points.compute_implicit_distance(mesh,inplace=True)

    l = grid.spacing[0]
    threshold = 3**(0.5)*l
    mask = abs(points.point_data['implicit_distance']) < threshold

    grid.cell_data['occupancy']=np.zeros(grid.n_cells,dtype=np.int8)
    grid.cell_data['occupancy'][mask] = 1

    array=grid.cell_data['occupancy'].reshape(resolution)

    if add_channel_dim: array = np.expand_dims(array,axis=0)

    return array

