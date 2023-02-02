import pandas as pd
from torch.utils.data import Dataset
import torch
import pyvista as pv
import numpy as np
import torch.nn.functional as F

# broken with slice indexing idk why

class ModelNetDataset(Dataset):

    def __init__(self,
                metadata_path : str,
                N : int = 30,
                split = None, 
                file_format : str = 'npy' # tells the object which file format he will provide
                ):

        # load csv file
        self.metadata = pd.read_parquet(metadata_path)

        # metadata should have the following columns
        expected_fields = [
                            'path',
                           'split',
                           'label',
                           'label_id',
                           'orientation_class',
                           'orientation_class_id',
                           'rot_x',
                           'rot_y',
                           'rot_z'
                            ]

        for expected_field in expected_fields: assert(expected_field in self.metadata.columns)

        ############## subset on file format
        
        self.file_format = file_format
        
        assert (file_format == 'npy') or (file_format == 'ply')
        self.metadata = self.metadata[self.metadata.path.str.contains(file_format)]

        ############## subset on split
        if split is None: split = 'test|train' # use regex or
        else: assert (split == 'test') or (split == 'train')

        self.metadata=self.metadata[self.metadata.split.str.contains(split)]

        ############## get the number of labels
        self.n_orientation_classes = self.metadata.orientation_class_id.unique().size
        self.n_classes = self.metadata.label_id.unique().size

        self.N = N

    
    def __len__(self):
        # metadata was already subset in initialization
        return len(self.metadata)

    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]

        if self.file_format == 'npy':
            # we expect a voxel_grid, one_hot_encoded_orientation_class, one_hot_encoded_class output
            voxel_grid = np.load(sample.path)
            orientation_class = torch.tensor(sample.orientation_class_id.item())
            one_hot_encoded_orientation_class = F.one_hot(orientation_class,num_classes=self.n_orientation_classes)
            label = torch.tensor(sample.label_id.item())
            one_hot_encoded_label = F.one_hot(label,num_classes=self.n_classes)

            return voxel_grid,one_hot_encoded_orientation_class,one_hot_encoded_label

        elif self.file_format == 'ply':
            # for now, just return a random voxelized rotation of the object
            # with the vector rot_xyz and one hot encoded label
            rot_xyz = 360*np.random.rand(3)
            voxel_grid = load_voxel_grid(sample.path,self.N,*rot_xyz)
            one_hot_encoded_label = F.one_hot(torch.Tensor(sample.label_id.item()),num_classes=self.n_classes)

            return voxel_grid, rot_xyz, one_hot_encoded_label
            



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

