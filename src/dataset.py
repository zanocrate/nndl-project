import pandas as pd
from torch.utils.data import Dataset
import torch
import pyvista as pv
import numpy as np

# BROKEN WITH SLICE INDEXING IDK WHY

class ModelNetDataset(Dataset):

    def __init__(self,metadata_path,N,split='all',orientation_classes_path=None):

        self.metadata = pd.read_csv(metadata_path,index_col=0)
        assert ('path' in self.metadata.columns) and ('label' in self.metadata.columns) # metadata should contain these fields
        
        if split not in ['train','test','all']: raise ValueError('Split must be "train","test" or "all"')
        self.split = split

        self.N = N

        if orientation_classes_path is not None:
            # orientation_classes csv file is a table that has label,label_str,label_orientation_class, rot_x, rot_y,rot_z features
            self.rotation = 'classes'
            self.orientation_classes = pd.read_csv(orientation_classes_path,index_col=0)
        else:
            self.rotation = 'random'

    
    def __len__(self):

        if self.split == 'all':
            return len(self.metadata)
        else:
            return len(self.metadata[self.metadata['split']==self.split])

    def __getitem__(self, idx):

        if self.split == 'all':
            mesh_path,label = self.metadata.loc[idx][['path','label']]
        else:
            mesh_path,label = self.metadata[self.metadata['split']==self.split].iloc[idx][['path','label']]
        

        if self.rotation == 'random':
            rot_xyz = 360*np.random.rand(3)
            voxel_grid = load_voxel_grid(mesh_path,self.N,*rot_xyz)
            return voxel_grid, rot_xyz, label
        elif self.rotation == 'classes':
            sample = self.orientation_classes.loc[label].sample(1) # grab a random orientation class for the label
            rot_xyz = sample[['rot_x','rot_y','rot_z']].values[0]     # get the corresponding rotation
            orientation_class = int(sample['orientation_class'])   # get the orientation class label
            voxel_grid = load_voxel_grid(mesh_path,self.N,*rot_xyz)
            return voxel_grid, orientation_class, label
            



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

