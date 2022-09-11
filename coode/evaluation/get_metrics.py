import torch
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

def get_matrics(path_predict,path_groundtruth,device):
    # path_predoct: pith for the predict mesh
    # path_groundtruth: path for the groundtruth mesh
    # the predict mesh and groundtruth mesh are aligned by meshlab

    #-- 1. load meshes
    verts,faces  = load_ply(path_predict)
    verts = verts.to(device)
    faces = faces.to(device)
    mesh_predict = Meshes(verts=[verts], faces=[faces])
    verts,faces  = load_ply(path_groundtruth)
    verts = verts.to(device)
    faces = faces.to(device)
    mesh_groundtruth = Meshes(verts=[verts], faces=[faces])

    #-- 2. sample points from the suface of each mesh
    sample_predict,sample_predict_normals = sample_points_from_meshes(mesh_predict,10000,return_normals=True)
    sample_groundtruth,sample_groundtruth_normals = sample_points_from_meshes(mesh_groundtruth,10000,return_normals=True)



    #-- 4. calc metrics
    loss_chamfer,loss_chamfer_normal = chamfer_distance(sample_predict,sample_groundtruth,x_normals=sample_predict_normals,y_normals=sample_groundtruth_normals,point_reduction='sum')
    print('loss_chamfer:',loss_chamfer)
    print('loss_chamfer_normal:',loss_chamfer_normal)

if __name__ == '__main__':
    device = torch.device('cuda:1')
    path_predict = '/media/disk2/smq_data/samples/evaluation-2/squirrel/squirrel-mask-only.ply'
    path_ground_truth = '/media/disk2/smq_data/samples/evaluation-2/squirrel/squirrel-ground-truth.ply'
    get_matrics(path_predict,path_ground_truth,device)
