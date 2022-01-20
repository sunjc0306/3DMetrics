import pdb  # pdb.set_trace()
import argparse
import os
import numpy as np
import time
import utils.ObjIO as ObjIO
from MyCamera import ProjectPointsOrthogonal
from MyRenderer import ColoredRenderer
import cv2 as cv
import torch
import sys

# from opendr.lighting import VertNormals

from utils.cam_util import compute_normal
from chamfer_distance import ChamferDistance

constBackground = 4294967295
dim_h=512
dim_w=512


def render_front_normals( mesh, rn):
    # init.
    rn.set(f=mesh['f'], bgcolor=np.zeros(3))
    mesh['vn'] = compute_normal(faces=mesh['f'], vertices=mesh['v'])

    # front
    ptsToRender = mesh["v"]
    colorToRender = mesh["vn"]
    rn.set(v=ptsToRender, vc=(colorToRender + 1.) / 2.)
    normal_front = np.float32(np.copy(rn.r))
    visMap = rn.visibility_image
    fg_front = np.asarray(visMap != constBackground, np.float32).reshape(visMap.shape)

    # [0,1] -> [-1,1]
    normal_front = 2. * normal_front - 1.

    # unit normalization
    normal_front /= np.linalg.norm(normal_front, ord=2, axis=2, keepdims=True)

    # reset bg color
    normal_front *= fg_front[:, :, None]

    return normal_front,fg_front


def compute_normal_errors(nml_refi, nml_gt, msk):
    # init.
    msk_sum = np.sum(msk)

    # ----- cos. dis in (0, 2) -----
    cos_diff_map_refi = msk * (1 - np.sum(nml_refi * nml_gt, axis=-1, keepdims=True))
    cos_error2 = (np.sum(cos_diff_map_refi) / msk_sum).astype(np.float32)

    # ----- l2 dis in (0, 4) -----
    l2_diff_map_refi = msk * np.linalg.norm(nml_refi - nml_gt, axis=-1, keepdims=True)
    l2_error2 = (np.sum(l2_diff_map_refi) / msk_sum).astype(np.float32)

    return cos_error2, l2_error2


def cos_n_l2_normal_dis(nml_1, nml_2, gtMask):
    cos_error2, l2_error2 = compute_normal_errors(nml_1, nml_2,gtMask[:, :, None].astype(np.float32))

    return cos_error2,l2_error2


def compute_point_based_metrics( estMeshPath, gtMeshPath, chamfer_dist, scale=10000):

    estMesh = ObjIO.load_obj_data(estMeshPath)
    gtMesh = ObjIO.load_obj_data(gtMeshPath)
    estMesh_v = torch.from_numpy((estMesh["v"][None, :, :]).astype(np.float32)).cuda().contiguous()
    gtMesh_v = torch.from_numpy((gtMesh["v"][None, :, :]).astype(np.float32)).cuda().contiguous()
    dist_left2right, dist_right2left = chamfer_dist(gtMesh_v, estMesh_v)
    gtV_2_estM_dis = torch.mean(dist_left2right).item()
    estV_2_gtM_dis = torch.mean(dist_right2left).item()
    chamfer_dis = (gtV_2_estM_dis + estV_2_gtM_dis) / 2.
    return chamfer_dis * scale, estV_2_gtM_dis * scale, estMesh,gtMesh



def main(f,estMeshPath,gtMeshPath):
    rn = ColoredRenderer()
    rn.camera = ProjectPointsOrthogonal(rt=np.array([0, 0, 0]), t=np.array([0, 0, 2]),
                                        f=np.array([f ,f]),
                                        c=np.array([dim_w, dim_h]), k=np.zeros(5))
    rn.frustum = {'near': 0.5, 'far': 25, 'height': dim_h , 'width': dim_w}
    chamfer_dist = ChamferDistance()
    CD , PSD , estMesh,gtMesh=compute_point_based_metrics(estMeshPath,gtMeshPath,chamfer_dist)
    normals_gt,mask_gt=render_front_normals(gtMesh,rn)
    normals_est,mask_est=render_front_normals(estMesh,rn)
    cos_error2,l2_error2 = cos_n_l2_normal_dis(normals_est, normals_gt, mask_gt)
    return CD,PSD,cos_error2,l2_error2

if __name__ == '__main__':
    main(512,'/home/sunjc0306/render/dataset_example/FRONT_mesh_normalized.obj','/home/sunjc0306/render/dataset_example/FRONT_mesh_normalized.obj')



