import numpy as np
import math
import copy
def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R
#
#
def generate_cameras(dist=2, view_num=360):
    cams = []
    target = [0, 0, 0]
    up = [0, 1, 0]
    for view_idx in range(view_num):
        angle = (math.pi * 2 / view_num) * view_idx
        eye = np.asarray([dist * math.sin(angle), 0, dist * math.cos(angle)])

        fwd = np.asarray(target, np.float64) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)

        cams.append(
            {
                'center': eye,
                'direction': fwd,
                'right': right,
                'up': -down,
            }
        )

    return cams
def voxelization_normalization(verts, useMean=True, useScaling=True,threshWD=0.3333,threshH=0.5):
    """
    normalize the mesh into H [-0.5,0.5]*(1-margin), W/D [-0.333,0.333]*(1-margin)
    """

    vertsVoxelNorm = copy.deepcopy(verts)
    vertsMean, scaleMin = None, None

    if useMean:
        vertsMean = np.mean(vertsVoxelNorm, axis=0, keepdims=True)  # (1, 3)
        vertsVoxelNorm -= vertsMean

    xyzMin = np.min(vertsVoxelNorm, axis=0)
    assert (np.all(xyzMin < 0))
    xyzMax = np.max(vertsVoxelNorm, axis=0)
    assert (np.all(xyzMax > 0))

    if useScaling:
        scaleArr = np.array(
            [threshWD / abs(xyzMin[0]), threshH / abs(xyzMin[1]), threshWD / abs(xyzMin[2]),
             threshWD / xyzMax[0], threshH / xyzMax[1], threshWD / xyzMax[2]])
        scaleMin = np.min(scaleArr)
        vertsVoxelNorm *= scaleMin

    return vertsVoxelNorm, vertsMean, scaleMin
def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm