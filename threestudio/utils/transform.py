import numpy as np
import torch
from kornia.geometry.quaternion import Quaternion


@torch.no_grad()
def scale_gaussians(gaussian, scale):
    gaussian._xyz.data = gaussian._xyz.data * scale
    g_scale = gaussian.get_scaling * scale
    gaussian._scaling.data = torch.log(g_scale + 1e-7)

@torch.no_grad()
def scale_gaussians_obj(gaussian, scale, startidx, obj_len):
    # Ensure the indices are within bounds
    endidx = startidx + obj_len
    if endidx > gaussian._xyz.data.shape[0]:
        raise IndexError("Index range exceeds Gaussian data size.")

    # Apply the scaling to the specified range of Gaussian points
    gaussian._xyz.data[startidx:endidx] = gaussian._xyz.data[startidx:endidx] * scale

    # Calculate and update the scaling factor for the object
    g_scale = gaussian.get_scaling[startidx:endidx] * scale
    gaussian._scaling.data[startidx:endidx] = torch.log(g_scale + 1e-7)



@torch.no_grad()
def rotate_gaussians(gaussian, rotmat):
    rot_q = Quaternion.from_matrix(rotmat[None, ...])
    g_qvec = Quaternion(gaussian.get_rotation)
    gaussian._rotation.data = (rot_q * g_qvec).data

    gaussian._xyz.data = torch.einsum("ij,bj->bi", rotmat, gaussian._xyz.data)

@torch.no_grad()
def rotate_gaussians_obj(gaussian, rotmat, startidx, obj_len):
    # Ensure indices are within bounds
    endidx = startidx + obj_len
    if endidx > gaussian._rotation.data.shape[0] or endidx > gaussian._xyz.data.shape[0]:
        raise IndexError("Index range exceeds Gaussian data size.")

    # Convert rotation matrix to quaternion
    rot_q = Quaternion.from_matrix(rotmat[None, ...])
    
    # Extract current rotation and apply new rotation
    g_qvec = Quaternion(gaussian.get_rotation[startidx:endidx])
    
    # Update the rotation for the new object
    gaussian._rotation.data[startidx:endidx] = (rot_q * g_qvec).data
    
    # Apply the rotation matrix to the XYZ positions of the new object
    gaussian._xyz.data[startidx:endidx] = torch.einsum("ij,bj->bi", rotmat, gaussian._xyz.data[startidx:endidx])

@torch.no_grad()
def translate_gaussians(gaussian, tvec):
    gaussian._xyz.data = gaussian._xyz.data + tvec[None, ...]

@torch.no_grad()
def translate_gaussians_obj(gaussian, tvec, startidx, obj_len):
    # Ensure the indices are within bounds
    endidx = startidx + obj_len
    if endidx > gaussian._xyz.data.shape[0]:
        raise IndexError("Index range exceeds Gaussian data size.")

    # Apply the translation to the specified range of Gaussian points
    gaussian._xyz.data[startidx:endidx] = gaussian._xyz.data[startidx:endidx] + tvec[None, ...]

from scipy.spatial.transform import Rotation as R

default_model_mtx = (
    torch.from_numpy(R.from_rotvec(-np.pi / 2 * np.array([1.0, 0.0, 0.0])).as_matrix())
    .float()
    .cuda()
)
