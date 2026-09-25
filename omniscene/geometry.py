"""Camera conventions and padding, independent of CUDA extensions."""
import torch
import torch.nn.functional as F


def pad_images(images, patch_size=14):
    h, w = images.shape[-2:]
    if h % patch_size:
        raise ValueError('Reviewed resolutions require width-only padding')
    padded_w = ((w + patch_size - 1) // patch_size) * patch_size
    flat = images.flatten(0, 1)
    return F.pad(flat, (0, padded_w - w, 0, 0), mode='replicate').unflatten(0, images.shape[:2])


def camera_rays(intrinsics, extrinsics, height, width):
    y, x = torch.meshgrid(torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype) + .5,
                          torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype) + .5,
                          indexing='ij')
    pixels = torch.stack((x, y, torch.ones_like(x)), -1)
    directions = torch.einsum('bvij,hwj->bvhwi', torch.linalg.inv(intrinsics), pixels)
    directions = torch.einsum('bvij,bvhwj->bvhwi', extrinsics[..., :3, :3], directions)
    origins = extrinsics[..., :3, 3][:, :, None, None].expand_as(directions)
    return origins, directions


def matching_voxel_features(means, features, centers, lower_bound, voxel_size, grid_size):
    """Exact occupied-cell lookup, equivalent to the original KNN's distance==0 gate.

    The original head uses nearest neighbours only when the quantized centers are
    equal. Integer keys avoid a new CUDA extension and handle an empty scaffold.
    """
    sampled = features.new_zeros((means.shape[0], features.shape[-1]))
    if not len(centers):
        return sampled
    cells = torch.floor((means - lower_bound) / (voxel_size * 2)).long()
    center_cells = torch.floor((centers - lower_bound) / (voxel_size * 2)).long()
    shape = (grid_size.long() + 1) // 2
    valid = ((cells >= 0) & (cells < shape)).all(-1)
    stride = shape.new_tensor([int(shape[1] * shape[2]), int(shape[2]), 1])
    keys, order = (center_cells * stride).sum(-1).sort()
    query = (cells * stride).sum(-1)
    pos = torch.searchsorted(keys, query).clamp_max(len(keys) - 1)
    found = valid & (keys[pos] == query)
    sampled[found] = features[order[pos[found]]]
    return sampled
