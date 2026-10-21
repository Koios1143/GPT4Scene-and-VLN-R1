import torch
import torch.nn.functional as F


def point_sample(input, point_coords, **kwargs):
    """Sample ``input`` (N, C, H, W) at normalized [0, 1] ``point_coords`` (N, P, 2)."""
    add_dim = point_coords.dim() == 3
    if add_dim:
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func, num_points,
                                               oversample_ratio, importance_sample_ratio):
    """PointRend training point sampling: most uncertain of an oversampled set + random ones."""
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)], dim=1)
    return point_coords
