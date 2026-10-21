import torch.distributed as dist


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()
