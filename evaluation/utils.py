"""Shared utility functions for evaluation."""


def _is_main_process():
    """Check if current process is the main process (rank 0) in distributed training."""
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


def _get_dist_info() -> tuple[int, int]:
    """Return (rank, world_size) for distributed training, else (0, 1)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


def _is_fsdp_model(model) -> bool:
    """Return True if the model is wrapped with FullyShardedDataParallel."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except ImportError:
        return False
