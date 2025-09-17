import os
import torch
import torch.distributed as dist
import logging

log = logging.getLogger(__name__)

def init_distributed_mode(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(gpu)
        
        if rank == 0:
            log.info(f"Initializing DDP: World size {world_size}")
        
        # Initialize process group - the device is already set via torch.cuda.set_device
        dist.init_process_group(
            backend=cfg.distributed.backend,
            world_size=world_size,
            rank=rank        )
        
        if rank == 0:
            log.info(f"Process group initialized successfully")
        
        # Test communication with a simple barrier
        try:
            dist.barrier()
            if rank == 0:
                log.info(f"Initial barrier successful for all {world_size} processes")
        except Exception as e:
            # Error logging should appear on all ranks for debugging
            log.error(f"Initial barrier failed for rank {rank}: {e}")
            raise
            
        if rank == 0:
            log.info(f"DDP initialization completed: {world_size} processes on {world_size} GPUs")
    else:
        log.warning("Not in a distributed environment. Running in single-process mode.")

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
