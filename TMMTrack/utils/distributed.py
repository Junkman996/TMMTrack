import torch

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        print('Not using distributed mode')