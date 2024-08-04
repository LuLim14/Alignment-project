import os
import random
import numpy as np
import torch
import wandb


class CFG:
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    FILE_STORAGE_PATH = os.path.join(os.getcwd(), 'file_storage')


def seed_env(seed: int = CFG.seed) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seed_torch(seed: int = CFG.seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything() -> None:
    """Set seeds"""
    seed_torch()
    seed_env()


def init_wandb() -> None:
    wandb.init(project='Alignment_experiments', entity='lulim')

