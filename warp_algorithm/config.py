from typing import Optional

class Configs:
    use_wandb: Optional[bool] = False
    path_to_checkpoints_reward_model: Optional[str] = './reward_model_bert'
    I: Optional[int] = 2
    M: Optional[int] = 2
    T: Optional[int] = 100
    mu: Optional[float] = 0.01
    lambd: Optional[float] = 0.5
    eta: Optional[float] = 0.5
    checkpoint_theta_dir: Optional[str] = 'train_checkpoints_theta_init',
    checkpoint_final_dir: Optional[str] = 'train_checkpoints_final',
    checkpoint_ema_dir: Optional[str] = 'train_checkpoints_ema'