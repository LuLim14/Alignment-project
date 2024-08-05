# Alignment тестовое задание

## Clone

git clone https://github.com/LuLim14/Alignment_project.git
cd ./warp_algorithm

## Prerequisites

Установка зависимостей из 'requirements.txt':

pip install -U -r requirements.txt

## Запуск пайплайна
```
python main.py --use_wandb '[False|True]' --path_to_checkpoints_reward_model [path to reward_model checkpoints directory] --checkpoint_theta_dir [path to train_checkpoints_theta directory] --checkpoint_final_dir [path to checkpoints final directory] --checkpoint_ema_dir [path to checkpoints ema directory]
```
