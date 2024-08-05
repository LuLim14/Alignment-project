import argparse
import torch
import numpy as np
import wandb

from torch.utils.data import DataLoader
from datasets import load_from_disk
from imdb_dataset_handler import ImdbDatasetHandler
from reward_model import DistilBertModel
from Warp import Warp
from validation_handler import ValidationResultsHandler
from constants import CFG, seed_everything, init_wandb
from config import Configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WARP project')

    parser.add_argument('--use_wandb', type=bool, required=True, help='True if you want to use wandb')
    parser.add_argument('--path_to_checkpoints_reward_model', type=str, required=True,
                        help='Path to reward model checkpoints')
    parser.add_argument('--checkpoint_theta_dir', type=str, required=True,
                        help='Path to theta models checkpoint')
    parser.add_argument('--checkpoint_final_dir', type=str, required=True,
                        help='Path to final model checkpoint')
    parser.add_argument('--checkpoint_ema_dir', type=str, required=True,
                        help='Path to ema models checkpoint')
    parser.add_argument('--I', type=int, default=Configs.I, help='Number of iterations')
    parser.add_argument('--M', type=int, default=Configs.M, help='Number of runs')
    parser.add_argument('--T', type=int, default=Configs.T, help='Number of steps')
    parser.add_argument('--mu', type=float, default=Configs.mu, help='Constant for EMA')
    parser.add_argument('--lambd', type=float, default=Configs.lambd, help='Constant for SLERPM')
    parser.add_argument('--eta', type=float, default=Configs.eta, help='Constant for LITI')

    args = parser.parse_args()
    Configs.use_wandb = args.use_wandb
    Configs.path_to_checkpoints_reward_model = args.path_to_checkpoints_reward_model
    Configs.checkpoint_theta_dir = args.checkpoint_theta_dir
    Configs.checkpoint_final_dir = args.checkpoint_final_dir
    Configs.checkpoint_ema_dir = args.checkpoint_ema_dir
    Configs.I = args.I
    Configs.M = args.M
    Configs.T = args.T
    Configs.mu = args.mu
    Configs.lambd = args.lambd
    Configs.eta = args.eta

    seed_everything()
    if Configs.use_wandb:
        init_wandb()
    imdb_ds = ImdbDatasetHandler()
    imdb_ds.create_train_pairs()

    distil_bert_model = DistilBertModel(path_to_save_checkpoint=Configs.path_to_checkpoints_reward_model)

    imdb_ds.all_pairs_after_tokenize_train = imdb_ds.all_pairs_ds_train.map(
        distil_bert_model.preprocess_function,
        batched=True,
        num_proc=4
    )
    imdb_ds.all_pairs_after_tokenize_train = imdb_ds.all_pairs_after_tokenize_train.filter(
        lambda x: len(x['input_ids_chosen']) <= 512 and len(x['input_ids_rejected']) <= 512
    )
    imdb_ds.all_pairs_after_tokenize_train = imdb_ds.all_pairs_after_tokenize_train.remove_columns(['positive', 'negative'])

    distil_bert_model.reward_train(imdb_ds.all_pairs_after_tokenize_train)

    prompt_dataloader = DataLoader(imdb_ds.create_train_for_warp(), batch_size=CFG.batch_size, shuffle=True)
    optimizer = torch.optim.Adam
    warp = Warp(
        distil_bert_model,
        prompt_dataloader,
        optimizer,
        I=Configs.I,
        M=Configs.M,
        T=Configs.T,
        mu=Configs.mu,
        lambd=Configs.lambd,
        eta=Configs.eta,
        batch_size=CFG.batch_size,
        checkpoint_theta_dir=Configs.checkpoint_theta_dir,
        checkpoint_final_dir=Configs.checkpoint_final_dir,
        checkpoint_ema_dir=Configs.checkpoint_ema_dir
    )
    final_model, policy_losses, rewards, kl_divs = warp.run_warp()

    if Configs.use_wandb:
        wandb.run.summary['mean_loss_train'] = np.mean(policy_losses)
        wandb.run.summary['mean_rewards_train'] = np.mean(rewards)
        wandb.run.summary['mean_KL_train'] = np.mean(kl_divs)

    test_ds = imdb_ds.create_test_for_warp()
    path_to_final_model = Configs.checkpoint_final_dir[0]
    validation_handler = ValidationResultsHandler(distil_bert_model, test_ds, path_to_final_model)
    rewards, kl_divs, rewards_final_model, rewards_sft_model = validation_handler.run_validation()

    if Configs.use_wandb:
        wandb.run.summary['mean_reward_final_model'] = np.mean(rewards_final_model)
        wandb.run.summary['mean_reward_sft_model'] = np.mean(rewards_sft_model)
        wandb.run.summary['mean_reward'] = np.mean(rewards)
        wandb.run.summary['mean_kl'] = np.mean(kl_divs)
