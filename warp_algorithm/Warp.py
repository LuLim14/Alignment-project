import os
import gc
import torch
import wandb

from tqdm import tqdm
from math import acos, sin
from copy import deepcopy
from transformers import AutoTokenizer, GPT2LMHeadModel
from typing import Type, Any, Optional
from torch import Tensor
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from reward_model import DistilBertModel
from constants import CFG
from config import Configs


class Warp:
    def __init__(self, reward_model_class: Type[DistilBertModel], prompt_dataset: DataLoader, optimizer: Any, I: int, M: int,
                 T: int, mu: float, lambd: float, eta: float, batch_size: int,
                 checkpoint_theta_dir: Optional[str] = Configs.checkpoint_theta_dir,
                 checkpoint_final_dir: Optional[str] = Configs.checkpoint_final_dir,
                 checkpoint_ema_dir: Optional[str] = Configs.checkpoint_ema_dir) -> None:
        self.sft_tokenizer = AutoTokenizer.from_pretrained('lvwerra/gpt2-imdb')
        self.sft_model = GPT2LMHeadModel.from_pretrained('lvwerra/gpt2-imdb').to(CFG.device)

        self.reward_tokenizer = reward_model_class.tokenizer
        self.reward_model = reward_model_class.model.to(CFG.device)

        self.prompt_dataset = prompt_dataset
        self.opt = optimizer
        self.I = I
        self.M = M
        self.T = T
        self.mu = mu
        self.lambd = lambd
        self.eta = eta
        self.batch_size = batch_size

        self.beta = 0.1  # from paper

        self.checkpoint_theta_dir = checkpoint_theta_dir
        self.checkpoint_final_dir = checkpoint_final_dir
        self.checkpoint_ema_dir = checkpoint_ema_dir

        # want to add b_ma from paper
    def sft_tokenize_func(self, sample: Any) -> tuple[Any, Any]:
        tokenized = self.sft_tokenizer(sample, truncation=True, max_length=15, return_tensors='pt').to(CFG.device)
        return tokenized['input_ids'], tokenized['attention_mask']

    def reward_tokenize_func(self, sample: Any) -> tuple[Any, Any]:
        tokenized = self.reward_tokenizer(sample, truncation=True, max_length=15, return_tensors='pt').to(CFG.device)
        return tokenized['input_ids'], tokenized['attention_mask']

    def compute_reward(self, y_input_ids_theta: Any, y_input_attention_mask_theta: Any, theta_m_model: Type[GPT2LMHeadModel],
                       theta_m_ema_model: Type[GPT2LMHeadModel]) -> tuple[Tensor, Tensor, Categorical]:
        """Compute KL regularized reward and KL divergence"""

        # calculate KL regularized component
        theta_m_logits = theta_m_model(input_ids=y_input_ids_theta, attention_mask=y_input_attention_mask_theta,
                                       output_hidden_states=True).logits[:, -1, :]
        theta_m_probs = Categorical(logits=theta_m_logits)

        theta_m_ema_logits = theta_m_ema_model(input_ids=y_input_ids_theta, attention_mask=y_input_attention_mask_theta,
                                               output_hidden_states=True).logits[:, -1, :]
        theta_m_ema_probs = Categorical(logits=theta_m_ema_logits)

        kl_div = torch.distributions.kl_divergence(theta_m_probs, theta_m_ema_probs).mean()

        # generate completion
        y_theta_m = theta_m_model.generate(input_ids=y_input_ids_theta, attention_mask=y_input_attention_mask_theta, max_length=50)
        y_theta_m = self.sft_tokenizer.decode(y_theta_m[0], skip_special_tokens=True)
        y_input_ids_reward, y_input_attention_mask_reward = self.reward_tokenize_func(y_theta_m)

        # reward for theta model output
        reward_logits = self.reward_model(input_ids=y_input_ids_reward, attention_mask=y_input_attention_mask_reward).logits

        # calculate KL regularized reward
        reward = reward_logits - self.beta * kl_div
        return reward, kl_div, theta_m_probs

    def policy_gradient_update(self, theta_m_model: Type[GPT2LMHeadModel], reward: Any, theta_m_probs: Type[GPT2LMHeadModel]) -> Tensor:
        """Compute loss and update weights of model"""
        policy_prod_loss = -torch.mean(theta_m_probs.log_prob(theta_m_probs.sample())) * torch.sum(reward)

        opt = torch.optim.Adam(theta_m_model.parameters(), lr=1e-6)  # from paper
        opt.zero_grad()
        policy_prod_loss.backward()
        opt.step()
        return policy_prod_loss

    def ema_update_weights(self, first_model: Type[GPT2LMHeadModel], second_model: Type[GPT2LMHeadModel], c: int) -> Type:
        """EMA for two models with coef c"""
        for first_param, second_param in zip(first_model.parameters(), second_model.parameters()):
            first_param.data = (1 - c) * first_param.data + c * second_param.data
        return first_model

    def get_angle_between_models(self, model_1: Type[GPT2LMHeadModel], model_2: Type[GPT2LMHeadModel]) -> float:
        """Compute angle between task vectors for slerp"""
        weights_1 = []
        weights_2 = []

        for param in model_1.parameters():
            weights_1.append(param.data.view(-1))
        weights_1 = torch.cat(weights_1)

        for param in model_2.parameters():
            weights_2.append(param.data.view(-1))
        weights_2 = torch.cat(weights_2)

        dot_prod = torch.dot(weights_1, weights_2)
        angle_rad = acos(dot_prod)
        return angle_rad

    def slerp(self, theta_init_model: Type[GPT2LMHeadModel], thetas: list, lambd_param: float) -> Type[GPT2LMHeadModel]:
        """Run slerp for two models"""
        result_model, delta_1, delta_2 = deepcopy(theta_init_model), deepcopy(thetas[0]), deepcopy(thetas[1])
        for delta_1_param, delta_2_param, theta_init_param in zip(delta_1.parameters(), delta_2.parameters(),
                                                                  theta_init_model.parameters()):
            delta_1_param.data -= theta_init_param.data
            delta_1_param.requires_grad = True
            delta_2_param.data -= theta_init_param.data
            delta_2_param.requires_grad = True

        omega = self.get_angle_between_models(delta_1, delta_2)

        for result_model_param, theta_init_model_param, delta_1_param, delta_2_param in zip(result_model.parameters(),
                                                                                            theta_init_model.parameters(),
                                                                                            delta_1.parameters(),
                                                                                            delta_2.parameters()):
            result_model_param.data = theta_init_model_param.data + sin((1 - lambd_param) * omega) / sin(omega) * delta_1_param.data + \
                                 sin(lambd_param * omega) / sin(omega) * delta_2_param.data
            result_model_param.requires_grad = True

        return result_model

    def slerpm(self, theta_init_model: Type[GPT2LMHeadModel], thetas: list) -> Type[GPT2LMHeadModel]:
        """Run slerpm"""
        m = len(thetas)
        if m == 2:
            return self.slerp(theta_init_model, thetas, self.lambd)
        else:
            return self.slerp(self.slerpm(theta_init_model, thetas[:-2]), thetas[:-1], self.lambd)

    def run_warp(self) -> tuple[Type[GPT2LMHeadModel], list[float], list[float], list[float]]:
        """Run WARP algorithm"""
        theta_init_model = deepcopy(self.sft_model)
        for i in range(self.I):
            theta_m_models = []
            rewards, kl_divs, policy_losses = [], [], []
            for m in range(self.M):
                theta_m_model, theta_m_ema_model = deepcopy(theta_init_model), deepcopy(theta_init_model)
                for _ in tqdm(range(self.T)):
                    for batch in self.prompt_dataset:
                        y_batch = batch['text']
                        y_input_ids_theta, y_input_attention_mask_theta = self.sft_tokenize_func(y_batch)
                        reward, kl_div, theta_m_probs = self.compute_reward(y_input_ids_theta, y_input_attention_mask_theta,
                                                                            theta_m_model, theta_m_ema_model)
                        rewards.append(reward)
                        kl_divs.append(kl_div)

                        policy_loss = self.policy_gradient_update(theta_m_model, reward, theta_m_probs)
                        policy_losses.append(policy_loss.item())

                        if Configs.use_wandb:
                            wandb.log({
                                'reward': reward,
                                'kl_div': kl_div,
                                'policy_loss': policy_loss
                            })

                        theta_m_ema_model = self.ema_update_weights(theta_m_ema_model, theta_m_model, self.mu)
                        theta_m_ema_model.save_pretrained(save_directory=os.path.join(os.getcwd(), Configs.checkpoint_ema_dir[0]))
                        gc.collect()
                theta_m_models.append(theta_m_model)

            slerp_model = self.slerpm(theta_init_model, theta_m_models)
            theta_init_model = self.ema_update_weights(theta_init_model, slerp_model, self.eta)
            theta_init_model.save_pretrained(save_directory=os.path.join(os.getcwd(), Configs.checkpoint_theta_dir[0]))
            gc.collect()

        final_model = self.ema_update_weights(self.sft_model, theta_init_model, self.eta)
        final_model.save_pretrained(save_directory=os.path.join(os.getcwd(), Configs.checkpoint_final_dir[0]))
        return final_model, policy_losses, rewards, kl_divs
