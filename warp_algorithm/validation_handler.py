import torch

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from typing import Any, Type
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import Dataset
from reward_model import DistilBertModel
from constants import CFG


class ValidationResultsHandler:
    def __init__(self, reward_model: Type[DistilBertModel], prompt_dataset: Dataset, path_to_final_weights: str):
        self.final_theta_tokenizer = AutoTokenizer.from_pretrained('lvwerra/gpt2-imdb') # equal tokenizer with sft_model
        self.final_theta_model = GPT2LMHeadModel.from_pretrained(path_to_final_weights).to(CFG.device)

        self.sft_model = GPT2LMHeadModel.from_pretrained('lvwerra/gpt2-imdb').to(CFG.device)

        self.reward_tokenizer = reward_model.tokenizer
        self.reward_model = reward_model.model.to(CFG.device)

        self.prompt_dataloader = DataLoader(prompt_dataset, batch_size=CFG.batch_size, shuffle=False)

        self.beta = 0.1 # from paper

    def final_theta_tokenize_func(self, sample: Any) -> tuple[Any, Any]:
        tokenized = self.final_theta_tokenizer(sample, truncation=True, max_length=15, return_tensors='pt').to(CFG.device)
        return tokenized['input_ids'], tokenized['attention_mask']

    def reward_tokenize_func(self, sample: Any) -> tuple[Any, Any]:
        tokenized = self.reward_tokenizer(sample, truncation=True, max_length=15, return_tensors='pt').to(CFG.device)
        return tokenized['input_ids'], tokenized['attention_mask']

    def compute_reward(self, y_input_ids_theta: Any, y_input_attention_mask_theta: Any) -> tuple[float, float, float, float]:
        theta_final_logits = self.final_theta_model(input_ids=y_input_ids_theta, attention_mask=y_input_attention_mask_theta,
                                                    output_hidden_states=True).logits[:, -1, :]  # last token
        theta_final_probs = Categorical(logits=theta_final_logits)

        theta_sft_logits = self.sft_model(input_ids=y_input_ids_theta, attention_mask=y_input_attention_mask_theta,
                                               output_hidden_states=True).logits[:, -1, :]  # last token
        theta_sft_probs = Categorical(logits=theta_sft_logits)

        kl_div = torch.distributions.kl_divergence(theta_final_probs, theta_sft_probs).mean()

        y_theta_final_model = self.final_theta_model.generate(input_ids=y_input_ids_theta,
                                                              attention_mask=y_input_attention_mask_theta,
                                                              max_length=50)
        y_theta_final_model = self.final_theta_tokenizer.decode(y_theta_final_model[0], skip_special_tokens=True)
        y_input_ids_reward, y_input_attention_mask_reward = self.reward_tokenize_func(y_theta_final_model)
        reward_logits_final_model = self.reward_model(input_ids=y_input_ids_reward,
                                                      attention_mask=y_input_attention_mask_reward).logits

        y_theta_sft_model = self.sft_model.generate(input_ids=y_input_ids_theta,
                                                    attention_mask=y_input_attention_mask_theta,
                                                    max_length=50)
        y_theta_sft_model = self.final_theta_tokenizer.decode(y_theta_sft_model[0], skip_special_tokens=True)
        y_input_ids_reward_sft, y_input_attention_mask_reward_sft = self.reward_tokenize_func(y_theta_sft_model)
        reward_logits_sft_model = self.reward_model(input_ids=y_input_ids_reward_sft,
                                                    attention_mask=y_input_attention_mask_reward_sft).logits

        reward = reward_logits_final_model - self.beta * kl_div
        return reward, kl_div, reward_logits_final_model, reward_logits_sft_model

    def run_validation(self) -> tuple[list[float], list[float], list[float], list[float]]:
        """Run validation process"""
        rewards, kl_divs, rewads_final_model, rewards_sft_model = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(self.prompt_dataloader):
                y_batch = batch['text']
                y_input_ids_theta, y_input_attention_mask_theta = self.final_theta_tokenize_func(y_batch)
                reward, kl_div, reward_logits_final_model, reward_logits_sft_model = self.compute_reward(y_input_ids_theta, y_input_attention_mask_theta)
                rewards.append(reward)
                kl_divs.append(kl_div)
                rewads_final_model.append(reward_logits_final_model)
                rewards_sft_model.append(reward_logits_sft_model)
        return rewards, kl_divs, rewads_final_model, rewards_sft_model

    def check_weights(self):
        for param_final, param_sft in zip(self.final_theta_model.parameters(), self.sft_model.parameters()):
            if not torch.equal(param_final, param_sft):
                print(f'yes')
