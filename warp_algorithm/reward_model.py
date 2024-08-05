import gc

from dataclasses import dataclass
from datasets import Dataset
from typing import Optional, Dict, Union, List, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from trl import RewardConfig, RewardTrainer


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_j["input_ids"],
            "attention_mask_chosen": batch_j["attention_mask"],
            "input_ids_rejected": batch_k["input_ids"],
            "attention_mask_rejected": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


class DistilBertModel:
  def __init__(self, max_length: Optional[int] = 512,
               path_to_save_checkpoint: Optional[str] = '/kaggle/working/reward_model_bert/checkpoint-2812') -> None:
    self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    self.model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=1)
    self.max_length = max_length
    self.path_to_save_checkpoint = path_to_save_checkpoint

  def load_from_checkpoint(self, path_to_checkpoint: str) -> None:
    self.tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint)
    self.model = AutoModelForSequenceClassification.from_pretrained(path_to_checkpoint)

  def preprocess_function(self, examples: Dataset) -> Dict[str, list]:
    """Create specific dict for reward model training"""
    new_examples = {
        'input_ids_chosen': [],
        'attention_mask_chosen': [],
        'input_ids_rejected': [],
        'attention_mask_rejected': []
    }
    for chosen, rejected in tqdm(zip(examples['positive'], examples['negative'])):
      tokenized_positive = self.tokenizer(chosen, truncation=True, max_length=self.max_length)
      tokenized_negative = self.tokenizer(rejected, truncation=True, max_length=self.max_length)

      if len(tokenized_positive['input_ids']) <= self.max_length \
          and len(tokenized_negative['input_ids']) <= self.max_length:
        new_examples['input_ids_chosen'].append(tokenized_positive['input_ids'])
        new_examples['attention_mask_chosen'].append(tokenized_positive['attention_mask'])
        new_examples['input_ids_rejected'].append(tokenized_negative['input_ids'])
        new_examples['attention_mask_rejected'].append(tokenized_negative['attention_mask'])

      gc.collect()
    return new_examples

  def reward_train(self, train_dataset: Dataset) -> None:
    training_arg = RewardConfig(
        output_dir='reward_model_bert',
        per_device_train_batch_size=16,
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=1.41e-5,
        remove_unused_columns=False,
        optim='adamw_torch',
        max_length=512
    )
    trainer = RewardTrainer(
        model=self.model,
        tokenizer=self.tokenizer,
        args=training_arg,
        train_dataset=train_dataset
    )
    trainer.train()
    trainer.save_model(self.path_to_save_checkpoint)
