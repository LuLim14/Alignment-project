from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets


class ImdbDatasetHandler:
  def __init__(self) -> None:
    self.train_split = load_dataset("stanfordnlp/imdb", split="train",)
    self.train_split = self.train_split.filter(lambda x: len(x["text"]) > 500, batched=False)

    self.test_split = load_dataset("stanfordnlp/imdb", split="test",)
    self.test_split = self.test_split.filter(lambda x: len(x["text"]) > 500, batched=False)
    self.all_pairs_train = []
    self.all_pairs_ds_train = None
    self.all_pairs_after_tokenize_train = None

    self.all_pairs_test = []
    self.all_pairs_ds_test = None
    self.all_pairs_after_tokenize_test = None

  def create_train_pairs(self) -> None:
    """Create train dataset of positive and negative examples"""
    positive_samples = self.train_split.filter(lambda x: x['label'] == 1)[:300]
    negative_samples = self.train_split.filter(lambda x: x['label'] == 0)[:300]
    for positive_item in tqdm(positive_samples['text']):
      for negative_item in negative_samples['text']:
        self.all_pairs_train.append({'positive': positive_item, 'negative': negative_item})
      self.all_pairs_ds_train = Dataset.from_list(self.all_pairs_train)

  def create_test_pairs(self) -> None:
    """Create test dataset of positive and negative examples"""
    positive_samples = self.test_split.filter(lambda x: x['label'] == 1)[-50:]
    negative_samples = self.test_split.filter(lambda x: x['label'] == 0)[-50:]
    for positive_item in tqdm(positive_samples['text']):
      for negative_item in negative_samples['text']:
        self.all_pairs_test.append({'positive': positive_item, 'negative': negative_item})
      self.all_pairs_ds_test = Dataset.from_list(self.all_pairs_test)

  def create_train_for_warp(self) -> Dataset:
    """Create train dataset that can use in WARP"""
    samples_zeros = Dataset.from_dict(self.train_split.filter(lambda x: x['label'] == 0)[500:550])
    samples_ones = Dataset.from_dict(self.train_split.filter(lambda x: x['label'] == 1)[600:650])
    return concatenate_datasets([samples_zeros, samples_ones]).shuffle(seed=42).remove_columns(['label'])

  def create_test_for_warp(self) -> Dataset:
    """Create test dataset that can use in WARP result validation"""
    samples_zeros = Dataset.from_dict(self.test_split.filter(lambda x: x['label'] == 0)[300:350])
    samples_ones = Dataset.from_dict(self.test_split.filter(lambda x: x['label'] == 1)[400:450])
    return concatenate_datasets([samples_zeros, samples_ones]).shuffle(seed=42)

