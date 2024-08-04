import torch

from torch.utils.data import DataLoader
from datasets import load_from_disk
from imdb_dataset_handler import ImdbDatasetHandler
from reward_model import DistilBertModel
from Warp import Warp
from constants import CFG


if __name__ == "__main__":
    imdb_ds = ImdbDatasetHandler()
    imdb_ds.create_train_pairs()

    path_300 = '/home/artem/PycharmProjects/AlignmentProject/Alignment_project/file_storage/dataset/train_300'
    imdb_ds.all_pairs_after_tokenize_train = load_from_disk(path_300)
    print(imdb_ds.all_pairs_after_tokenize_train)

    distil_bert_model = DistilBertModel()
    path_to_checkpoint = '/home/artem/PycharmProjects/AlignmentProject/Alignment_project/reward_model_bert/reward_model_checkpoints_200/reward_model_checkpoints_200'
    distil_bert_model.load_from_checkpoint(path_to_checkpoint)


    prompt_dataloader = DataLoader(imdb_ds.create_train_for_warp(), batch_size=CFG.batch_size, shuffle=True)
    optimizer = torch.optim.Adam
    warp = Warp(
        distil_bert_model,
        prompt_dataloader,
        optimizer,
        I=2,
        M=2,
        T=100,
        mu=0.01,
        lambd=0.5,
        eta=0.5,
        batch_size=CFG.batch_size
    )
    warp.run_warp()
