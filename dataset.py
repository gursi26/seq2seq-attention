from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import pickle
import string

class EngSpaDataset(Dataset):

    def __init__(self, csv_path, glove_path, start_idx = 0, end_idx = None):
        self.df = self.read_df(csv_path, start_idx, end_idx)
        self.eng_max_len, self.spa_max_len = self.get_max_sentence_length(self.df)
        self.eng2embed, self.spa2idx, self.idx2spa = self.get_idx_mappings(self.df, glove_path)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        eng, spa = self.df.iloc[index]
        eng, spa, src_mask, tgt_mask = self.add_tokens(eng, spa)
        eng, spa = self.embed(eng, spa)
        return (eng, src_mask), (spa, tgt_mask)

    def preprocess(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def read_df(self, csv_path, start_idx, end_idx):
        if end_idx is not None:
            df = pd.read_csv(csv_path).iloc[start_idx:end_idx]
        else:
            df = pd.read_csv(csv_path).iloc[start_idx:]
        return df

    def get_idx_mappings(self, df, glove_path):
        spa2idx = {word: i + 3 for i, word in enumerate(sorted(list(set(" ".join(list(np.array(df[["SPA"]]).reshape(-1))).split()))))}
        spa2idx["<SOS>"] = 0
        spa2idx["<EOS>"] = 1
        spa2idx["<PAD>"] = 2
        idx2spa = {i: word for word, i in spa2idx.items()}
        eng2embed = {word: torch.tensor(embed) for word, embed in pickle.load(open(glove_path, "rb")).items()}
        return eng2embed, spa2idx, idx2spa

    def get_max_sentence_length(self, df):
        eng_max_len, spa_max_len = 0, 0
        for i in df.index:
            eng, spa = df.iloc[i]
            eng, spa = eng.split(), spa.split()
            if len(eng) > eng_max_len:
                eng_max_len = len(eng)
            if len(spa) > spa_max_len:
                spa_max_len = len(spa)
        return eng_max_len, spa_max_len + 2

    def add_tokens(self, eng, spa):
        eng, spa = eng.split(), ["<SOS>"] + spa.split() + ["<EOS>"]
        src_mask, tgt_mask = self.generate_mask(len(eng), len(spa))
        eng = eng + (["<PAD>"] * (self.eng_max_len - len(eng)))
        spa = spa + (["<PAD>"] * (self.spa_max_len - len(spa)))
        return eng, spa, src_mask, tgt_mask

    def generate_mask(self, eng_len, spa_len):
        eng_mask = torch.cat([torch.ones(eng_len), torch.zeros(self.eng_max_len - eng_len)]) == 1
        spa_mask = torch.cat([torch.ones(spa_len), torch.zeros(self.spa_max_len - spa_len)]) == 1
        return eng_mask, spa_mask

    def embed(self, eng, spa):
        eng = torch.cat([self.eng2embed[word][:300].unsqueeze(0) if word in self.eng2embed else torch.rand(1, 300) for word in eng], dim=0)
        spa = torch.tensor([self.spa2idx[word] for word in spa])
        return eng, spa