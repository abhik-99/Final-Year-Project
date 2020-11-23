import pandas as pd
import os
# from torchtext.data import Dataset, BucketIterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

#Modified from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/image_captioning/get_loader.py
class Vocabulary:
    def __init__(self, freq_thres=1):

        # [CLS] - > Start of sentance/sequence
        # [SEP] - > END of sentance/sequence
        self.itos = {0: "[PAD]", 1: "[CLS]", 2: "[SEP]", 3: "[UNK]", 4: "[MASK]"}
        self.stoi = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4}
        self.freq_threshold = freq_thres

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_seq(fasta_seq):
#         print(fasta_seq)
        return [str(x) for x in list(fasta_seq)]

    def build_vocabulary(self, seq_list):
        frequencies = {}
        idx = len(self.itos)
        for idx1, base in enumerate(list('acgut')):
            self.stoi[base] = idx+idx1
            self.itos[idx+idx1] = base

    def numericalize(self, fasta_seq):
        tokenized_seq = self.tokenizer_seq(fasta_seq.lower())

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_seq
        ]

class SequenceDataset(Dataset):
    def __init__(self, filename, freq_threshold=5):
        self.df = pd.read_csv(filename, header=None)

        # Get Sequences (miRNA and Target mRNA)
        # Dataset Column Positions - miRNA, mRNA, miRNA_Seq, mRNA_Seq, Relative_score
        self.mirna = self.df.iloc[:, 2]
        self.mrna = self.df.iloc[:, 3]
        self.rel_score = self.df.iloc[:, -1]
        
        #concatenating row-wise to create a combined vocabulary
        all_seq = self.mirna[:] + self.mrna
        
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(all_seq.tolist())

    def __len__(self):
        return len(self.df)

    def numericalize_seq(self,seq):
        numericalized_seq = [self.vocab.stoi["<SOS>"]]
        numericalized_seq += self.vocab.numericalize(seq)
        numericalized_seq.append(self.vocab.stoi["<EOS>"])
        return numericalized_seq
    def get_vocabulary(self):
        return self.vocab.stoi

    def __getitem__(self, index):
        mirna, mrna, score = torch.tensor(self.numericalize_seq(self.mirna[index])), torch.tensor(self.numericalize_seq(self.mrna[index])),torch.tensor(self.rel_score[index])
#         mirna, mrna, score = mirna.unsqueeze(0), mrna.unsqueeze(0), score.unsqueeze(0)
#         print(mirna.size(), mrna.size())
        return mirna, mrna, score


class CollateSequences:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
#         imgs = [item[0].unsqueeze(0) for item in batch]
#         imgs = torch.cat(imgs, dim=0)
#         targets = [item[1] for item in batch]
#         targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        mirna = [item[0] for item in batch]
        mrna = [item[1] for item in batch]
        
        mirna = pad_sequence(mirna, batch_first=True, padding_value=self.pad_idx)
        mrna = pad_sequence(mrna, batch_first=True, padding_value=self.pad_idx)

        return mirna, mrna, [item[2] for item in batch]

# Returns a ready Loader and the Dataset Class for the Sequence
def get_loader(
    seq_csv,
    batch_size=5,
    num_workers=8,
    shuffle=True,
    pin_memory=True
):
    dataset = SequenceDataset(filename=seq_csv)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CollateSequences(pad_idx=pad_idx)
    )

    return loader, dataset
