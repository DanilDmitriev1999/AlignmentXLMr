import torch
import numpy as np
from typing import List

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    from torch.cuda import LongTensor
else:
    from torch import LongTensor


class CustomDataset:
    def __init__(self,
                 sentences: list,
                 tokenizer,
                 tags2ids: dict,
                 max_len: int = 300,
                 align_token: str = 'last') -> None:

        self.sentences = sentences
        self.tokenizer = tokenizer
        self.tags2ids = tags2ids
        self.max_len = max_len
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.pad_token = tokenizer.pad_token

        assert align_token in ['last', 'first'], print('only last or first is possible')
        self.align_token = align_token

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = LongTensor([token_ids])
        assert ids.size(1) < self.max_len, print(ids.size(1))
        if pad:
            padded_ids = torch.zeros(self.max_len).long()
            padded_ids[:ids.size(1)] = ids
            mask = torch.zeros(self.max_len).long()
            mask[0:ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    @staticmethod
    def flatten(list_of_lists: list):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens: List[str], labels: List[str]):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.cls_token] + list(self.flatten(subwords)) + [self.sep_token]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        if self.align_token == 'last':
            bert_labels = [(sublen - 1) * ["X"] + [label] for sublen, label in zip(subword_lengths, labels)]
        else:
            bert_labels = [[label] + (sublen - 1) * ["X"] for sublen, label in zip(subword_lengths, labels)]
        bert_labels = [self.pad_token] + list(self.flatten(bert_labels)) + [self.pad_token]

        assert len(subwords) <= 512
        return subwords, token_start_idxs, bert_labels

    def subword_tokenize_to_ids(self, tokens: List[str], labels: List[str]) -> dict:
        assert len(tokens) == len(labels)
        subwords, token_start_idxs, bert_labels = self.subword_tokenize(tokens, labels)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(self.max_len)
        token_starts[token_start_idxs] = 1
        bert_labels = [self.tags2ids[label] for label in bert_labels]
        padded_bert_labels = torch.ones(self.max_len).long() * self.tags2ids["X"]
        padded_bert_labels[:len(bert_labels)] = LongTensor(bert_labels)

        mask.require_grad = False
        return {
            "input_ids": subword_ids,
            "attention_mask": mask,
            "bert_token_starts": token_starts,
            "labels": padded_bert_labels,
        }

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        result = self.subword_tokenize_to_ids(self.sentences[idx]['words'],
                                              self.sentences[idx]['labels'])

        return result
