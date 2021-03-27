import collections
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from conllu import parse_incr


class Reader:
    def __init__(self, filename: str, tags: str, tokenizer) -> None:
        data = open(filename, "r", encoding="utf-8")
        self.token_list = [i for i in parse_incr(data)]

        self.tokenizer = tokenizer
        self.tags = tags
        self.upos_tags = ['ADJ', 'ADP', 'ADV', 'CCONJ', 'NOUN', 'PRON', 'VERB']
        self.train_tag = {'Tense': ['Pres', 'Fut', 'Past'],
                          'Gender': ['Neut', 'Fem', 'Masc']}
        
        self.sentences = self.prepare_data()
        self.tag2idx = self.get_tag2idx()
        self.idx2tag = self.get_idx2tag()

    def prepare_data(self) -> List[dict]:
        sentences = []
        sentence = []
        s = {}

        for tokens in self.token_list:
            for i in tokens:
                s['form'] = i['form']
                if i['feats']:
                    if self.tags == 'upos':
                        if i['upos'] in self.upos_tags:
                            s['labels'] = i['upos']
                        else:
                            s['labels'] = 'X'
                    else:
                        if (self.tags in list(i['feats'].keys())) and (
                                i['feats'][self.tags] in self.train_tag[self.tags]):
                            s['labels'] = i['feats'][self.tags]
                        else:
                            s['labels'] = 'X'
                else:
                    if self.tags == 'upos':
                        if i['upos'] in self.upos_tags:
                            s['labels'] = i['upos']
                        else:
                            s['labels'] = 'X'
                    else:
                        s['labels'] = 'X'

                sentence.append(s)
                s = {}
            sentences.append(self.prepare_result(sentence))
            sentence = []

        return sentences

    @staticmethod
    def prepare_result(sentence: List[dict]) -> dict:
        words = [word_pos['form'] for word_pos in sentence]
        tags_dict = [word_pos['labels'] for word_pos in sentence]
        if len(words) > 100:
            words = words[0:100]
            tags_dict = tags_dict[0:100]
        return {'words': words, 'labels': tags_dict}

    def get_tag2idx(self) -> dict:
        tags = list(set(i for sent in self.sentences for i in sent['labels'] if i != self.tokenizer.pad_token))
        tags = [self.tokenizer.pad_token] + tags
        tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        return tag2idx

    def print_example(self, idx: int) -> None:
        s = self.sentences[idx]['words']
        t = self.sentences[idx]['labels']
        for i, j in zip(s, t):
            print(f'{i}\t\t\t{j}')

    def get_idx2tag(self) -> dict:
        idx2tag = {v: k for k, v in self.tag2idx.items()}
        return idx2tag

    def count_label(self) -> dict:
        c = collections.Counter()
        for s in self.sentences:
            for label in s['labels']:
                c[label] += 1
        c = dict(c)
        return dict(c)

    def plot_tag_distribution(self) -> None:
        count = self.count_label()
        plt.figure(figsize=(15, 4))
        sns.set(style="darkgrid")
        x = list(count.keys())
        y = list(count.values())
        sns.barplot(x=x, y=y).set_title(self.tags)

    def plot_len_distribution(self) -> None:
        w_len = [len(i['words']) for i in self.sentences]
        plt.figure(figsize=(15, 4))
        sns.set(style="darkgrid")
        sns.distplot(w_len)


if __name__ == '__main__':
    pass
