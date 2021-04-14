import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DatasetForAttention:
    def __init__(self,
                 sentences: list,
                 tokenizer,
                 device: str = device) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.tokenizer_pl = lambda x: tokenizer.encode_plus(x,
                                                            add_special_tokens=True,
                                                            return_tensors="pt").to(device)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        words = ' '.join(self.sentences[idx]['words'])
        tokens = self.tokenizer_pl(words)
        words_bpe = [self.tokenizer.cls_token] + self.tokenizer.tokenize(words) + [self.tokenizer.sep_token]

        result = {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'words': words,
            'words_bpe': words_bpe,
        }

        return result
