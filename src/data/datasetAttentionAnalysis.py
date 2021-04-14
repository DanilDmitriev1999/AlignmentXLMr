import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DatasetForAttention:
    def __init__(self,
                 sentences: list,
                 tokenizer,
                 device: str = device) -> None:
        self.sentences = sentences
        self.tokenizer = lambda x: tokenizer.encode_plus(x,
                                                         add_special_tokens=True,
                                                         return_tensors="pt").to(device)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        result = self.tokenizer(self.sentences[idx]['words'])

        return result
