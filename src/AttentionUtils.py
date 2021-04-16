import torch
import seaborn as sns
import matplotlib.pyplot as plt


class VisAttention:
    def __init__(self, sample, model, drop_special_tokens=True,
                 mode_visualization='layer2head', save_result=False):
        self.tok = sample['words_bpe']
        ids = sample['input_ids']
        mask = sample['attention_mask']
        with torch.no_grad():
            output = model(ids, mask)

        self.attention = output['attentions']
        self.save_result = save_result
        self.mode_visualization = mode_visualization

        if drop_special_tokens:
            self.attention = [attn[:, :, 1:-1, 1:-1] for attn in self.attention]
            self.tok = self.tok[1:-1]

        self.attention = torch.stack([attn.squeeze(0) for attn in self.attention])
        self.cmap = plt.cm.OrRd

    def layer2head(self):
        # attns = torch.stack([attn.squeeze(0) for attn in self.attention])

        att_map = [torch.max(torch.flatten(att, start_dim=-2), dim=-1).values for att in self.attention]
        att_map = torch.stack([att.T for att in att_map], dim=0)

        plt.figure(figsize=(10, 7))
        sns.heatmap(att_map.detach().to('cpu').numpy(), cmap=self.cmap)

    def layer2word(self, position):
        attention_avg = self._other2word(position, 1)
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(attention_avg.detach().to('cpu').numpy(), xticklabels=self.tok, cmap=self.cmap)
        ax.set(xlabel='words', ylabel='Layer')
        plt.show()

    def head2word(self, position):
        attention_avg = self._other2word(position, 0)
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(attention_avg.detach().to('cpu').numpy(), xticklabels=self.tok, cmap=self.cmap)
        ax.set(xlabel='words', ylabel='Head')
        plt.show()

    def _other2word(self, position, dim):
        attention = self.attention.permute(2, 1, 0, 3)
        attention_position = attention[position]
        attention_avg = attention_position.mean(dim=dim)
        return attention_avg

    def word2word(self, ):
        # attns = torch.stack([attn.squeeze(0) for attn in self.attention])
        attns = self.attention.permute(2, 3, 0, 1)

        att_map = [torch.max(torch.flatten(att, start_dim=-2), dim=-1).values for att in attns]
        att_map = torch.stack([att.T for att in att_map], dim=0)

        plt.figure(figsize=(10, 7))
        sns.heatmap(att_map.detach().to('cpu').numpy(),
                    xticklabels=self.tok, yticklabels=self.tok,
                    cmap=self.cmap)

    def word2word_position(self, position_layer=-1, position_head=-1):
        attn = self.attention[position_layer, position_head]  # attns[layer, head]

        plt.figure(figsize=(10, 7))
        sns.heatmap(attn.detach().to('cpu').numpy(),
                    xticklabels=self.tok, yticklabels=self.tok,
                    cmap=self.cmap)


class AnalysisAttention:
    def __init__(self, iter, model, drop_special_tokens=True, save_result=False):
        # self.lang = lang
        self.iter = iter
        self.model = model
        self.drop_special_tokens = drop_special_tokens
        self.cmap = plt.cm.BuPu

    def step(self, sample):
        ids = sample['input_ids']
        mask = sample['attention_mask']
        with torch.no_grad():
            output = self.model(ids, mask)

        attention = output['attentions']
        return attention

    def layer2head(self):
        atts = []
        for sample in self.iter:
            attention = self.step(sample)
            if self.drop_special_tokens:
                attention = [attn[:, :, 1:-1, 1:-1] for attn in attention]

            # attention = torch.stack([attn.squeeze(0) for attn in attention])
            att_map = [torch.max(torch.flatten(att, start_dim=-2), dim=-1).values for att in attention]
            att_map = torch.stack([att.T for att in att_map], dim=0)
            atts.append(att_map)

        mean_max_head_weight = torch.mean(torch.cat(atts, dim=-1), dim=-1).detach().to('cpu').numpy()
        sns.heatmap(mean_max_head_weight, cmap=self.cmap)
