import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score


class model(pl.LightningModule):
    def __init__(self,
                 transformer,
                 output_dim,
                 tag2idx,
                 idx2tag,
                 dropout=0.1,
                 lr=1e-4):
        super().__init__()

        self.transformer_model = transformer
        emb_dim = 768

        self.linear_layer = nn.Linear(emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        # other
        self.X = tag2idx['X']
        self.idx2tag = idx2tag
        self.num_labels = output_dim
        self.lr = lr

        self.res = {'pred': [], 'label': []}
        self.lang = 'ru'

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.X)
        self.flatten = lambda l: [item for sublist in l for item in sublist]

    def forward(self, text, mask):
        embedded = self.dropout(self.transformer_model(text, mask)[0])

        result = self.linear_layer(embedded)

        return result

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def cross_entropy_loss(self, predictions, labels):
        not_X_mask = labels != self.X
        active_loss = not_X_mask.view(-1)
        active_logits = predictions.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(self.criterion.ignore_index).type_as(labels))

        loss = self.criterion(active_logits, active_labels)
        return loss

    def prepare_result(self, out):
        y_true, y_pred = [], []

        batch_pred = out['y_pred'].cpu().numpy().tolist()
        batch_y = out['y_true'].cpu().numpy().tolist()
        batch_seq = out['mask'].cpu().numpy().sum(-1).tolist()
        for i, length in enumerate(batch_seq):
            batch_pred[i] = batch_pred[i][:length]
            batch_y[i] = batch_y[i][:length]
        y_true += batch_y
        y_pred += batch_pred

        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)
        y_true = [self.idx2tag[l] for l in y_true]
        y_pred = [self.idx2tag[l] for l in y_pred]

        ids = [i for i, label in enumerate(y_true) if label != "X" and label != self.idx2tag[0]]
        y_true_cleaned = [y_true[i] for i in ids]
        y_pred_cleaned = [y_pred[i] for i in ids]
        return y_true_cleaned, y_pred_cleaned

    def training_step(self, batch, batch_idx):
        text = batch['input_ids']
        labels = batch['labels']
        mask = batch['attention_mask']

        predictions = self(text, mask)

        loss = self.cross_entropy_loss(predictions, labels)

        predict = torch.argmax(predictions, dim=-1)

        output = {
            'y_true': labels,
            'y_pred': predict,
            'mask': mask,
        }

        y_gold, y_pred = self.prepare_result(output)

        acc = accuracy_score(y_gold, y_pred)
        f1 = f1_score(y_gold, y_pred, average='macro')

        values = {'train_loss': loss, 'train_accuracy': acc, 'train_f1': f1}

        self.log_dict(values)

        return loss

    def validation_step(self, batch, batch_idx):
        text = batch['input_ids']
        labels = batch['labels']
        mask = batch['attention_mask']

        predictions = self(text, mask)

        loss = self.cross_entropy_loss(predictions, labels)

        predict = torch.argmax(predictions, dim=-1)

        output = {
            'y_true': labels,
            'y_pred': predict,
            'mask': mask,
        }

        y_gold, y_pred = self.prepare_result(output)

        acc = accuracy_score(y_gold, y_pred)
        f1 = f1_score(y_gold, y_pred, average='macro')

        values = {'val_loss': loss, 'val_accuracy': acc, 'val_f1': f1}

        self.log_dict(values)

        return loss

    def test_step(self, batch, batch_idx):
        text = batch['input_ids']
        labels = batch['labels']
        mask = batch['attention_mask']

        predictions = self(text, mask)

        loss = self.cross_entropy_loss(predictions, labels)

        predict = torch.argmax(predictions, dim=-1)

        output = {
            'y_true': labels,
            'y_pred': predict,
            'mask': mask,
        }

        y_gold, y_pred = self.prepare_result(output)

        self.res['pred'].append(y_pred)
        self.res['label'].append(y_gold)

        acc = accuracy_score(y_gold, y_pred)
        f1 = f1_score(y_gold, y_pred, average='macro')

        values = {f'test_{self.lang}_loss': loss,
                  f'test_{self.lang}_accuracy': acc,
                  f'test_{self.lang}_f1': f1}

        self.log_dict(values)

        return loss


if __name__ == '__main__':
    pass
