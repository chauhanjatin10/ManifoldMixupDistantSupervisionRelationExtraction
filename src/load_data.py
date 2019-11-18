import json
import os
import torch
from torch.utils.data import Dataset

class NYT(Dataset):

    def __init__(self, root, tokenizer, split='example',max_length=128):
        self.root = os.path.expanduser(root)
        self.split = split
        self.split_json = os.path.join(root, self.split + '.json')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        print('Data file : ',self.split_json)

        with open(self.split_json, 'rb') as f:
            self.nyt_dataset = json.load(f)

        with open(os.path.join(root, 'relation2id.json'), 'rb') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.nyt_dataset)

    def __getitem__(self, index):
        data = self.nyt_dataset[index]
        sentence = self.tokenizer.encode_plus(data['sentence'],max_length=self.max_length)["input_ids"]
        attention_mask = [1] * len(sentence)
        padding_length = self.max_length - len(sentence)
        sentence = sentence + ([self.pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        sentence = torch.tensor(sentence)
        attention_mask = torch.tensor(attention_mask)
        try:
            relation = self.labels[data['relation']]
        except KeyError:
            return self.__getitem__((index + 1) % self.__len__())
        return sentence, attention_mask, relation