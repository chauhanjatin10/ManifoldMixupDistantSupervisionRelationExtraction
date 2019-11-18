import torch
import numpy as np
import json
# from torchtext.data import get_tokenizer
import nltk

class Dataset:
	def __init__(self):
		self.vocab = {"<pad>": 0}
		self.vocab_inverse = {0: "<pad>"}

		self.relation_dict = {}
		self.relation_dict_inverse = {}

		# one can also use torchtext tokenizer, if nltk is not available
		# self.tokenizer = get_tokenizer("basic_english")
		self.tokenizer = nltk.word_tokenize

		train_sentences, train_labels = self.load_data("train")
		train_data = list(zip(train_sentences, train_labels))
		train_data.sort(key = lambda x: len(x[0]))
		self.train_sentences, self.train_labels = list(zip(*train_data))

		test_sentences, test_labels = self.load_data("test")
		test_data = list(zip(test_sentences, test_labels))
		test_data.sort(key = lambda x: len(x[0]))
		self.test_sentences, self.test_labels = list(zip(*test_data))

	def create_sent_from_index(self, batch):
		for sent in batch:
			word_sentence = [self.vocab_inverse[word] for word in sent]
			print(word_sentence)

	def load_data(self, type_):
		with open("../data-mining/data/main_data_splits/{}.json".format(type_), "r") as f:
			all_sents = json.load(f)

		sentences = []
		labels = []
		for sent in all_sents:
			actual_sentence = sent["sentence"]
			relation_type = sent["relation"]
			if relation_type not in self.relation_dict.keys():
				self.relation_dict[relation_type] = len(self.relation_dict)
				self.relation_dict_inverse[len(self.relation_dict_inverse)] = relation_type

			raw_tokens = self.tokenizer(actual_sentence)

			idx_sent = []
			for word in raw_tokens:
				if word not in self.vocab.keys():
					self.vocab[word] = len(self.vocab)
					self.vocab_inverse[len(self.vocab_inverse)] = word
				idx_sent.append(self.vocab[word])

			sentences.append(idx_sent)
			labels.append(self.relation_dict[relation_type])

		return sentences, labels

	def get_batch(self, data_req, idx, batch_size):
		if data_req == "train":
			data_ = self.train_sentences
			data_labels = self.train_labels
		elif data_req == "test":
			data_ = self.test_sentences
			data_labels = self.test_labels

		batch = data_[ (idx)*batch_size : (idx+1)*batch_size ]
		padded_batch = self.padded_batch(batch)

		cls_targets = data_labels[ (idx)*batch_size : (idx+1)*batch_size ]
		cls_targets = torch.LongTensor(cls_targets)
		return padded_batch, cls_targets

	def padded_batch(self, batch):
		max_sent_len = 0
		for sent in batch:
			if max_sent_len < len(sent):
				max_sent_len = len(sent)

		idx_word_batch = []
		for sent in batch:
			sent_diff = max_sent_len - len(sent)
			idx_word_batch.append(sent + [self.vocab["<pad>"]]*sent_diff)
			
		return torch.LongTensor(idx_word_batch)