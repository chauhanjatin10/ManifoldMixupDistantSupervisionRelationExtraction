import numpy as np
import torch

from load_data import OpenNreNYTReader, Batchify
from data_iterator import BagIterator

def check():
	datareader = OpenNreNYTReader()
	all_instances = []
	# for inst in datareader._read("../data/main_data_splits/example.json"):
		# print(inst['sentence'].tokens)
		# print(inst['label'].label)
		# break
		# all_instances.append(inst)

	# Example usage for iterating via batches
	dataiterator = Batchify(2, datareader._read("../data/main_data_splits/example.json"))
	for batch in dataiterator:
		print(batch)

	# sorting_keys = [("sentence1", "num_tokens"), ("sentence2", "num_tokens")]
	# dataiterator = BagIterator(sorting_keys, batch_size=3)
	# print(len(all_instances))
	# for batch in dataiterator._create_batches(all_instances, shuffle=True):
	# 	for sent in batch:
	# 		print(sent)
	# 	break

if __name__ == '__main__':
	check()