import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class SelfAttention(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(SelfAttention, self).__init__()

		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : Number of Relation types
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		"""

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length


		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)

		self.bilstm = nn.LSTM(embedding_length, hidden_size, bidirectional=True)
		# We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
		self.W_s1 = nn.Linear(2*hidden_size, 350)
		self.W_s2 = nn.Linear(350, 30)
		self.fc_layer = nn.Linear(30*2*hidden_size, 2000)
		self.label = nn.Linear(2000, output_size)

	def attention_net(self, lstm_output):
		attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix

	def forward(self, input_sentences, batch_size=None):

		"""
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

		Returns
		-------
		Output of the linear layer containing logits for number of relation types.

		"""

		input_ = self.word_embeddings(input_sentences)
		input_ = input_.permute(1, 0, 2)
		if batch_size is None:
			h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

		output, (h_n, c_n) = self.bilstm(input_, (h_0, c_0))
		output = output.permute(1, 0, 2)

		attn_weight_matrix = self.attention_net(output)
		hidden_matrix = torch.bmm(attn_weight_matrix, output)
		# Let's now concatenate the hidden_matrix and connect it to the fully connected layer.

		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
		logits = self.label(fc_out)

		return logits
