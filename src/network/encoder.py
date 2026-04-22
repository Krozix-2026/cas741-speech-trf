import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    r"""Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=True, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns: output, hidden
            - **output**: N x T x H
            - **hidden**: (num_layers * num_directions) x N x H 
        """
        # Add total_length for supportting nn.DataParallel() later
        # see https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        total_length = padded_input.size(1)  # get the max sequence length
        print("total_length:", total_length)#120
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
        print("packed_input:", packed_input)
        
        packed_output, hidden = self.rnn(packed_input)
        print("packed_output:", packed_output)
        print("packed_output:", packed_output[0].shape)#[120, 256]
        print("hidden:", len(hidden))#2
        print("hidden:", hidden[0].shape)#[4, 1, 128]
        print("hidden:", hidden[1].shape)#[4, 1, 128]
        
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        print("output:", output.shape)#[1, 120, 256]
        return output, hidden

    def flatten_parameters(self):
        self.rnn.flatten_parameters()