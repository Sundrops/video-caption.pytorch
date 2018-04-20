import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .Attention import Attention


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1):
        # python 3
        # super().__init__()
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional

        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, dim_word)
        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden + dim_word,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                opt={}):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        rnn_cell_type = opt.get('rnn_type', 'gru')
        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []

        if mode == 'train':
            # use targets as rnn inputs
            for i in range(self.max_length - 1):
                current_words = self.embedding(targets[:, i])
                if rnn_cell_type == 'lstm':
                    context = self.attention(decoder_hidden[0].squeeze(0), encoder_outputs)
                elif rnn_cell_type == 'gru':
                    context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                self.rnn.flatten_parameters()
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            seq_logprobs = torch.cat(seq_logprobs, 1)

        elif mode == 'inference':
            if beam_size == 1:
                for t in range(self.max_length - 1):
                    if rnn_cell_type == 'lstm':
                        context = self.attention(decoder_hidden[0].squeeze(0), encoder_outputs)
                    elif rnn_cell_type == 'gru':
                        context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)

                    if t == 0:  # input <bos>
                        it = Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda()
                    elif sample_max:
                        sampleLogprobs, it = torch.max(logprobs, 1)
                        seq_logprobs.append(sampleLogprobs.view(-1, 1))
                        it = it.view(-1).long()

                    else:
                        # sample according to distribuition
                        if temperature == 1.0:
                            prob_prev = torch.exp(logprobs)
                        else:
                            # scale logprobs by temperature
                            prob_prev = torch.exp(torch.div(logprobs, temperature))
                        it = torch.multinomial(prob_prev, 1).cuda()
                        sampleLogprobs = logprobs.gather(1, Variable(it))
                        seq_logprobs.append(sampleLogprobs.view(-1, 1))
                        it = it.view(-1).long()

                    seq_preds.append(it.view(-1, 1))

                    xt = self.embedding(it)
                    decoder_input = torch.cat([xt, context], dim=1)
                    decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                    self.rnn.flatten_parameters()
                    decoder_output, decoder_hidden = self.rnn(
                        decoder_input, decoder_hidden)
                    logprobs = F.log_softmax(
                        self.out(decoder_output.squeeze(1)), dim=1)

                seq_logprobs = torch.cat(seq_logprobs, 1)
                seq_preds = torch.cat(seq_preds[1:], 1)
            #if beam_size > 1 and sample_max:
            else:
                # input <bos>
                start = [Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda()]
                current_words = [[start, 0.0, decoder_hidden]]
                for t in range(self.max_length - 1):
                    temp = []
                    for s in current_words:
                        if rnn_cell_type == 'lstm':
                            context = self.attention(s[2][0].squeeze(0), encoder_outputs)
                        elif rnn_cell_type == 'gru':
                            context = self.attention(s[2].squeeze(0), encoder_outputs)
                        xt = self.embedding(s[0][-1])
                        decoder_input = torch.cat([xt, context], dim=1)
                        decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                        self.rnn.flatten_parameters()
                        decoder_output, s[2] = self.rnn(
                            decoder_input, s[2])
                        logits = F.log_softmax(
                            self.out(decoder_output.squeeze(1)), dim=1)
                        # batch*beam
                        topk_prob, topk_word = torch.topk(logits, k=beam_size, dim=1)
                        # batch*beam -> beam*batch
                        topk_prob = topk_prob.permute(1, 0)
                        topk_word = topk_word.permute(1, 0)
                        # Getting the top <beam_size>(n) predictions and creating a
                        # new list so as to put them via the model again
                        for prob, word in zip(topk_prob, topk_word):
                            next_cap = s[0][:]
                            next_cap.append(word)
                            temp.append([next_cap, s[1] + prob,
                                         (s[2][0].clone(), s[2][1].clone()) if isinstance(s[2], tuple)
                                         else s[2].clone()])
                    current_words = temp
                    # sort by prob
                    current_words = sorted(current_words, reverse=False, cmp=lambda x, y: cmp(int(x[1]), int(y[1])))
                    # get the top words
                    current_words = current_words[-beam_size:]
                seq_preds = torch.cat(current_words[-1][0][1:], 0).unsqueeze(0)

        return seq_logprobs, seq_preds

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal(self.out.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
