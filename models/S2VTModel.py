import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.autograd import Variable


class S2VTModel(nn.Module):
    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, sos_id=1, eos_id=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', rnn_dropout_p=0.2):
        # python 3
        # super().__init__()
        super(S2VTModel, self).__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        #  hidden_size * num_directions
        #  num_directions = 2 if bidirectional else 1
        rnn_output_size = dim_hidden * 2 if bidirectional else dim_hidden

        self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers, bidirectional=bidirectional,
                                  batch_first=True, dropout=rnn_dropout_p)
        self.rnn2 = self.rnn_cell(rnn_output_size + dim_word, dim_hidden, n_layers, bidirectional=bidirectional,
                                  batch_first=True, dropout=rnn_dropout_p)
        self.rnn_cell_type = rnn_cell.lower()
        self.n_layers = n_layers
        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.out = nn.Linear(rnn_output_size, self.dim_output)

    def forward(self, vid_feats, target_variable=None,
                mode='train', opt={}):

        batch_size, n_frames, _ = vid_feats.shape
        padding_words = Variable(vid_feats.data.new(batch_size, n_frames, self.dim_word)).zero_()
        state1 = None
        state2 = None
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        output1, state1 = self.rnn1(vid_feats, state1)
        input2 = torch.cat((output1, padding_words), dim=2)
        output2, state2 = self.rnn2(input2, state2)

        padding_frames = Variable(vid_feats.data.new(batch_size, 1, self.dim_vid)).zero_()
        seq_probs = []
        seq_preds = []
        if mode == 'train':
            for i in range(self.max_length - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i])
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
        else:
            beam_size = opt.get('beam_size', 1)
            if beam_size == 1:
                current_words = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())
                for i in range(self.max_length - 1):
                    self.rnn1.flatten_parameters()
                    self.rnn2.flatten_parameters()
                    output1, state1 = self.rnn1(padding_frames, state1)
                    input2 = torch.cat(
                        (output1, current_words.unsqueeze(1)), dim=2)
                    output2, state2 = self.rnn2(input2, state2)
                    logits = self.out(output2.squeeze(1))
                    logits = F.log_softmax(logits, dim=1)
                    seq_probs.append(logits.unsqueeze(1))
                    _, preds = torch.max(logits, 1)
                    current_words = self.embedding(preds)
                    seq_preds.append(preds.unsqueeze(1))
                seq_probs = torch.cat(seq_probs, 1)
                seq_preds = torch.cat(seq_preds, 1)
            else:
                # batch*dim_word
                start = [Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda()]
                current_words = [[start, 0.0, state2]]
                for i in range(self.max_length - 1):
                    self.rnn1.flatten_parameters()
                    self.rnn2.flatten_parameters()
                    # output1: batch*1*dim_hidden
                    output1, state1 = self.rnn1(padding_frames, state1)
                    temp = []
                    for s in current_words:
                        # s: [[batch*word_embed1, batch*word_embed2...], prob, state2]
                        input2 = torch.cat(
                            (output1, self.embedding(s[0][-1]).unsqueeze(1)), dim=2)
                        output2, s[2] = self.rnn2(input2, s[2])
                        logits = self.out(output2.squeeze(1))
                        # batch*voc_size
                        logits = F.log_softmax(logits, dim=1)
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
                            temp.append([next_cap, s[1]+prob,
                                         (s[2][0].clone(), s[2][1].clone()) if isinstance(s[2], tuple)
                                         else s[2].clone()])
                    current_words = temp
                    # sort by prob
                    current_words = sorted(current_words, reverse=False, cmp=lambda x,y:cmp(int(x[1]),int(y[1])))
                    # get the top words
                    current_words = current_words[-beam_size:]
                seq_preds = torch.cat(current_words[-1][0][1:], 0).unsqueeze(0)
        return seq_probs, seq_preds

