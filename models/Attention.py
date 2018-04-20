import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        # python 3
        # super().__init__()
        super(Attention, self).__init__()
        #self.dim = dim
        #self.linear1 = nn.Linear(dim * 2, dim)
        #self.linear2 = nn.Linear(dim, 1, bias=False)
        #self._init_hidden()
        #self.dk = dim/2
        # self.contextW = nn.Linear(dim, self.dk)
        # nn.init.xavier_normal(self.contextW.weight)
        # self.hidderW = nn.Linear(dim, self.dk)
        # nn.init.xavier_normal(self.hidderW.weight)
    def _init_hidden(self):
        nn.init.xavier_normal(self.linear1.weight)
        nn.init.xavier_normal(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        ############### original ###################
        '''
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        (batch, seq_len, dim*2)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        (batch, seq_len, dim*2)->(batch, seq_len, dim)->(batch, seq_len, 1)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context
        '''
        ################# seq2seq #######################
        '''
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        # batch, seq_len, dim
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        # (batch, seq_len, dim) * (batch, dim, seq_len) -> (batch, seq_len, seq_len)
        attn = torch.bmm(hidden_state, encoder_outputs.transpose(1, 2))
        attn = F.softmax(attn.view(-1, seq_len)).view(batch_size, -1, seq_len)
        # (batch, seq_len, seq_len) * (batch, seq_len, dim) -> (batch, seq_len, dim)
        mix = torch.bmm(attn, encoder_outputs)
        # concat -> (batch, seq_len, 2*dim)
        combined = torch.cat((mix, hidden_state), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output
        '''
        ######## after reducing dim, calculate the similarity of between encoder_outputs and hidden_state #########
        '''
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        # (batch, seq_len, self.dk)
        encoder_outputs_dk = self.contextW(encoder_outputs)
        # (batch, self.dk)
        hidden_state_dk = self.hidderW(hidden_state)
        # (batch, seq_len, self.dk) * (batch, self.dk, 1) -> (batch, seq_len, 1)-> (batch, seq_len)
        attn = torch.bmm(encoder_outputs_dk, hidden_state_dk.unsqueeze(2)).squeeze(2)
        # (batch, seq_len)-> (batch, 1, seq_len)
        attn = F.softmax(attn, dim=1).unsqueeze(1)
        # (batch, 1, seq_len) * (batch, seq_len, dim) -> (batch, 1, dim)
        context = torch.bmm(attn, encoder_outputs).squeeze(1)
        return context
        '''
        ######### directly calculate the similarity of between encoder_outputs and hidden_state ############
        # batch_size, seq_len, hidden_size = encoder_outputs.size()
        # (batch, seq_len, dim) * (batch, dim, 1) -> (batch, seq_len, 1)-> (batch, seq_len)
        attn = torch.bmm(encoder_outputs, hidden_state.unsqueeze(2)).squeeze(2)
        # (batch, seq_len)-> (batch, 1, seq_len)
        attn = F.softmax(attn, dim=1).unsqueeze(1)
        # (batch, 1, seq_len) * (batch, seq_len, dim) -> (batch, 1, dim)
        context = torch.bmm(attn, encoder_outputs).squeeze(1)
        return context