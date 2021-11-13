##
import torch
import torch.nn as nn

##

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i_x = nn.Linear(input_size, hidden_size)
        self.i_h = nn.Linear(hidden_size, hidden_size)
        self.f_x = nn.Linear(input_size, hidden_size)
        self.f_h = nn.Linear(hidden_size, hidden_size)
        self.g_x = nn.Linear(input_size, hidden_size)
        self.g_h = nn.Linear(hidden_size, hidden_size)
        self.o_x = nn.Linear(input_size, hidden_size)
        self.o_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x:torch.Tensor, *args):
        batch_size, seq_length, hidden_size = x.shape
        hidden_states = []
        if len(args) == 0 or args[0] is None:
            h_0 = torch.zeros(self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.hidden_size, device=x.device)
        else:
            h_0, c_0 = args

        h_t_minus_1, c_t_minus_1 = h_0, c_0
        for t in range(seq_length):
            x_t = x[:, t, :]
            # print(x_t.shape)
            i_t = torch.sigmoid(self.i_x(x_t) + self.i_h(h_t_minus_1))
            f_t = torch.sigmoid(self.f_x(x_t) + self.f_h(h_t_minus_1))
            g_t = torch.tanh(self.g_x(x_t) + self.g_h(h_t_minus_1))
            o_t = torch.sigmoid(self.o_x(x_t) + self.o_h(h_t_minus_1))
            c_t = f_t * c_t_minus_1 + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            h_t_minus_1 = h_t
            c_t_minus_1 = c_t
            hidden_states.append(h_t)

        return torch.stack(hidden_states, dim=1), (hidden_states[-1], c_t_minus_1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([LSTMLayer(input_size, hidden_size)])
        self.layers.extend(nn.ModuleList([LSTMLayer(hidden_size, hidden_size) for i in range(num_layers-1)]))


    def initialize_weights(self, approach='default'):
        for layer in self.layers:
            for param in self.parameters():
                if approach == 'kaiming_uniform':
                    if len(list(param.shape)) > 1:
                        torch.nn.init.kaiming_uniform_(param)
                elif approach == 'kaiming_normal':
                    if len(list(param.shape)) > 1:
                        torch.nn.init.kaiming_normal_(param)
                elif approach == 'xavier_uniform':
                    if len(list(param.shape)) > 1:
                        torch.nn.init.xavier_uniform_(param)
                elif approach == 'xavier_normal':
                    if len(list(param.shape)) > 1:
                        torch.nn.init.xavier_normal_(param)
                elif approach == 'constant':
                    torch.nn.init.constant_(param, 0.01)
                else:
                    break

    def forward(self, x:torch.Tensor, *args):
        h_0 = None; c_0 = None
        if len(args) != 0:
            h_0, c_0 = args
        hidden_states = x
        for layer in self.layers:
            hidden_states, (last_hidden_state, cell_state) = layer(hidden_states, h_0, c_0)

        return hidden_states, (last_hidden_state, cell_state)

class TextLSTM(nn.Module):
    def __init__(self, n_class, emb_size, hidden_size, num_layers=1):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.LSTM = LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers)
        self.LSTM.initialize_weights('kaiming_uniform')
        self.W = nn.Linear(hidden_size, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)

        # hidden_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # cell_state = torch.zeros(1, len(X), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]

        # outputs, (_, _) = self.LSTM(X, (hidden_state, cell_state))
        outputs, (_, _) = self.LSTM(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    rnn1 = LSTM(input_size=4, hidden_size=6, num_layers=2)
    constant = 0.01
    rnn1.initialize_weights('constant')
    input = torch.randn(3, 2, 4)
    output1, (_, _) = rnn1(input)
    print(output1)

    rnn2 = torch.nn.LSTM(input_size=4, hidden_size=6, batch_first=True, num_layers=2)
    for param in rnn2.parameters():
        torch.nn.init.constant_(param, constant)
    output2, (_, _) = rnn2(input)
    print(output2)
    print("The output of LSTM implemented and output of LSTM in torch are " +
          ("the same." if torch.equal(output1, output2) else "not the same."))



