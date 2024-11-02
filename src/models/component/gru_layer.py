import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRULayer(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, num_layers, output_size):
        super(GRULayer, self).__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        # self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, lengths):
        # print(f"gru input x.shape: {x.shape}") # train: [32, 50, 400]
        # print(f"gru lengths.shape: {lengths.shape}") # train: [32] -> 每个length对应一个用户真实浏览新闻的个数
        # print(f"gru x.size(0) = {x.size(0)}") # train: [32]
        # batch_size = x.shape[0], num_news = x.shape[1], dim = x.shape[2]
        # print(f"x.shape: {x.shape}")
        # print(f"x: {x}")
        batch_size, num_news, dim = x.size(0), x.size(1), x.size(2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        x_reversed = torch.flip(x, dims=[1])
        packed_input = pack_padded_sequence(x_reversed, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input, h0)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output_reversed = torch.flip(output, dims=[1])

        # out = self.fc(output_reversed)

        # # TODO  处理填充项在前
        # # print(f"x.shape: {x.shape}")
        # print(f"lengths: {lengths}")
        # trimmed_input = torch.zeros(batch_size, num_news, dim).cuda()
        # for i in range(batch_size):
        #     length = lengths[i].item()
        #     # trimmed_input[i, :lengths[i]] = x[i, num_news-lengths[i]:]
        #     reversed_seq = torch.flip(x[i], dims=[0])
        #     if length > 0:
        #         trimmed_input[i, :length] = torch.flip(reversed_seq[:length], dims=[0])
        #
        # print(f"trimmed_input.shape: {trimmed_input.shape}") # train: [32, 50, 400]
        # print(f"trimmed_input: {trimmed_input}")
        #
        # packed_input = pack_padded_sequence(trimmed_input, lengths, batch_first=True, enforce_sorted=False)
        #
        # packed_output, _ = self.gru(packed_input, h0)
        # print(f"packed_output.shape: {packed_output.data.size()}")
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # # output_reversed = torch.flip(output, dims=[1])
        #
        # # out = self.fc(output)
        # # output, _ = self.gru(trimmed_input, h0)
        out = self.fc(output_reversed[:, -1, :])
        # print(f"gru out.shape: {out.shape}")
        # out = self.relu(out)
        return out