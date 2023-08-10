import torch
from torch import nn


class BiGRUEncoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BiGRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        return self.gru(x)[0]


class AlignmentModel(nn.Module):

    def __init__(self, input_size):
        super(AlignmentModel, self).__init__()
        self.layer = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.layer(x)


class GRUDecoder(nn.Module):

    def __init__(self, hidden_size, out_size, max_len, device, p=0.3):
        super(GRUDecoder, self).__init__()
        self.hidden_size, self.out_size, self.max_len = hidden_size, out_size, max_len
        self.device = device
        self.alignment = AlignmentModel(hidden_size * 3)
        self.embed = nn.Embedding(out_size, hidden_size)
        self.linear = nn.Linear(hidden_size * 3, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
            )
        self.out_proj = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(p=p)
        self.relu = nn.ReLU()

    def forward(self, encoder_outputs, tgt_seq=None, hidden=None):
        batch_size, _, _ = encoder_outputs.shape
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

        initial_token = torch.zeros(batch_size, self.out_size).to(self.device)
        initial_token[:, 0] = 1
        outputs = [initial_token.unsqueeze(1)]

        if tgt_seq is None: # No teacher forcing
            for i in range(self.max_len):
                out, hidden = self.forward_t(outputs[-1].squeeze(1).argmax(dim=1), hidden, encoder_outputs)
                outputs.append(out.unsqueeze(1))
        else: # with teacher forcing
            for i in range(tgt_seq.shape[1]):
                out, hidden = self.forward_t(tgt_seq[:, i], hidden, encoder_outputs)
                outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)


    # model_input: Previously generated word (or target shifted right if teacher forcing)
    # hidden: Hidden state from previous timestep
    # encoder_outputs: Output activations from bidirectional encoder
    def forward_t(self, model_input, hidden, encoder_outputs):
        batch_size, _, _ = encoder_outputs.shape
        alignment_input = torch.cat([encoder_outputs, hidden.permute(1, 0, 2).repeat(1, encoder_outputs.shape[1], 1)], dim=-1)
        attention_weights = self.alignment(alignment_input).view(batch_size, -1).softmax(dim=1)
        context = (encoder_outputs * attention_weights.unsqueeze(-1)).sum(dim=1)
        model_input = self.embed(model_input)
        model_input = torch.cat([model_input, context], dim=-1).unsqueeze(1)
        model_input = self.dropout(self.relu(self.linear(model_input)))
        out, hidden = self.gru(model_input, hidden)
        return self.out_proj(self.dropout(out)).squeeze(1), hidden