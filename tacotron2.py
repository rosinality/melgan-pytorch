import torch
from torch import nn
from torch.nn import functional as F


from time import perf_counter


def tick():
    torch.cuda.synchronize()
    return perf_counter()


class Conv1dBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, activation='relu'):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channel)

        if activation is not None:
            activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
            self.activation = activation_map[activation]

        else:
            self.activation = None

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        embed_dim,
        kernel_size,
        n_conv,
        conv_dim,
        lstm_dim,
        n_lstm,
        dropout,
    ):
        super().__init__()

        self.embed = nn.Embedding(n_vocab, embed_dim)

        convs = []
        channel = embed_dim
        for i in range(n_conv):
            convs.append(Conv1dBN(channel, channel, kernel_size))
            convs.append(nn.Dropout(dropout))
            channel = conv_dim

        self.conv = nn.Sequential(*convs)

        self.h0 = nn.Parameter(torch.randn(n_lstm * 2, 1, lstm_dim // 2) * 0.05)
        self.c0 = nn.Parameter(torch.randn(n_lstm * 2, 1, lstm_dim // 2) * 0.05)

        self.lstm = nn.LSTM(conv_dim, lstm_dim // 2, n_lstm, bidirectional=True)

    def forward(self, text):
        embed = self.embed(text).transpose(1, 2)
        conv = self.conv(embed).permute(2, 0, 1)

        batch = text.shape[0]
        h = self.h0.expand(-1, batch, -1).contiguous()
        c = self.c0.expand(-1, batch, -1).contiguous()

        self.lstm.flatten_parameters()
        out, _ = self.lstm(conv, (h, c))
        out = out.transpose(0, 1)

        return out


class PreNet(nn.Module):
    def __init__(self, in_dim, dim, dropout, deterministic=False):
        super().__init__()

        self.linear1 = nn.Sequential(nn.Conv1d(in_dim, dim, 1), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.ReLU())

        self.dropout = dropout
        self.deterministic = deterministic

    def forward(self, input):
        out = self.linear1(input)

        if self.training or not self.deterministic:
            out = F.dropout(out, p=self.dropout, training=True)

        out = self.linear2(out)

        if self.training or not self.deterministic:
            out = F.dropout(out, p=self.dropout, training=True)

        return out


class MultiLSTMCell(nn.Module):
    def __init__(self, in_dim, dim, n_layer, zoneout):
        super().__init__()

        self.cell = nn.ModuleList()
        self.h0 = nn.ParameterList()
        self.c0 = nn.ParameterList()

        cell_in_dim = in_dim

        for i in range(n_layer):
            self.cell.append(nn.LSTMCell(cell_in_dim, dim))
            self.h0.append(nn.Parameter(torch.randn(1, dim) * 0.05))
            self.c0.append(nn.Parameter(torch.randn(1, dim) * 0.05))

            cell_in_dim = dim

        self.zoneout = zoneout

    def init_state(self, batch_size):
        h0 = [h.expand(batch_size, -1) for h in self.h0]
        c0 = [c.expand(batch_size, -1) for c in self.c0]

        return h0, c0

    def forward(self, input, hidden=None):
        out = input

        if hidden is not None:
            prev_h, prev_c = hidden

        else:
            batch = input.shape[0]

            prev_h = [h.expand(batch, -1) for h in self.h0]
            prev_c = [c.expand(batch, -1) for c in self.c0]

        h_list = []
        c_list = []

        for cell, h, c in zip(self.cell, prev_h, prev_c):
            next_h, next_c = cell(out, (h.contiguous(), c.contiguous()))

            if self.training and self.zoneout is not None:
                mask = torch.empty_like(next_h).bernoulli_(self.zoneout)
                next_h = mask * h + (1 - mask) * next_h
                mask = torch.empty_like(next_c).bernoulli_(self.zoneout)
                next_c = mask * c + (1 - mask) * next_c

            h_list.append(next_h)
            c_list.append(next_c)

            out = next_h

        next_hidden = (h_list, c_list)

        return next_h, next_hidden


class LocationAwareAttention(nn.Module):
    def __init__(self, dec_dim, attention_dim, loc_kernel=31, loc_dim=32):
        super().__init__()

        self.lin_query = nn.Linear(dec_dim, attention_dim, bias=False)

        self.lin_score = nn.Linear(attention_dim, 1, bias=False)

        self.conv_loc = nn.Sequential(
            nn.Conv1d(1, loc_dim, loc_kernel, padding=loc_kernel // 2, bias=False),
            nn.Conv1d(loc_dim, attention_dim, 1, bias=False),
        )

        self.mask = float('-inf')

    def forward(self, query, target, l_target, prev_attention, mask):
        l_query = self.lin_query(query).unsqueeze(1)

        conv_loc = self.conv_loc(prev_attention).permute(0, 2, 1)

        score = self.lin_score(torch.tanh(l_query + l_target + conv_loc))

        score.masked_fill_(mask, self.mask)
        attn = F.softmax(score, 1).permute(0, 2, 1)
        align = attn @ target
        out = align.squeeze(1)

        return out, attn


def length_to_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max().item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = ids < lengths.unsqueeze(1)

    return mask


class Decoder(nn.Module):
    def __init__(
        self,
        n_mels,
        pre_dim,
        enc_dim,
        attention_dim,
        loc_dim,
        loc_kernel,
        dim,
        n_layer,
        dropout,
        zoneout,
        deterministic,
        max_length,
        stop_threshold,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.max_length = max_length
        self.stop_threshold = stop_threshold

        self.prenet = PreNet(n_mels, pre_dim, dropout, deterministic)

        self.lin_target = nn.Linear(enc_dim, attention_dim, bias=False)
        self.attention = LocationAwareAttention(dim, attention_dim, loc_kernel, loc_dim)
        self.cell = MultiLSTMCell(pre_dim + enc_dim, dim, n_layer, zoneout)
        self.projection = nn.Linear(enc_dim + dim, n_mels + 1)

    def forward(self, mels, enc_out, text_len):
        l_enc_out = self.lin_target(enc_out)
        dec_h, dec_c = self.cell.init_state(mels.shape[0])
        batch, n_mel, length = mels.shape
        go = mels.new_zeros(batch, n_mel, 1)
        mels = torch.cat([go, mels], 2)
        mels_pad = self.prenet(mels)

        attn_score = enc_out.new_zeros(batch, 1, enc_out.shape[1])
        attn_scores = []

        attn_mask = ~length_to_mask(text_len, max_len=enc_out.shape[1]).unsqueeze(2)

        out = []

        for i in range(length):
            mel = mels_pad[:, :, i]
            attn, attn_score_next = self.attention(
                dec_h[-1], enc_out, l_enc_out, attn_score, attn_mask
            )
            attn_score = attn_score + attn_score_next
            attn_scores.append(attn_score_next)
            cell_in = torch.cat([mel, attn], 1)
            dec_out, (dec_h, dec_c) = self.cell(cell_in, (dec_h, dec_c))
            cell_out = torch.cat([dec_out, attn], 1)
            out.append(cell_out)

        out = torch.stack(out, 1)
        out = self.projection(out).transpose(1, 2)

        mel_out, stop_out = out.split([n_mel, 1], 1)
        stop_out = stop_out.squeeze(1)

        attn_scores = torch.cat(attn_scores, 1).transpose(1, 2)

        return mel_out, stop_out, attn_scores

    def decode(self, enc_out, text_len):
        batch = enc_out.shape[0]
        l_enc_out = self.lin_target(enc_out)
        dec_h, dec_c = self.cell.init_state(batch)
        mels = enc_out.new_zeros(batch, self.n_mels)

        attn_score = enc_out.new_zeros(batch, 1, enc_out.shape[1])

        attn_mask = ~length_to_mask(text_len).unsqueeze(2)

        out = []
        stop_out = []

        finished = torch.zeros(batch, dtype=torch.torch.bool, device=enc_out.device)
        attn_scores = []

        for i in range(self.max_length):
            mel = self.prenet(mels.unsqueeze(2)).squeeze(2)
            attn, attn_score_next = self.attention(
                dec_h[-1], enc_out, l_enc_out, attn_score, attn_mask
            )
            attn_score = attn_score + attn_score_next
            attn_scores.append(attn_score_next)
            cell_in = torch.cat([mel, attn], 1)
            dec_out, (dec_h, dec_c) = self.cell(cell_in, (dec_h, dec_c))
            cell_out = torch.cat([dec_out, attn], 1)
            mels = self.projection(cell_out)
            mels, stop = mels.split([self.n_mels, 1], 1)
            out.append(mels)
            stop_out.append(stop)

            check_finish = stop > self.stop_threshold
            finished = finished | check_finish

            if finished.all().item() == 1:
                break

        out = torch.stack(out, 2)

        stop_out = torch.stack(stop_out, 2)
        attn_scores = torch.cat(attn_scores, 1).transpose(1, 2)

        return out, stop_out, attn_scores


class Tacotron2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(
            config.n_vocab,
            config.embed_dim,
            config.enc_kernel_size,
            config.enc_n_conv,
            config.enc_conv_dim,
            config.enc_lstm_dim,
            config.enc_n_lstm,
            config.dropout,
        )

        self.decoder = Decoder(
            config.n_mels,
            config.pre_dim,
            config.enc_lstm_dim,
            config.attention_dim,
            config.loc_dim,
            config.loc_kernel,
            config.dec_dim,
            config.dec_n_layer,
            config.dropout,
            config.zoneout,
            config.deterministic,
            config.dec_max_length,
            config.dec_stop_threshold,
        )

        postnet = []
        post_in_dim = config.n_mels
        post_dim = config.post_dim
        for i in range(config.n_post):
            if i == config.n_post - 1:
                post_dim = config.n_mels
                activation = None

            else:
                activation = 'tanh'

            postnet.append(
                Conv1dBN(post_in_dim, post_dim, config.post_kernel_size, activation)
            )
            postnet.append(nn.Dropout(config.dropout))

            post_in_dim = post_dim

        self.postnet = nn.Sequential(*postnet)

        self.mel_loss = nn.MSELoss()
        self.stop_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, text, text_len, mels=None, mels_len=None, stop_target=None, valid=False
    ):
        enc_out = self.encoder(text)

        if self.training or valid:
            mel_out, stop_out, attn_scores = self.decoder(mels, enc_out, text_len)
            mel_post = self.postnet(mel_out)
            mel_post = mel_out + mel_post

            if mels_len is not None:
                mask = ~length_to_mask(mels_len, mels.shape[2])
                mask = mask.expand(mel_out.shape[1], -1, -1)
                mask = mask.permute(1, 0, 2)

                # print(mel_out.shape, mel_post.shape, stop_out.shape, mels_len.max(), mask.shape)

                mel_out.masked_fill_(mask, 0)
                mel_post.masked_fill_(mask, 0)
                stop_out.masked_fill_(mask[:, 0, :], 1e3)

            if stop_target is not None:
                # print(mel_out[0], stop_out[0])
                mel_loss = self.mel_loss(mel_out, mels)
                mel_post = self.mel_loss(mel_post, mels)
                stop_loss = self.stop_loss(stop_out, stop_target)

                return (
                    mel_out,
                    stop_out,
                    attn_scores,
                    {'mel': mel_loss, 'mel_post': mel_post, 'stop': stop_loss},
                )

            else:
                return mel_out, stop_out, attn_scores, None

        else:
            mel_out, stop_out, attn_scores = self.decoder.decode(enc_out, text_len)
            mel_post = self.postnet(mel_out)
            mel_post = mel_out + mel_post

            return mel_post, stop_out, attn_scores, None
