from turtle import forward
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()


        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride = 2, padding = 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        y= self.block(x)
        return y


class SequenceClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_layers=3,
        dropout_p=.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()


        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            # bidirectional이어서 2배
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, h, w)

        z, _ = self.rnn(x)


        z = z[:, -1]

        y = self.layers(z)

        return y

        
