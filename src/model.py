"""
CS289A INTRODUCTION TO MACHINE LEARNING
MACHINE LEARNING FOR LIP READING
Authors: Alejandro Molina & Miguel Fernandez Montes

Model
"""


import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    """
    The Encoder module is heavily based on the WATCH module from Chung et al. (2016).
    The input to the Encoder is a sequence of frames.
    The output of the Encoder is:
        a sequence of output vectors, one per time step
        a final hidden state vector
        a final cell state vector

    The Encoder comprises a convolutional network that generates a feature vector for each input
    time-step.
    These feature vectors are used as inputs to an LSTM
    """

    def __init__(self, hidden_size=256, num_layers=3):
        super(Encoder, self).__init__()
        # assume input size is 120
        # NOTE: move the convolutional block to another class?
        self.conv1 = torch.nn.Conv2d(5, 96, 3, stride=1)  # output: (batch x seq) x 118 x 118 x 96
        self.bn1 = torch.nn.BatchNorm2d(96)
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2)  # output: (batch x seq) x 58 x 58 x 96
        self.conv2 = torch.nn.Conv2d(96, 256, 3, stride=2)  # output: (batch x seq) x 28 x 28 x 256
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2)  # output: (batch x seq) x 13 x 13 x 256
        self.conv3 = torch.nn.Conv2d(256, 512, 3, stride=1, padding=1)  # output: (batch x seq) x 13 x 13 x 512
        self.conv4 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)  # output: (batch x seq) x 13 x 13 x 512
        self.conv5 = torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)  # output: (batch x seq) x 13 x 13 x 512
        self.maxpool5 = torch.nn.MaxPool2d(3, stride=2)  # output: (batch x seq) x 6 x 6 x 512
        self.fc6 = torch.nn.Linear(6 * 6 * 512, 512)  # output: (batch x seq) x 512

        # NOTE: increase size of last pooling layer: too many params in FC
        # NOTE: we could use one pooling layer and repeat it several times in forward
        # NOTE: we could use a global average pooling instead of the final FC layer

        self.lstm = torch.nn.LSTM(input_size=512,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)

    def forward(self, x):
        # OPTION 1: reshape from batch x seq x channels x h x w -> (batch x seq) x channels x h x w
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(-1, 5, 120, 120)

        # conv
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool5(x)

        # flatten and to FC
        x = x.view(-1, 6 * 6 * 512)
        x = self.fc6(x)

        # reshape
        x = x.view(batch_size, seq_len, 512)

        # rnn
        out, (hn, cn) = self.lstm(x)
        return out, hn, cn


class MLP(torch.nn.Module):
    def __init__(self, input_size=512, output_size=500):
        super(MLP, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.fc(x)
        return out


class Decoder:
    """
    The Decoder module is heavily based on the SPELL module from Chung et al. (2016).
    The input to the Decoder is:
        a sequence of output vectors from the Encoder, one per time step
        the final hidden state vector from the Encoder
        the final cell state vector from the Encoder
    The output of the Decoder is a sequence of vectors of size vocab_dim,
    each representing the probabilities of each element of our vocabulary
    at the given output time-step.

    The Decoder comprises an LSTM and an Attention module.
    """
    # TODO implement Decoder
    def __init__(self, input_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)

    def forward(self, x, out):
        pass


class Attention:
    pass


class LipReadingWords(torch.nn.Module):
    def __init__(self, enc_hidden_size=512, enc_num_layers=3, output_size=500):
        super(LipReadingWords, self).__init__()
        self.encoder = Encoder(hidden_size=enc_hidden_size, num_layers=enc_num_layers)
        self.mlp = MLP(input_size=enc_hidden_size, output_size=output_size)

    def forward(self, x):
        out_encoder, hn, cn = self.encoder(x)
        out = self.mlp(hn[-1, :, :])
        return out


class LipReading(torch.nn.Module):
    def __init__(self, enc_hidden_size=256, enc_num_layers=3, dec_hidden_size=512, dec_hidden_layers=3):
        super(LipReading, self).__init__()
        self.encoder = Encoder(hidden_size=enc_hidden_size, num_layers=enc_num_layers)
        self.decoder = Decoder(hidden_size=dec_hidden_size, num_layers=dec_hidden_layers)

    def forward(self):
        pass
