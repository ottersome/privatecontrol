import torch
from torch import nn 

class Adversary(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Adversary, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrivialTemporalAdversary(torch.nn.Module):
    def __init__(
        self,
        num_pub_features: int,
        num_prv_features: int,
        dnn_hidden_size: int,
        rnn_hidden_size: int,
    ):
        super(TrivialTemporalAdversary, self).__init__()
        self.input_size = num_pub_features
        self.dnn_hidden_size = dnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.lstm = nn.LSTM(self.input_size, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, self.dnn_hidden_size)
        self.fc2 = nn.Linear(self.dnn_hidden_size, self.dnn_hidden_size * 2)
        self.fc3 = nn.Linear(self.dnn_hidden_size * 2, num_prv_features)

    def forward(self, x):
        seq_representation, (rnn_hn, rnn_cn)  = self.lstm(x)
        x = torch.relu(self.fc1(seq_representation[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PCATemporalAdversary(torch.nn.Module):
    def __init__(
        self,
        num_principal_components: int,
        num_features_to_recon: int,
        dnn_hidden_size: int,
        rnn_hidden_size: int,
    ):
        super(PCATemporalAdversary, self).__init__()
        self.input_size = num_principal_components
        self.dnn_hidden_size = dnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_output_features = num_features_to_recon

        self.lstm = nn.LSTM(self.input_size, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, self.dnn_hidden_size)
        self.fc2 = nn.Linear(self.dnn_hidden_size, self.dnn_hidden_size * 2)
        self.fc3 = nn.Linear(self.dnn_hidden_size * 2, self.num_output_features)

    def forward(self, x):
        seq_representation, (rnn_hn, rnn_cn)  = self.lstm(x)
        x = torch.relu(self.fc1(seq_representation[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
