import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels, lstm_hidden_dim=512, lstm_hidden_dim_2=320, num_classes=3, dropout_rate=0.5):
        super(CNN_LSTM_Model, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.batch_norm2 = nn.BatchNorm1d(lstm_hidden_dim)

        self.lstm2 = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim_2, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.batch_norm3 = nn.BatchNorm1d(lstm_hidden_dim_2)


        self.fc = nn.Linear(lstm_hidden_dim_2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size, seq_len, input_channels = x.size()

        x = x.permute(0, 2, 1)

        x = self.conv1d(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.avg_pool(x)

        x = x.permute(0, 2, 1)


        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = self.batch_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)


        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.batch_norm3(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x[:, -1, :]
        x = self.fc(x)
        x = self.softmax(x)

        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0)