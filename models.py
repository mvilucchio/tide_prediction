from turtle import forward
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from torch.autograd import Variable
import math

device = torch.device('cuda')

class PressureEncorderLinear(nn.Module):
    def __init__(self, image_size = 41, patch_size = 4, num_channels = 40, encoder_stride = 4):
        super(PressureEncorderLinear, self).__init__()
        config = ViTConfig(image_size = 41, patch_size = 4, num_channels = num_channels, encoder_stride = 4)
        self.hidden_size = int((image_size // encoder_stride)**2 + 1) * config.hidden_size
        self.ViT = ViTModel(config)
        self.linear = nn.Linear(self.hidden_size + 20, 20)
        
    def forward(self, x):
        pressure, surge = x
        hidden = self.ViT(pressure).last_hidden_state.reshape(-1, self.hidden_size)
        x = torch.concat([hidden, surge], dim = 1)
        x = self.linear(x)
        return x

class PressureEncorderFull(nn.Module):
    def __init__(self, image_size = 41, patch_size = 4, num_channels = 40, encoder_stride = 4):
        super(PressureEncorderFull, self).__init__()
        config = ViTConfig(image_size = 41, patch_size = 4, num_channels = num_channels, encoder_stride = 4)
        self.hidden_size = int((image_size // encoder_stride)**2 + 1) * config.hidden_size
        self.ViT = ViTModel(config)
        layers = [
            nn.Linear(self.hidden_size + 46, 200),
            nn.ReLU(),
            nn.Linear(200, 20)
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        pressure, surge, time, scale_and_size = x
        hidden = self.ViT(pressure).last_hidden_state.reshape(-1, self.hidden_size)
        x = torch.concat([hidden, surge, time, scale_and_size], dim = 1)
        x = self.mlp(x)
        return x

class PressureEncorderSemiFull(nn.Module):
    def __init__(self, image_size = 41, patch_size = 4, num_channels = 40, encoder_stride = 4):
        super(PressureEncorderSemiFull, self).__init__()
        config = ViTConfig(image_size = 41, patch_size = 4, num_channels = num_channels, encoder_stride = 4)
        self.hidden_size = int((image_size // encoder_stride)**2 + 1) * config.hidden_size
        self.ViT = ViTModel(config)
        layers = [
            nn.Linear(self.hidden_size + 26, 200),
            nn.ReLU(),
            nn.Linear(200, 20)
        ]
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        pressure, surge, scale_and_size = x
        hidden = self.ViT(pressure).last_hidden_state.reshape(-1, self.hidden_size)
        x = torch.concat([hidden, surge, scale_and_size], dim = 1)
        x = self.mlp(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_features, seq_len=10, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
        self.num_layers = 3

        self.lstm = nn.LSTM(
          input_size=self.n_features,
          hidden_size=self.hidden_dim,
          num_layers=self.num_layers,
          batch_first=True,
          dropout = 0.1
        )
   
    def forward(self, x): 
        # the x of the encoder is a feature vector that concatenates all features
        x = x.reshape((1, self.seq_len, self.n_features))

        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        )
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        )
              
        x, (hidden, cell) = self.lstm(x, (h_0, c_0))
        return x, hidden, cell


def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    return torch.matmul(p_attn, value), p_attn


class AttentionDecoder(nn.Module):
    def __init__(self, attention = attention, seq_len=10, input_dim=64, n_features=1, encoder_hidden_state=512, add_features=2):
        super(AttentionDecoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.attention_cell = attention
        self.additional_features = add_features
        
        self.lstm = nn.LSTM(
          input_size = 1,
          hidden_size = input_dim,
          num_layers = 3,
          batch_first = True,
          dropout = 0.1
        )

        self.output_layer = nn.Linear(2 * self.hidden_dim + self.additional_features, n_features)

    def forward(self, encoder_hidden, encoder_cell, encoder_out):
        lstm_input = torch.zeros((1))

        decoder_out, (hidden_n, cell_n) = self.lstm(lstm_input, (encoder_hidden, encoder_cell))
        a = self.attention_cell(decoder_out, encoder_cell, encoder_cell)
        (a + encoder_out)
        
        output = x.squeeze(0)
        weighted = weighted.squeeze(0)
        
        x = self.output_layer(torch.cat((output, weighted), dim = 1))
        return x, hidden_n, cell_n

class Seq2SeqSurge(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim = 64, output_length = 10):
        super(Seq2SeqSurge, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.attention = Attention(512, 512)
        self.output_length = output_length
        self.decoder = AttentionDecoder(seq_len, self.attention, embedding_dim, n_features).to(device)
        
    def forward(self, x, prev_y):
        encoder_output, hidden, cell = self.encoder(x)
         
        #Prepare place holder for decoder output
        targets_ta = []
        #prev_output become the next input to the LSTM cell
        prev_output = prev_y
        
        # itearate over LSTM - according to the required output days
        for out_days in range(self.output_length) :
            prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell, encoder_output)
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x
            
            targets_ta.append(prev_x.reshape(1))
           
        targets = torch.stack(targets_ta)

        return targets

    
class EncoderSeqVit(nn.Module):
    def __init__(self, seq_len=10, num_channels = 1, output_length = 10, image_size=41, encoder_stride=4, hidden_dim=64):
        super(EncoderSeqVit, self).__init__()

        config1 = ViTConfig(image_size = 41, patch_size = 4, num_channels = seq_len, encoder_stride = 4)
        self.hidden_size1 = int((image_size // encoder_stride)**2 + 1) * config1.hidden_size
        self.ViT = ViTModel(config1)

        config2 = ViTConfig(image_size = 41, patch_size = 4, num_channels = seq_len, encoder_stride = 4)
        self.hidden_size2 = int((image_size // encoder_stride)**2 + 1) * config2.hidden_size
        self.ViT2 = ViTModel(config2)

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = 1

        self.lstm = nn.LSTM(
          input_size=4,
          hidden_size=self.hidden_dim,
          num_layers=self.num_layers,
          batch_first=True,
          # dropout = 0.1
        )

        # self.fc = nn.Linear(self.hidden_size1 + self.hidden_size2 + self.hidden_dim + 8, 2 * output_length)

        layers = [
            nn.Linear(self.hidden_size1 + self.hidden_size2 + self.hidden_dim + 8, 1000),
            nn.ReLU(), 
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 2 * output_length)
        ]

        self.mlp = nn.Sequential(*layers)
   
    def forward(self, x, device=torch.device('cuda')):
        (
            pressure1, 
            pressure2, 
            time1, time2, 
            surge1, surge2, 
            mean_surge_1, mean_surge_2, 
            std_surge_1, std_surge_2, 
            mean_p_1, mean_p_2, 
            std_p_1, std_p_2
        ) = x

        batch_size = surge1.size(0)
        
        hidden1 = self.ViT(pressure1).last_hidden_state.reshape(-1, self.hidden_size1)
        hidden2 = self.ViT(pressure2).last_hidden_state.reshape(-1, self.hidden_size2)

        x = torch.empty(batch_size, self.seq_len, 4).to(device)

        x[:,:,-4:] = torch.concat(
            [time1.unsqueeze(2), time2.unsqueeze(2), surge1.unsqueeze(2), surge2.unsqueeze(2)], 
            dim=2
        )

        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
              
        x, (hidden, cell) = self.lstm(x, (h_0, c_0))

        last_x = torch.concat(
            [
                hidden1,
                hidden2,
                hidden.squeeze(0), 
                mean_surge_1.unsqueeze(1), 
                mean_surge_2.unsqueeze(1), 
                std_surge_1.unsqueeze(1), 
                std_surge_2.unsqueeze(1), 
                mean_p_1.unsqueeze(1), 
                mean_p_2.unsqueeze(1), 
                std_p_1.unsqueeze(1), 
                std_p_2.unsqueeze(1)
            ], 
            dim=1
        )

        out = self.mlp(last_x)

        return out

class EncoderSeqVitBig(nn.Module):
    def __init__(self, seq_len=10, num_channels = 1, output_length = 10, image_size=41, encoder_stride=4, hidden_dim=64):
        super(EncoderSeqVitBig, self).__init__()
        config1 = ViTConfig(image_size = 41, patch_size = 4, num_channels = num_channels, encoder_stride = 4)
        self.hidden_size1 = int((image_size // encoder_stride)**2 + 1) * config1.hidden_size
        self.ViT = ViTModel(config1)
        config2 = ViTConfig(image_size = 41, patch_size = 4, num_channels = num_channels, encoder_stride = 4)
        self.hidden_size2 = int((image_size // encoder_stride)**2 + 1) * config2.hidden_size
        self.ViT2 = ViTModel(config2)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.lstm = nn.LSTM(
          input_size=self.hidden_size1 + self.hidden_size2 + 4,
          hidden_size=self.hidden_dim,
          num_layers=self.num_layers,
          batch_first=True,
          # dropout = 0.1
        )
        self.fc = nn.Linear(self.hidden_dim + 8, 2 * output_length)
   
    def forward(self, x):
        (
            pressure1, 
            pressure2, 
            time1, time2, 
            surge1, surge2, 
            mean_surge_1, mean_surge_2, 
            std_surge_1, std_surge_2, 
            mean_p_1, mean_p_2, 
            std_p_1, std_p_2
        ) = x
        batch_size = surge1.size(0)
        
        x = torch.empty(batch_size, self.seq_len, self.hidden_size1 + self.hidden_size2 + 4).to(device)
        
        for i in range(self.seq_len):
            hidden1 = self.ViT(pressure1[:,i,:,:].unsqueeze(1)).last_hidden_state.reshape(-1, self.hidden_size1)
            hidden2 = self.ViT(pressure2[:,i,:,:].unsqueeze(1)).last_hidden_state.reshape(-1, self.hidden_size2)
            x[:,i,:-4] = torch.concat([hidden1, hidden2], dim=1)

        x[:,:,-4:] = torch.concat(
            [time1.unsqueeze(2), time2.unsqueeze(2), surge1.unsqueeze(2), surge2.unsqueeze(2)], 
            dim=2
        )
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
              
        x, (hidden, cell) = self.lstm(x, (h_0, c_0))
        last_x = torch.concat(
            [
                hidden.squeeze(0), 
                mean_surge_1.unsqueeze(1), 
                mean_surge_2.unsqueeze(1), 
                std_surge_1.unsqueeze(1), 
                std_surge_2.unsqueeze(1), 
                mean_p_1.unsqueeze(1), 
                mean_p_2.unsqueeze(1), 
                std_p_1.unsqueeze(1), 
                std_p_2.unsqueeze(1)
            ], 
            dim=1
        )
        out = self.fc(last_x)
        return out
