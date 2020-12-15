#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
from models.tdnn import TDNN
import torch
import torch.nn.functional as F
from utils import PreEmphasis
import torchaudio

class X_vector(nn.Module):
    def __init__(self, nOut, n_mels = 40, log_input=True, encoder_type='SAP', **kwargs):
        super(X_vector, self).__init__()

        self.log_input  = log_input
        self.n_mels     = n_mels
        self.encoder_type = encoder_type
        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.stdim = 512
        self.statsdim = 1500

        self.torchfb        = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
                )

        if self.encoder_type == "SAP":
            out_dim = self.statsdim
        elif self.encoder_type == "ASP":
            out_dim = self.statsdim*2
        else:
            raise ValueError('Undefined encoder')

        self.tdnn1 = TDNN(input_dim=n_mels, output_dim=self.stdim, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=self.stdim, output_dim=self.stdim, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=self.stdim, output_dim=self.stdim, context_size=7, dilation=2,dropout_p=0.5)

        self.dense1 = nn.Linear(self.stdim, self.stdim)
        self.dense2 = nn.Linear(self.stdim, out_dim)
        self.dense3 = nn.Linear(out_dim, self.stdim) #x-vector
        self.dense4 = nn.Linear(self.stdim, self.stdim) #x-vector
        self.output = nn.Linear(self.stdim, nOut)

        self.nonlinearity = nn.ReLU()

        self.attention = nn.Sequential(
            nn.Conv1d(out_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, out_dim, kernel_size=1),
            nn.Softmax(dim=2),
            )

    def forward(self, x):

        with torch.no_grad():
            x = self.torchfb(x)+1e-6
            if self.log_input: x = x.log()
            x = self.instancenorm(x).permute(0,2,1)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)

        x = self.nonlinearity(self.dense1(x))
        x = self.nonlinearity(self.dense2(x))

        x = x.permute(0, 2, 1)

        ### Stat Pool
        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)

        x = self.nonlinearity(self.dense3(x)) #x-vector
        x = self.nonlinearity(self.dense4(x)) #x-vector
        predictions = self.output(x)

        return predictions

def MainModel(nOut=256, **kwargs):
    model = X_vector(nOut, **kwargs)
    return model
