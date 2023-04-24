import torch
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .base import Model


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, *args, **kwargs):
        """
        :param n_inputs: кол-во входных каналов
        :param n_outputs: кол-во выходных каналов = кол-во входных каналов на след. слое
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) # drop padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.05)
        self.conv2.weight.data.normal_(0, 0.05)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.05)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, dropout=0.2):
        """
        :param in_channels: кол-во входных каналов (for sample = 1)
        :param channels: ls of channels [n_outs', n_out'', n_outs''', n_outputs]
                            len(channels) N TemporalBlocks
        * кол-во выходных каналов = channels[-1]
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_layers = len(channels)  # num_layers <=> num of TemporalBlock
        for i in range(num_layers):
            dilation_size = 2 ** i
            if i > 0: in_channels = channels[i-1]
            out_channels = channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        input: (BS, in_channels, T)
        output: (BS, channels[-1], T)
        """
        return self.network(x)
    

class TCNClassification(Model):
    def __init__(self, timesteps, in_channels, channels, n_classes=3, 
                 kernel_size=3, hidden_dim=100, dropout=0.2):
        """
        This classificator works for single frame (F, T)
        Frames in batch (B, N, F, Tfr) will be computed undependently, 
        i.e. (B, N, F, Tfr) -> (BN, F, Tfr).
        If arg 'clip_frames' is defined, sequence of frames N
        will be pooled into sequence of clips N* using linear softmax.
        N* = N // clip_frames
        Args:
            timesteps: time dim Tfr in input
            in_channels: feature dim F in input
            n_classes: number of classes C in output
        """
        super(TCNClassification, self).__init__()
        self.tcn = TemporalConvNet(in_channels, channels, kernel_size, dropout)
        self.out_dim = timesteps*channels[-1]
        self.fc1 = nn.Linear(self.out_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, n_classes)  
        self.act = nn.Softmax(dim=1)
        self.n_classes = n_classes
        
    def forward(self, x):
        """
        B - batch size
        F - feature dim (for ex. mels)
        N - frame sequence length
        N* - clip sequence length
        T - time dim
        C - n classes
        Args:
            x: (B, F, T)
        Return:
            tuple:
                logits, probs: (B, C)
        """
        x = self.tcn(x)                # (_, C[-1], T)
        x = x.view(-1, self.out_dim)   # (_, C[-1]*T)
        x = self.fc1(x)                # (_, C)
        x = self.act1(x)
        
        logits = self.fc(x)
        probs = self.act(logits)       # (_, C)

        return logits, probs


    
if __name__ == '__main__':
    bs = 10            # B
    f_dim = 40         # F
    t_dim = 26         # T
    n_classes = 3      # C
    n = 150            # N
    clip_frames = 50   # U
    
    model = TCN_classification(timesteps=t_dim, in_channels=f_dim,
                               n_classes=n_classes,
                               channels=[64, 32, 16],
                               clip_frames=clip_frames,
                               )
    dummy_input = torch.rand(bs, n, f_dim, t_dim)
                           
    # (B, N, F, T)
    out = model(dummy_input)
    # (B, N, C)
    
    print(out[0].size(), out[1].size())
    