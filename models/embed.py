import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__>='1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
#                                     kernel_size=3, padding=padding, padding_mode='circular')
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
#         return x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, tao=1, m=0, pad=True, is_split=False):
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in
        self.is_split = is_split
        self.kernels = int(d_model / c_in)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the embedding layers on the proper device
        self.conv = nn.Conv1d(in_channels=m+1, out_channels=self.kernels, 
                              kernel_size=3, padding=1, padding_mode='circular').to(self.device)
        self.leftout_conv = nn.Conv1d(in_channels=m+1, 
                                      out_channels=self.d_model - self.c_in * self.kernels, 
                                      kernel_size=3, padding=1, padding_mode='circular').to(self.device)
        self.total_conv = nn.Conv1d(in_channels=c_in*(m+1), 
                                      out_channels=self.d_model, 
                                      kernel_size=3, padding=1, padding_mode='circular').to(self.device)

        # Weight initialization for all conv layers
        for m_module in self.modules():
            if isinstance(m_module, nn.Conv1d):
                nn.init.kaiming_normal_(m_module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def data_extract(self, ts_batch):
        """
        Vectorized extraction of faithful vectors.
        """
        #print("Inside data_extract, ts_batch.shape:", ts_batch.shape, flush=True)
        # ts_batch is assumed to be already on self.device with shape (n_seq, c_in)
        n_seq, cin = ts_batch.shape
        n_valid = n_seq - self.m * self.tao  # valid time indices

        if n_valid <= 0:
            raise ValueError(f"Invalid n_valid={n_valid}. Check seq_length, m, and tao values.")

        # Create time indices
        t_indices = torch.arange(self.m * self.tao, n_seq, device=ts_batch.device)

        # Create offsets and compute time indices
        offsets = torch.arange(0, self.m + 1, device=ts_batch.device) * self.tao  
        time_indices = t_indices.unsqueeze(1) - offsets.unsqueeze(0)

        # Create a channel index tensor
        channel_idx = torch.arange(cin, device=ts_batch.device).view(1, cin, 1).expand(n_valid, cin, self.m + 1)
        time_idx_expanded = time_indices.unsqueeze(1).expand(n_valid, cin, self.m + 1)

        # Extract values using advanced indexing
        extracted = ts_batch[time_idx_expanded, channel_idx]
        faithful_vec = extracted.reshape(n_valid, cin * (self.m + 1))

        return faithful_vec

    def forward(self, x):
        """Forward pass of the TokenEmbedding layer."""

        batch_size, seq_len, cin = x.shape

        x_list = []

        # Convert input to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        # Process each batch entry separately
        for batch_val in range(batch_size):
            ts_batch = x[batch_val]  # shape: (seq_len, c_in)

            try:
                extracted_data = self.data_extract(ts_batch)
                x_list.append(extracted_data)
            except Exception as e:
                print(f"Error in data_extract for batch {batch_val}: {e}", flush=True)
                raise

        # Stack along the batch dimension
        x_embedded = torch.stack(x_list)

        # Padding along time dimension if needed
        if self.pad:
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))
     
        #print(f'Faithful vector shape: {x_embedded.shape}', flush=True)
        if self.is_split == True:
            # Split last dimension for convolution
            x_embedded1 = torch.split(x_embedded, self.m + 1, dim=2)
    
            channel_splitter = []
    
            # Process each split through convolutions
            for j, part in enumerate(x_embedded1):
                conv_in = part.permute(0, 2, 1)
                conv_out = self.conv(conv_in)
                channel_splitter.append(conv_out)
    
                if j == (len(x_embedded1) - 1):
                    leftout_out = self.leftout_conv(conv_in)
                    channel_splitter.append(leftout_out)
    
            # Concatenate and transpose back
            x_embedded = torch.cat(channel_splitter, dim=1).transpose(1, 2)

        else:
            x_embedded = self.total_conv(x_embedded.permute(0,2,1)).transpose(1,2)
        return x_embedded





class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
