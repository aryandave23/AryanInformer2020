import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

###############################################################################
# 1. Channel Positional Encoding
###############################################################################

class ChannelPositionalEmbedding(nn.Module):
    """
    Uses the sinusoidal logic (like in PositionalEmbedding) to generate a 
    (c_in, m+1) matrix. It then flattens it to (1, c_in*(m+1)) and, given an 
    integer n (e.g., the sequence length), repeats that vector n times to yield 
    a (n, c_in*(m+1)) tensor.
    """
    def __init__(self, c_in): #here original was (self, c_in, m)
        """
        Args:
            c_in (int): Number of channels.
            m (int): The extra offset parameter so that each channel has (m+1) columns.
        """
        super(ChannelPositionalEmbedding, self).__init__()
        self.c_in = c_in
        #self.m = m
        ma=8
        # Create a (c_in, m+1) matrix using sin and cos functions.
        pe = torch.zeros(c_in, ma+1).float()
        pe.requires_grad = False  # fixed embedding

        # Here we treat the channel index as the “position” and m+1 as the “dimension”.
        position = torch.arange(0, c_in).float().unsqueeze(1)  # shape: (c_in, 1)
        # We use a div_term computed over (m+1) dimensions.
        div_term = torch.exp(torch.arange(0, ma+1, 2).float() * -(math.log(10000.0) / (ma+1)))
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].size(1)])
        if (ma+1) > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe)

    def forward(self, n):
        """
        Args:
            n (int): The number of rows (e.g., time steps) in your dataset.
        Returns:
            Tensor of shape (n, c_in*(m+1))
        """
        # Flatten the (c_in, m+1) matrix to (1, c_in*(m+1))
        flat = self.pe.flatten().unsqueeze(0)
        # Repeat it n times along the batch/time dimension.
        return flat.repeat(n, 1)


###############################################################################
# 2. Modified TokenEmbedding
###############################################################################

class TokenEmbedding(nn.Module):
    """
    Original token embedding that extracts features from the raw input time-series.
    Now modified so that it can also receive a precomputed channel encoding,
    concatenate it to the extracted features, and then project back to d_model.
    """
    def __init__(self, c_in, d_model, tao=1, m=0, pad=True, is_split=False):
        """
        Args:
            c_in (int): Number of input channels.
            d_model (int): Desired embedding dimension.
            tao (int): Time offset parameter.
            m (int): Determines that each channel provides (m+1) columns.
            pad (bool): Whether to pad the extracted tokens.
            is_split (bool): Whether to use the split convolutions.
        """
        super(TokenEmbedding, self).__init__()
        self.tao = tao
        self.m = m
        self.m = int(m)
        self.d_model = d_model
        self.pad = pad
        self.c_in = c_in
        self.is_split = is_split
        self.kernels = int(d_model / c_in)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv = nn.Conv1d(in_channels=m+1, out_channels=self.kernels, 
                              kernel_size=3, padding=1, padding_mode='circular').to(self.device)
        self.leftout_conv = nn.Conv1d(in_channels=m+1, 
                                      out_channels=self.d_model - self.c_in * self.kernels, 
                                      kernel_size=3, padding=1, padding_mode='circular').to(self.device)
        self.total_conv = nn.Conv1d(in_channels=c_in*(m+1), 
                                      out_channels=self.d_model, 
                                      kernel_size=3, padding=1, padding_mode='circular').to(self.device)

        # Additional projection layer: after concatenating the channel encoding 
        # (which has dimension c_in*(m+1)) with the token embedding (d_model), 
        # we project back to d_model.
        self.concat_proj = nn.Linear(d_model + c_in*(m+1), d_model).to(self.device)
        
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def data_extract(self, ts_batch):
        """
        Extracts a “faithful” vector from a time-series batch using a sliding-window
        approach. (This is the original logic you provided.)
        
        Args:
            ts_batch (Tensor): shape (seq_len, c_in)
        Returns:
            Tensor: shape (n_valid, c_in*(m+1))
        """
        n_seq, cin = ts_batch.shape
        n_valid = n_seq - self.m * self.tao
        if n_valid <= 0:
            raise ValueError(f"Invalid n_valid={n_valid}. Check seq_length, m, and tao values.")
        t_indices = torch.arange(self.m * self.tao, n_seq, device=ts_batch.device)
        offsets = torch.arange(0, self.m + 1, device=ts_batch.device) * self.tao  
        time_indices = t_indices.unsqueeze(1) - offsets.unsqueeze(0)
        channel_idx = torch.arange(cin, device=ts_batch.device).view(1, cin, 1).expand(n_valid, cin, self.m + 1)
        time_idx_expanded = time_indices.unsqueeze(1).expand(n_valid, cin, self.m + 1)
        extracted = ts_batch[time_idx_expanded, channel_idx]
        faithful_vec = extracted.reshape(n_valid, cin * (self.m + 1))
        return faithful_vec

    def forward(self, x, channel_encoding=None):
        """
        Args:
            x (Tensor): Raw input of shape (batch_size, seq_len, c_in)
            channel_encoding (Tensor, optional): 
                Precomputed channel encoding of shape either 
                (seq_len, c_in*(m+1)) or (batch_size, seq_len, c_in*(m+1)).
        Returns:
            Tensor: Embedded tokens of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, cin = x.shape
        x_list = []
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)
        for batch_val in range(batch_size):
            ts_batch = x[batch_val]  # shape: (seq_len, c_in)
            try:
                extracted_data = self.data_extract(ts_batch)
                x_list.append(extracted_data)
            except Exception as e:
                print(f"Error in data_extract for batch {batch_val}: {e}", flush=True)
                raise
        x_embedded = torch.stack(x_list)  # shape: (batch_size, n_valid, c_in*(m+1))
        if self.pad:
            x_embedded = F.pad(x_embedded, (0, 0, self.m * self.tao, 0))
        if self.is_split:
            x_embedded_parts = torch.split(x_embedded, self.m + 1, dim=2)
            conv_outs = []
            for j, part in enumerate(x_embedded_parts):
                conv_in = part.permute(0, 2, 1)
                out = self.conv(conv_in)
                conv_outs.append(out)
                if j == (len(x_embedded_parts) - 1):
                    leftout_out = self.leftout_conv(conv_in)
                    conv_outs.append(leftout_out)
            x_embedded = torch.cat(conv_outs, dim=1).transpose(1, 2)
        else:
            x_embedded = self.total_conv(x_embedded.permute(0,2,1)).transpose(1,2)
        
        # --- Incorporate Channel Encoding ---
        if channel_encoding is not None:
            # Ensure channel_encoding has shape (batch_size, seq_len, c_in*(m+1))
            if channel_encoding.dim() == 2:
                channel_encoding = channel_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            # Concatenate along the last dimension
            x_embedded = torch.cat([x_embedded, channel_encoding], dim=-1)
            # Project back to d_model so the resulting dimension matches other embeddings.
            x_embedded = self.concat_proj(x_embedded)
        return x_embedded

###############################################################################
# 3. Modified DataEmbedding
###############################################################################

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x is used only to determine the sequence length.
        return self.pe[:, :x.size(1)]

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

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)
    def forward(self, x):
        return self.emb(x).detach()

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
