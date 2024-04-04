import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class FCEncoder(nn.Module):
    def __init__(self, n_features, embedding_size, dropout_rate=0.1):
        super(FCEncoder, self).__init__()
        # Sizes of the fully connected layers
        layer_sizes = [n_features, embedding_size*30, embedding_size*20, embedding_size*20, embedding_size*10, embedding_size]
        # Create the fully connected layers dynamically
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(nn.Linear(in_features, out_features))
            self.batch_norms.append(nn.BatchNorm1d(out_features))  # add Batch Normalization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply ReLU activation function after each linear transformation
        for layer, batch_norm in zip(self.layers[:-1], self.batch_norms[:-1]):  
            x = F.relu(batch_norm(layer(x)))
            x = self.dropout(x)
        x = F.relu(self.layers[-1](x)) 
        return x

class FCDecoder(nn.Module):
    def __init__(self, embedding_size, n_features, dropout_rate=0.1):
        super(FCDecoder, self).__init__()
        # Sizes of the fully connected layers in reverse
        layer_sizes = [embedding_size, embedding_size*10, embedding_size*20, embedding_size*20, embedding_size*30, n_features]
        # Create the fully connected layers dynamically
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(nn.Linear(in_features, out_features))

            if out_features != n_features:  
                self.batch_norms.append(nn.BatchNorm1d(out_features))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply ReLU activation function for all but the last layer, apply Sigmoid for the last layer
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)  
            x = self.dropout(x)
        # Apply sigmoid to the output of the last layer, without dropout
        x = torch.sigmoid(self.layers[-1](x))
        return x    

class DenoiseAutoEncoder(nn.Module):
    def __init__(self, n_features, embedding_dim, dropout_rate=0.1):
        super(DenoiseAutoEncoder, self).__init__()
        self.encoder = FCEncoder(n_features, embedding_dim, dropout_rate)  # Use original FCEncoder
        self.decoder = FCDecoder(embedding_dim, n_features, dropout_rate)  # Use original FCDecoder
        # Apply weight initialization
        self.apply(initialize_weights)
    
    def forward(self, x):
        # Introduce noise to the input
        noisy_x = x + 0.1 * torch.randn_like(x)
        encoded = self.encoder(noisy_x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    


class FCAutoEncoder(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super(FCAutoEncoder, self).__init__()
        self.encoder = FCEncoder(n_features, embedding_dim)
        self.decoder = FCDecoder(embedding_dim, n_features)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded  # Returns the encoded embedded representation and the decoded output





class DynamicLSTMEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_size, e_hidden_size_multiplier):
        super(DynamicLSTMEncoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_size = embedding_size
        self.hidden_size_multiplier = e_hidden_size_multiplier  # Multipliers for hidden sizes in each LSTM layer
        
        # Dynamically create LSTM layers
        lstm_sizes = [n_features] + [embedding_size * m for m in self.hidden_size_multiplier]
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=lstm_sizes[i], hidden_size=lstm_sizes[i+1], num_layers=1, batch_first=True)
            for i in range(len(lstm_sizes) - 1)
        ])
        self.final_lstm = nn.LSTM(input_size=lstm_sizes[-1], hidden_size=embedding_size, num_layers=1, batch_first=True)

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x, (hidden_state, _) = self.final_lstm(x)
        # print(hidden_state.shape)
        return hidden_state[-1, :, :]

class DynamicLSTMDecoder(nn.Module):
    def __init__(self, seq_len, embedding_size, n_features, d_hidden_size_multiplier):
        super(DynamicLSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.n_features = n_features
        self.hidden_size_multiplier = d_hidden_size_multiplier
        
        # Dynamically create LSTM layers
        # lstm_sizes = [embedding_size, 15 * embedding_size, 30 * embedding_size]
        lstm_sizes = [embedding_size] + [embedding_size * m for m in self.hidden_size_multiplier]
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=lstm_sizes[i], hidden_size=lstm_sizes[i+1], num_layers=1, batch_first=True)
            for i in range(len(lstm_sizes) - 1)
        ])
        # self.final_lstm = nn.LSTM(input_size=lstm_sizes[-1], hidden_size=lstm_sizes[-1] // 2, num_layers=1, batch_first=True)

        # self.fc = nn.Linear(lstm_sizes[-1] // 2, n_features)
        self.final_lstm = nn.LSTM(input_size=lstm_sizes[-1], hidden_size=2 * lstm_sizes[-1], num_layers=1, batch_first=True)
        self.fc = nn.Linear(2 * lstm_sizes[-1], n_features)
        self.fc 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x, _ = self.final_lstm(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class LSTM_AE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, e_hidden_size_multiplier, d_hidden_size_multiplier):
        super(LSTM_AE, self).__init__()
        self.encoder = DynamicLSTMEncoder(seq_len, n_features, embedding_dim, e_hidden_size_multiplier)
        self.decoder = DynamicLSTMDecoder(seq_len, embedding_dim, n_features, d_hidden_size_multiplier)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded  # Return encoded latent features and decoded output






class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TransformerAutoEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Positional Encoding
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(max_seq_length, d_model), requires_grad=False)

        # Encoder part
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder part
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layers for encoding inputs and decoding outputs
        self.input_fc = nn.Linear(input_dim, d_model)
        self.output_fc = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # Input shape: (batch_size, seq_length, input_dim)
        src = self.input_fc(src)  # Shape: (batch_size, seq_length, d_model)
        src += self.positional_encoding[:src.size(1), :]

        # Encoding
        encoded_memory = self.transformer_encoder(src)  # Encoded representations

        # Decoding - Using encoded memory as both src (input) and tgt (target) for the decoder
        decoded_output = self.transformer_decoder(encoded_memory, encoded_memory)
        decoded_output = self.output_fc(decoded_output)  # Shape: (batch_size, seq_length, input_dim), same as input

        # Apply Sigmoid activation to ensure the output range is (0, 1)
        decoded_output = torch.sigmoid(decoded_output)


        # Return both encoded and decoded outputs
        return encoded_memory, decoded_output


    def _generate_positional_encoding(self, length, d_model):
        """Generate positional encoding for a given length and model dimension"""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # Add batch dimension



class TransformerDenoiseAutoEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TransformerDenoiseAutoEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Encoder and Decoder parts
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layers for inputs and outputs
        self.input_fc = nn.Linear(input_dim, d_model)
        self.output_fc = nn.Linear(d_model, input_dim)

    def forward(self, src, noise_factor=0.1):
        # Add Gaussian noise
        noise = torch.randn_like(src) * noise_factor
        noisy_src = src + noise

        noisy_src = self.input_fc(noisy_src)  # Shape: (batch_size, seq_length, d_model)
        seq_length = noisy_src.size(1)
        positional_encoding = self._generate_positional_encoding(seq_length, self.d_model).to(noisy_src.device)
        noisy_src += positional_encoding

        # Encoding and Decoding
        encoded_memory = self.transformer_encoder(noisy_src)
        decoded_output = self.transformer_decoder(encoded_memory, encoded_memory)
        decoded_output = self.output_fc(decoded_output)
        decoded_output = torch.sigmoid(decoded_output)

        return encoded_memory, decoded_output

    def _generate_positional_encoding(self, length, d_model):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # Add batch dimension
