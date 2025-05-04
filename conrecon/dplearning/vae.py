from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from conrecon.utils.common import create_logger
import pdb


class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        # Start with normal Normal distribution
        self.N = Normal(torch.zeros(latent_size), torch.ones(latent_size))
        self.kl = 0

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, sigma):
        z = mu + sigma * self.N.sample(mu.shape)
        # CHECK: this to be correct.
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(sigma**2) - 1)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z)


class DP2VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size):
        super(DP2VAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        # Start with normal Normal distribution
        self.N = Normal(torch.zeros(latent_size), torch.ones(latent_size))
        self.kl = 0


class SequentialVAE(nn.Module):

    # ADVERSARY_SOURCE = "DECODED" # For taking decoded output
    ADVERSARY_SOURCE = "LATENT" # For using latent features

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: int,
        num_features_to_guess: int,
        rnn_num_layers: int = 1,
        rnn_hidden_size: int = 32,
    ):
        super(SequentialVAE, self).__init__()

        # This one will have a RNN For encodeing the state data
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.path_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        # Once we have the path encoder 

        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.logger = create_logger(__class__.__name__)

        self.relu = nn.ReLU()

        # Start with normal Normal distribution
        self.N = Normal(0, 1)

    def encode(self, x):
        # This normally expects (batch_size, sequence_length, input_size)
        if len(x.shape) == 3:
            reshaped_x = x.reshape(-1, x.shape[-1])
        elif len(x.shape) == 2:
            reshaped_x = x
        else:
            raise ValueError(f"Shape of x is {x.shape} and its type is {type(x)}")

        # Keep x i
        self.logger.debug(f"Shape of x is {x.shape} and its type is {type(x)}")
        h1 = F.relu(self.fc1(reshaped_x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        # d0 = sigma * self.N.sample(mu.shape).to(sigma.device)
        # z = mu + d0
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        """
        Args:
            - x: The input data (batch_size, sequence_length, input_size)
        Returns:
            - z: The latent representation of the data
            - decoded: The decoded data
            - kl: The KL divergence between the posterior and prior (batch_size,)
        """
        rnn_output, (rnn_hn, rnn_cn)  = self.path_encoder(x)
        rnn_output_flat = rnn_output.reshape(-1, rnn_output.shape[-1])
        mu, logvar = self.encode(rnn_output_flat)
        z = self.reparameterize(mu, logvar)
        var = torch.exp(logvar)
        # kl = 0.5 * (torch.pow(sigma,2) + torch.pow(mu,2) - torch.log(torch.pow(sigma,2)) - 1)
        kl = 0.5 * (var + torch.pow(mu,2) - logvar - 1).reshape(x.shape[0], x.shape[1], -1)
        kl = kl.sum(dim=-1)
        decoded = self.decode(z)
        decoded = decoded.reshape(x.shape)

        return z,decoded, kl
class SeqAdversarialVAE_Failed(nn.Module):

    # ADVERSARY_SOURCE = "DECODED" # For taking decoded output
    ADVERSARY_SOURCE = "LATENT" # For using latent features

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: int,
        num_features_to_guess: int,
        rnn_num_layers: int = 1,
        rnn_hidden_size: int = 32,
    ):
        super(SeqAdversarialVAE, self).__init__()

        # This one will have a RNN For encodeing the state data
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.path_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        # Once we have the path encoder 

        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.logger = create_logger(__class__.__name__)

        # For adversary
        adversary_input_size = latent_size if self.ADVERSARY_SOURCE == "LATENT" else input_size
        self.adversary = nn.Sequential(
            nn.Linear(adversary_input_size, adversary_input_size*2),
            nn.ReLU(),
            nn.Linear(adversary_input_size*2, num_features_to_guess),
        )

        self.relu = nn.ReLU()

        # Start with normal Normal distribution
        self.N = Normal(0, 1)

    def encode(self, x):
        # This normally expects (batch_size, sequence_length, input_size)
        if len(x.shape) == 3:
            reshaped_x = x.reshape(-1, x.shape[-1])
        elif len(x.shape) == 2:
            reshaped_x = x
        else:
            raise ValueError(f"Shape of x is {x.shape} and its type is {type(x)}")

        # Keep x i
        self.logger.debug(f"Shape of x is {x.shape} and its type is {type(x)}")
        h1 = F.relu(self.fc1(reshaped_x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        # d0 = sigma * self.N.sample(mu.shape).to(sigma.device)
        # z = mu + d0
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        """
        Args:
            - x: The input data (batch_size, sequence_length, input_size)
        Returns:
            - decoded: The decoded data
            - adversary_guess: The guessed data from the adversary
            - kl: The KL divergence between the posterior and prior (batch_size,)
        """
        rnn_output, (rnn_hn, rnn_cn)  = self.path_encoder(x)
        rnn_output_flat = rnn_output.reshape(-1, rnn_output.shape[-1])
        mu, logvar = self.encode(rnn_output_flat)
        z = self.reparameterize(mu, logvar)
        var = torch.exp(logvar)
        # kl = 0.5 * (torch.pow(sigma,2) + torch.pow(mu,2) - torch.log(torch.pow(sigma,2)) - 1)
        kl = 0.5 * (var + torch.pow(mu,2) - logvar - 1).reshape(x.shape[0], x.shape[1], -1)
        kl = kl.sum(dim=-1)
        decoded = self.decode(z)
        decoded = decoded.reshape(x.shape)

        # NOTE: See if there is any difference between guessing from latent features vs decoded_features. 
        # guessed_features = self.adversary(z)
        if self.ADVERSARY_SOURCE == "LATENT":
            guessed_features = self.adversary(z)    
        elif self.ADVERSARY_SOURCE == "DECODED":
            guessed_features = self.adversary(
                decoded.reshape(-1, decoded.shape[-1])
            ).reshape(decoded.shape[0], decoded.shape[1],-1)
        else:
            raise ValueError(f"ADVERSARY_SOURCE is {self.ADVERSARY_SOURCE} and it is not valid")

        return decoded, guessed_features, kl

class AdversarialVAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: int,
        num_features_to_guess: int,
    ):
        super(AdversarialVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.logger = create_logger(__class__.__name__)

        # For adversary
        self.adversary = nn.Sequential(
            nn.Linear(latent_size, latent_size*2),
            nn.ReLU(),
            nn.Linear(latent_size*2, num_features_to_guess),
        )
        self.relu = nn.ReLU()

        # Start with normal Normal distribution
        self.N = Normal(0, 1)

    def encode(self, x):
        # This normally expects (batch_size, sequence_length, input_size)
        if len(x.shape) == 3:
            reshaped_x = x.view(-1, x.shape[-1])
        elif len(x.shape) == 2:
            reshaped_x = x
        else:
            raise ValueError(f"Shape of x is {x.shape} and its type is {type(x)}")

        # Keep x i
        self.logger.debug(f"Shape of x is {x.shape} and its type is {type(x)}")
        h1 = F.relu(self.fc1(reshaped_x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, sigma) -> torch.Tensor:
        d0 = sigma * self.N.sample(mu.shape).to(sigma.device)
        z = mu + d0
        # CHECK: this to be correct.
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        """
        Returns:
            - decoded: The decoded data
            - adversary_guess: The guessed data from the adversary
            - kl: The KL divergence between the posterior and prior (batch_size,)
        """
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        kl = 0.5 * torch.pow(sigma**2,2) + torch.pow(mu,2) - torch.log(sigma**2) - 1
        kl = kl.sum(dim=-1)
        decoded = self.decode(z)
        decoded = decoded.view(x.shape)

        guessed_features = self.adversary(z)
        return decoded, guessed_features, kl


class FlexibleVAE(nn.Module):

    def __init__(self, input_size, latent_size, hidden_size):
        super(FlexibleVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.logger = create_logger(__class__.__name__)

        # Start with normal Normal distribution
        self.N = Normal(0, 1)
        self.kl = 0

    def encode(self, x):
        # This normally expects (batch_size, sequence_length, input_size)
        if len(x.shape) == 3:
            reshaped_x = x.view(-1, x.shape[-1])
        elif len(x.shape) == 2:
            reshaped_x = x
        else:
            raise ValueError(f"Shape of x is {x.shape} and its type is {type(x)}")

        # Keep x i
        self.logger.debug(f"Shape of x is {x.shape} and its type is {type(x)}")
        h1 = F.relu(self.fc1(reshaped_x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, sigma):
        self.logger.debug(f"The sigma and mu shapes are ({sigma.shape},{mu.shape})")
        sample = self.N.sample(mu.shape)
        self.logger.debug(f"Shape of sample is {sample.shape}")
        d0 = sigma * self.N.sample(mu.shape).to(sigma.device)
        z = mu + d0
        # CHECK: this to be correct.
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(torch.pow(sigma,2)) - 1)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        decoded = self.decode(z)
        decoded = decoded.view(x.shape)
        return decoded


# Recurrent Variational Autoencoder for Time Series
class RecurrentVAE(nn.Module):
    def __init__(
        self, input_size: int, latent_size: int, hidden_size: int, sequence_length: int
    ):
        self.logger = create_logger(__class__.__name__)
        super(RecurrentVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # Encoder LSTM
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.fc21 = nn.Linear(hidden_size * sequence_length, latent_size)
        # self.fc22 = nn.Linear(hidden_size * sequence_length, latent_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        # Decoder LSTM
        self.fc3 = nn.Linear(latent_size, hidden_size * sequence_length)
        self.lstm_decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

        self.N = Normal(torch.zeros(latent_size), torch.ones(latent_size))
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def encode(self, x):
        # Pass input through LSTM encoder
        rnn_output, _ = self.lstm_encoder(x)
        self.logger.debug(f"The shapes of all the hidden bois are {rnn_output.shape}")
        rnn_output = rnn_output.contiguous().view(
            rnn_output.size(0), -1
        )  # Flatten LSTM output
        return (
            self.fc21(rnn_output).view(rnn_output.shape[0], rnn_output.shape[1], -1),
            self.fc22(rnn_output).view(rnn_output.shape[0], rnn_output.shape[1], -1),
        )

    def reparameterize(self, mu, sigma):
        self.logger.debug(f"The sigma and mu shapes are ({sigma.shape},{mu.shape})")
        sample = self.N.sample(mu.shape)
        self.logger.debug(
            f"Sigze of sigma {sigma.shape} as awell as tha tof the sample {sample.shape}"
        )
        d0 = sigma * sample
        z = mu + d0
        self.kl = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(sigma**2) - 1)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = h3.view(h3.size(0), self.sequence_length, -1)  # Reshape to sequence format
        h3, _ = self.lstm_decoder(h3)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, sigma = self.encode(x)
        # We have the resulting mu and sigma for the sequence to regenerate.
        z = self.reparameterize(mu, sigma)
        return self.decode(z)

class SequenceToOneVector(nn.Module):

    # ADVERSARY_SOURCE = "DECODED" # For taking decoded output
    ADVERSARY_SOURCE = "LATENT" # For using latent features

    def __init__(
        self,
        input_size: int,
        num_sanitized_features: int,
        latent_size: int,
        hidden_size: int,
        rnn_num_layers: int = 1,
        rnn_hidden_size: int = 32,
    ):
        super(SequenceToOneVector, self).__init__()

        # This one will have a RNN For encodeing the state data
        self.path_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        # Once we have the path encoder 

        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_sanitized_features = num_sanitized_features
        self.fc1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_sanitized_features)
        self.logger = create_logger(__class__.__name__)

        self.relu = nn.ReLU()

    def encode(self, x):
        # This normally expects (batch_size, sequence_length, input_size)
        if len(x.shape) == 3:
            reshaped_x = x.reshape(-1, x.shape[-1])
        elif len(x.shape) == 2:
            reshaped_x = x
        else:
            raise ValueError(f"Shape of x is {x.shape} and its type is {type(x)}")

        # Keep x i
        self.logger.debug(f"Shape of x is {x.shape} and its type is {type(x)}")
        h1 = F.relu(self.fc1(reshaped_x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        # d0 = sigma * self.N.sample(mu.shape).to(sigma.device)
        # z = mu + d0
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            - x: The input data (batch_size, sequence_length, input_size)
        Returns:
            - z: The latent representation of the data
            - decoded: The decoded data
            - kl: The KL divergence between the posterior and prior (batch_size,)
        """
        rnn_output, (rnn_hn, rnn_cn)  = self.path_encoder(x)
        rnn_output = rnn_output[:, -1, :] # Now (batch_size, 1, column_size)
        # rnn_output_flat = rnn_output.reshape(-1, rnn_output.shape[-1])
        mu, logvar = self.encode(rnn_output)
        z = self.reparameterize(mu, logvar)
        var = torch.exp(logvar)
        # kl = 0.5 * (torch.pow(sigma,2) + torch.pow(mu,2) - torch.log(torch.pow(sigma,2)) - 1)
        kl = 0.5 * (var + torch.pow(mu,2) - logvar - 1).sum(-1)
        decoded = self.decode(z)

        return z,decoded, kl

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequence data.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 1000,
    ):
        super(TransformerEncoder, self).__init__()
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, max_seq_length, hidden_size)
        )
        self._init_positional_encoding(max_seq_length, hidden_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.hidden_size = hidden_size
        self.logger = create_logger(__class__.__name__)
    
    def _init_positional_encoding(self, max_seq_length, hidden_size):
        """Initialize positional encodings."""
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * 
            (-math.log(10000.0) / hidden_size)
        )
        
        pos_encoding = torch.zeros(max_seq_length, hidden_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        if hidden_size % 2 == 0:  # Handle even hidden size
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
        else:  # Handle odd hidden size
            pos_encoding[:, 1::2] = torch.cos(position * div_term[:hidden_size//2])
        
        self.pos_encoder.data = pos_encoding.unsqueeze(0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            Encoded sequence of shape (batch_size, seq_length, hidden_size)
        """
        # Create padding mask (1 for padding, 0 for actual data)
        # For now, we assume all sequences are valid (no padding)
        seq_length = x.size(1)
        
        # Embed input and add positional encoding
        x = self.input_embedding(x)
        x = x + self.pos_encoder[:, :seq_length, :]
        
        # Pass through transformer encoder
        output = self.transformer_encoder(x)
        
        # Return the output for the entire sequence
        return output


class SequenceToOneVector(nn.Module):
    """
    A VAE that encodes a sequence to a single vector representation.
    Supports multiple sequence processing architectures: LSTM, BiLSTM, or Transformer.
    """
    # Supported sequence processor types
    PROCESSOR_LSTM = "lstm"
    PROCESSOR_BILSTM = "bilstm"
    PROCESSOR_TRANSFORMER = "transformer"
    
    # For adversarial models
    ADVERSARY_SOURCE = "LATENT"  # For using latent features

    def __init__(
        self,
        input_size: int,
        num_sanitized_features: int,
        latent_size: int,
        hidden_size: int,
        seq_processor_type: str = "lstm",  # Options: "lstm", "bilstm", "transformer"
        rnn_num_layers: int = 1,
        rnn_hidden_size: int = 32,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 2,
        transformer_dropout: float = 0.1,
        max_seq_length: int = 1000,
    ):
        super(SequenceToOneVector, self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_sanitized_features = num_sanitized_features
        self.seq_processor_type = seq_processor_type.lower()
        self.logger = create_logger(__class__.__name__)
        
        # Determine the output size of the sequence processor
        if self.seq_processor_type == self.PROCESSOR_BILSTM:
            seq_processor_output_size = rnn_hidden_size * 2  # BiLSTM outputs concatenated forward/backward
        else:
            seq_processor_output_size = rnn_hidden_size
        
        # Initialize the appropriate sequence processor
        if self.seq_processor_type == self.PROCESSOR_LSTM:
            self.seq_processor = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
            )
            self.logger.info(f"Using LSTM sequence processor with {rnn_num_layers} layers")
        
        elif self.seq_processor_type == self.PROCESSOR_BILSTM:
            self.seq_processor = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.logger.info(f"Using BiLSTM sequence processor with {rnn_num_layers} layers")
        
        elif self.seq_processor_type == self.PROCESSOR_TRANSFORMER:
            self.seq_processor = TransformerEncoder(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                dropout=transformer_dropout,
                max_seq_length=max_seq_length,
            )
            self.logger.info(f"Using Transformer sequence processor with {transformer_num_layers} layers and {transformer_num_heads} heads")
        
        else:
            raise ValueError(f"Unknown sequence processor type: {seq_processor_type}. "
                            f"Supported types: {self.PROCESSOR_LSTM}, {self.PROCESSOR_BILSTM}, {self.PROCESSOR_TRANSFORMER}")
        
        # VAE encoder components
        self.fc1 = nn.Linear(seq_processor_output_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)  # mu
        self.fc22 = nn.Linear(hidden_size, latent_size)  # logvar
        
        # VAE decoder components
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_sanitized_features)
        
        self.relu = nn.ReLU()

    def encode(self, x):
        """
        Encode the input to the latent space.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim) or (batch_size, seq_length, feature_dim)
        
        Returns:
            mu, logvar: Parameters of the latent distribution
        """
        # Ensure x has the right shape
        if len(x.shape) == 3:
            reshaped_x = x.reshape(-1, x.shape[-1])
        elif len(x.shape) == 2:
            reshaped_x = x
        else:
            raise ValueError(f"Shape of x is {x.shape} and its type is {type(x)}")

        self.logger.debug(f"Shape of x is {x.shape} and its type is {type(x)}")
        h1 = F.relu(self.fc1(reshaped_x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        Decode the latent vector to the output space.
        
        Args:
            z: Latent vector
        
        Returns:
            Reconstructed output
        """
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def _process_sequence_lstm(self, x):
        """Process sequence with LSTM and return the final hidden state."""
        output, (hn, cn) = self.seq_processor(x)
        # Use the last output from the sequence
        return output[:, -1, :]
    
    def _process_sequence_bilstm(self, x):
        """Process sequence with BiLSTM and return the final hidden state."""
        output, (hn, cn) = self.seq_processor(x)
        # For BiLSTM, we get the last output from both directions
        # The shape of output is (batch_size, seq_len, hidden_size*2)
        return output[:, -1, :]
    
    def _process_sequence_transformer(self, x):
        """Process sequence with Transformer and return a summary vector."""
        # Process the entire sequence
        output = self.seq_processor(x)
        
        # For transformers, we have several options to get a single vector:
        # 1. Use the last token's representation (like BERT's [CLS] token)
        # 2. Use mean pooling across the sequence
        # 3. Use max pooling across the sequence
        
        # Here we'll use the last token's representation for consistency with LSTM
        return output[:, -1, :]
        
        # Alternative: mean pooling
        # return torch.mean(output, dim=1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: The input data (batch_size, sequence_length, input_size)
        
        Returns:
            z: The latent representation of the data
            decoded: The decoded data
            kl: The KL divergence between the posterior and prior (batch_size,)
        """
        # Process the sequence based on the selected processor type
        if self.seq_processor_type == self.PROCESSOR_LSTM:
            seq_features = self._process_sequence_lstm(x)
        elif self.seq_processor_type == self.PROCESSOR_BILSTM:
            seq_features = self._process_sequence_bilstm(x)
        elif self.seq_processor_type == self.PROCESSOR_TRANSFORMER:
            seq_features = self._process_sequence_transformer(x)
        else:
            raise ValueError(f"Unknown sequence processor type: {self.seq_processor_type}")
        
        # Encode to latent space
        mu, logvar = self.encode(seq_features)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Calculate KL divergence
        var = torch.exp(logvar)
        kl = 0.5 * (var + torch.pow(mu, 2) - logvar - 1).sum(-1)
        
        # Decode from latent space
        decoded = self.decode(z)
        
        return z, decoded, kl
