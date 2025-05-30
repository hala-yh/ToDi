import json
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================================
# KL Divergence loss
def kld_loss(mu, logvar):
    """
    mu: Means of encoder output [batch_size, latent_size]
    logvar: log varances of encoder output [batch_size, latent_size]
    returns:
        KLD of the specified distribution and a unit Gaussian.
    """

    mu = mu.double().to(get_device())
    logvar = logvar.double().to(get_device())

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kld

# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
def symbol2hsa(input_symbol):
    with open('datasets/tools/symbol2hsa.json', mode='rt', encoding='utf-8')as f:
        symbol_data = json.load(f)
        symbols = list(symbol_data.keys())
    hsas = []
    for sym in input_symbol:
        if sym in symbols:
            hsas.append(symbol_data[sym])
        else:
            hsas.append('-')
    return hsas
def common(df_tgt, gene_type):
    # Source gene names
    df_source = pd.read_csv('datasets/tools/source_genes.csv', sep=',')
    source_hsas = list(df_source.columns)
    # Target gene names
    tgt_hsas = list(df_tgt.columns)

    if not gene_type == 'gene_symbol':
        tgt_hsas = symbol2hsa(tgt_hsas)
        df_tgt = df_tgt.set_axis(tgt_hsas, axis=1)

    # Common gene names
    common_hsas = list(set(tgt_hsas) & set(source_hsas))
    common_hsas = sorted(common_hsas, key=source_hsas.index)
    # Processed target gene expression profile data
    df_source[common_hsas] = df_tgt[common_hsas]

    return df_source
# ============================================================================

# Create a VAE encoder
class GeneEncoder(nn.Module):

    def __init__(
        self, 
        input_size,
        hidden_sizes,
        latent_size,
        activation_fn,
        dropout
    ):
        """
        input_size: number of gene columns (eg. 978)
        hidden_sizes: number of neurons of stack dense layers
        latent_size: size of the latent vector
        activation_fn: activation function
        dropout: dropout probabilites
        """
        super(GeneEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.activation_fn = activation_fn
        self.dropout = [dropout] * len(self.hidden_sizes)

        num_units = [self.input_size] + self.hidden_sizes
        
        dense_layers = []
        for index in range(1, len(num_units)):
            dense_layers.append(nn.Linear(num_units[index-1], num_units[index]))
            dense_layers.append(self.activation_fn)

            if self.dropout[index-1] > 0.0:
                dense_layers.append(nn.Dropout(p=self.dropout[index-1]))

        self.encoding = nn.Sequential(*dense_layers)

        self.encoding_to_mu = nn.Linear(self.hidden_sizes[-1], self.latent_size)
        self.encoding_to_logvar = nn.Linear(self.hidden_sizes[-1], self.latent_size)

    def forward(self, inputs):
        """
        inputs: [batch_size, input_size]
        returns: 
            mu: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
        """
        projection = self.encoding(inputs)
        mu = self.encoding_to_mu(projection)
        logvar = self.encoding_to_logvar(projection)

        return (mu, logvar)

# ============================================================================
# Create a VAE decoder
class GeneDecoder(nn.Module):

    def __init__(
        self,
        latent_size,
        hidden_sizes,
        output_size,
        activation_fn,
        dropout

    ):
        """
        latent_size: size of the latent vector
        hidden_sizes: number of neurons of stack dense layers
        output_size: number of gene columns (eg. 978)
        activation_fn: activation function
        dropout: dropout probabilites
        """
        super(GeneDecoder, self).__init__()

        self.latent_size = latent_size
        # Reverse the number of neurons of dense layers
        hidden_sizes.reverse()
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.dropout = [dropout] * len(self.hidden_sizes)

        num_units = [self.latent_size] + self.hidden_sizes + [self.output_size]
        # num_units = [64, 512, 256, 128, 978]
        dense_layers = []
        # Last layer does not use dropout but requires a sigmoid function
        for index in range(1, len(num_units)-1):
            dense_layers.append(nn.Linear(num_units[index-1], num_units[index]))
            dense_layers.append(self.activation_fn)
            #dense_layers.append(nn.BatchNorm1d(num_units[index]))

            if self.dropout[index-1] > 0.0:
                dense_layers.append(nn.Dropout(p=self.dropout[index-1]))

        # Last layer
        dense_layers.append(nn.Linear(num_units[-2], num_units[-1]))
        #dense_layers.append(nn.Sigmoid())

        self.decoding = nn.Sequential(*dense_layers)

    def forward(self, latent_z):
        """
        latent_z: [batch_size, latent_size]
        returns:
            reconstructed inputs: [batch_size, input_size]
        """
        outputs = self.decoding(latent_z)

        return outputs

# ============================================================================
# Create a VAE to extract features of gene expression
class GeneVAE(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_sizes,
        latent_size,
        output_size,
        activation_fn,
        dropout
    ):
        """
        input_size: number of gene columns (eg. 978)
        latent_size: size of the latent vector
        hidden_sizes: number of neurons of stack dense layers
        output_size: number of gene columns (eg. 978)
        activation_fn: activation function
        dropout: dropout probability
        """
        super(GeneVAE, self).__init__()

        self.encoder = GeneEncoder(
            input_size,
            hidden_sizes,
            latent_size,
            activation_fn,
            dropout
        )
        self.decoder = GeneDecoder(
            latent_size,
            hidden_sizes,
            output_size,
            activation_fn,
            dropout
        )
        self.reconstruction_loss = nn.MSELoss(reduction='sum')
        self.kld_loss = kld_loss

    def reparameterize(self, mu, logvar):
        """
        Apply reparameterization trick to obtain samples from latent space.
        returns:
            sampled Z from the latnet distribution
        """
        return torch.randn_like(mu).mul_(torch.exp(0.5*logvar)).add_(mu)

    def forward(self, inputs):
        """
        inputs: [batch_size, input_size]
        returns:
            output samples: [batch_size, input_size]
        """
        self.mu, self.logvar = self.encoder(inputs)
        latent_z = self.reparameterize(self.mu, self.logvar)
        outputs = self.decoder(latent_z)

        return latent_z, outputs

    def joint_loss(
        self,
        outputs,
        targets,
        alpha=0.5,
        beta=1
    ):
        """
        outputs: decoder outputs [batch_size, input_size]
        targets: encoder inputs [batch_size, input_size]
        alpha: L2 loss
        beta: Scaling of the KLD in range [1, 100]
        returns:
            joint_loss, rec_loss, kld_loss
        """
        rec_loss = self.reconstruction_loss(outputs, targets)
        rec_loss = rec_loss.double().to(get_device())

        kld_loss = self.kld_loss(self.mu, self.logvar)

        joint_loss = alpha * rec_loss + (1 - alpha) * beta * kld_loss

        return joint_loss, rec_loss, kld_loss

    def load_model(self, path):
        weights = torch.load(path, map_location=get_device())
        #weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)















