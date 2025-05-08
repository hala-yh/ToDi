from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch
import numpy as np
import torch as th
import torch.nn as nn
from .GeneVAE import GeneVAE
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
    checkpoint,
)

print('checkpoint 0810 in model.py')
class TransformerNetModel2(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        dropout=0.1,
        vocab_size=None,
        num_heads=8,
        hidden_size=768,
        num_attention_heads = 12,
        num_hidden_layers=12,
        mask = False
    ):
        super().__init__()
        config = AutoConfig.from_pretrained('.../bert-base-uncased')
        config.is_decoder=True
        config.add_cross_attention=True
        config.hidden_dropout_prob = 0.1
        config.hidden_size = hidden_size
        config.num_attention_heads = num_attention_heads
        config.num_hidden_layers = num_hidden_layers

        self.vae_latent_proj = nn.Linear(1042, config.hidden_size)
        self.vae_model = GeneVAE(
            input_size=978,
            hidden_sizes=[512, 256, 128],
            latent_size=64,
            output_size=978,
            activation_fn=nn.ReLU(),
            dropout=0.2
        )

        vae_state_dict = torch.load('.../saved_gene_vae_train_gen3.pkl')
        self.vae_model.load_state_dict(vae_state_dict)
        self.vae_model.eval()

        self.mask = mask
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.dropout = dropout
        self.num_classes = None
        self.use_checkpoint = False
        self.num_heads_upsample = 4
        self.logits_mode = 1
        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)

        self.lm_head = nn.Linear(self.in_channels, vocab_size)
        self.lm_head.weight = self.word_embedding.weight

        self.conditional_gen = False

        self.desc_down_proj = nn.Sequential(
            linear(768,config.hidden_size),
            SiLU(),
            linear(config.hidden_size, config.hidden_size),
        )
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size))
        self.input_transformers = BertEncoder(config)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size*2,
                                                        config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, in_channels))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_embeds_with_deep(self, input_ids):
        atom , deep = input_ids
        atom = self.word_embedding(atom)
        deep = self.deep_embedding(deep)
        return torch.concat([atom, deep], dim=-1)

    def get_logits_deep(self, hidden_repr):
        return self.deep_head(hidden_repr)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1))
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def cosine_similarity_loss(self, hidden_repr, latent_rep):
        hidden_repr_normalized = F.normalize(hidden_repr, p=2, dim=-1)
        latent_rep_normalized = F.normalize(latent_rep, p=2, dim=-1)
        cos_sim = F.cosine_similarity(hidden_repr_normalized, latent_rep_normalized, dim=-1)
        loss = 1 - cos_sim
        return loss.mean()

    def forward(self, x, timesteps, desc_state, desc_mask , gene_expression, y=None):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.mask:
            desc_state = torch.where(timesteps.reshape(-1,1,1)<200,0.,desc_state)
            assert(len(desc_mask.shape)==2)
            desc_mask = torch.where(timesteps.reshape(-1,1)<200,1.,desc_mask)

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        desc_state = self.dropout(self.LayerNorm(self.desc_down_proj(desc_state)))
        desc_mask = desc_mask.unsqueeze(1).unsqueeze(2)
        desc_mask = desc_mask.expand(-1, num_heads, -1, -1)

        output = self.input_transformers(emb_inputs, encoder_hidden_states=desc_state, encoder_attention_mask=desc_mask)
        input_trans_hidden_states = output.last_hidden_state

        mu, logvar = self.vae_model(gene_expression)

        latent_rep = torch.cat([mu, logvar], dim=-1)
        latent_rep = self.vae_latent_proj(latent_rep)

        expanded_gene_expression = latent_rep.unsqueeze(1).expand(-1, seq_length, -1)

        adjusted_output = torch.cat((input_trans_hidden_states, expanded_gene_expression), dim=-1)

        cos_sim_loss = self.cosine_similarity_loss(input_trans_hidden_states, expanded_gene_expression)
        h = self.output_down_proj(adjusted_output)
        h = h.type(x.dtype)

        return h, cos_sim_loss
