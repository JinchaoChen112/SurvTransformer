import numpy as np
import torch
from torch import nn
from models.layers.attention_network import FeedForward, MixedAttentionLayer

def SNN_Block(dim1, dim2, dropout=0.25):

    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))

class SurvTransformer(nn.Module):
    def __init__(
            self,
            omic_sizes=[],
            wsi_embedding_dim=1024,
            dropout=0.1,
            num_classes=4,
            wsi_projection_dim=256,
            omic_names=[],
    ):
        super(SurvTransformer, self).__init__()

        self.num_celltypes = len(omic_sizes)
        self.dropout = dropout

        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim)
        )

        self.init_per_path_model(omic_sizes)

        self.identity = nn.Identity()

        self.attention_network = MixedAttentionLayer(
            dim=self.wsi_projection_dim,
            heads=1,
            kernel_size=5,
            num_celltypes=self.num_celltypes
        )

        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim*2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim*2)

        self.to_logits = nn.Sequential(
            nn.Linear(int(self.wsi_projection_dim*2), int(self.wsi_projection_dim)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim), self.num_classes)
        )

    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        cell_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            cell_networks.append(nn.Sequential(*fc_omic))
        self.cell_networks = nn.ModuleList(cell_networks)

    def forward(self, **kwargs):

        wsi = kwargs['x_path']
        num_path_bag = wsi.shape[1]
        mask = None
        return_attn = kwargs["return_attn"]
        contrast_loss = kwargs["contrast_loss"]
        batch_omic_bag_list = []
        for i_b in range(1, wsi.shape[0] + 1):
            x_omic = [kwargs['batch%s' % i_b]['x_omic%d' % i] for i in range(1, self.num_celltypes + 1)]
            h_omic = [self.cell_networks[idx].forward(cell_feat) for idx, cell_feat in enumerate(x_omic)]
            h_omic_bag = torch.stack(h_omic).unsqueeze(0)
            batch_omic_bag_list.append(h_omic_bag)

        batch_omic_bag = torch.stack(batch_omic_bag_list).squeeze(1)

        wsi_embed = self.wsi_projection_net(wsi)

        tokens = torch.cat([batch_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            out, selfattn_celltypes, cross_attn_celltypes, cross_attn_path = self.attention_network(x=tokens, mask=mask if mask is not None else None,return_attention=True)
        else:
            out = self.attention_network(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        out = self.feed_forward(out)
        out = self.layer_norm(out)

        logits = self.to_logits(out)

        if return_attn:
            return logits, selfattn_celltypes, cross_attn_celltypes, cross_attn_path
        else:
            return logits