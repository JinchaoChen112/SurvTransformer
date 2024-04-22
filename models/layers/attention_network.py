import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class Attn_Net_Gated(nn.Module):
    def __init__(self, indim=256, outdim=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(indim, outdim),
            nn.Tanh()]

        self.attention_b = [nn.Linear(indim, outdim), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(outdim, n_classes)

    def forward(self, x):
        x = x.squeeze()
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A


class Att_Fusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.linear3 = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, A1, A2, x1, x2):
        A1 = torch.transpose(A1, 1, 0)
        h1 = torch.mm(F.softmax(A1, dim=1), x1.squeeze())
        h1 = self.linear1(h1)

        A2 = torch.transpose(A2, 1, 0)
        h2 = torch.mm(F.softmax(A2, dim=1), x2.squeeze())
        h2 = self.linear2(h2)

        fusion = torch.cat((h1, h2), dim=1)
        fusion = self.linear3(fusion)

        return fusion, h1, h2


def Find_closest_factors(num):
    for i in range(int(num ** 0.5), 0, -1):
        if num % i == 0:
            return i, num // i
    return 1, num


class MixedAttentionLayer(nn.Module):
    def __init__(self,
                 dim=256,
                 eps=1e-8,
                 num_celltypes=50,
                 focusing_factor=3,
                 kernel_size=5,
                 heads=1,
                 dropout=0.1

                 ):
        super().__init__()
        self.num_celltypes = num_celltypes
        self.eps = eps
        self.heads = heads
        self.scale = dim ** -0.5
        self.focusing_factor = focusing_factor
        self.scale_linear = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.kernel_function = nn.ReLU()
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                             groups=dim, padding=kernel_size // 2)
        self.gated_att_net = Attn_Net_Gated(dim, dim)
        self.att_fusion = Att_Fusion(dim)
        self.feamap_linear = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None, return_attention=False, ablation_experiment=False):
        b, n, d = x.shape
        h, m, eps = self.heads, self.num_celltypes, self.eps
        qkv = self.to_qkv(x).reshape(b, n, 3, d).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q_celltypes = q[:, :self.num_celltypes, :]
        k_celltypes = k[:, :self.num_celltypes, :]
        v_celltypes = v[:, :self.num_celltypes, :]

        q_path = q[:, self.num_celltypes:, :]
        k_path = k[:, self.num_celltypes:, :]
        v_path = v[:, self.num_celltypes:, :]

        einops_eq = '... i d, ... j d -> ... i j'
        selfattn_celltypes = einsum(einops_eq, q_celltypes * self.scale, k_celltypes)
        cross_attn_path = einsum(einops_eq, q_path * self.scale, k_celltypes)
        cross_attn_celltypes = einsum(einops_eq, q_celltypes * self.scale, k_path)

        selfattn_celltypes_a = selfattn_celltypes.softmax(dim=-1)
        cross_attn_path_a = cross_attn_path.softmax(dim=-1)
        cross_attn_celltypes_a = cross_attn_celltypes.softmax(dim=-1)

        out_self_celltypes = selfattn_celltypes_a @ v_celltypes
        out_cross_path = cross_attn_path_a @ v_celltypes
        out_cross_celltypes = cross_attn_celltypes_a @ v_path

        q_path_l = self.kernel_function(q_path) + 1e-6
        k_path_l = self.kernel_function(k_path) + 1e-6
        q_path_l = q_path_l / nn.Softplus()(self.scale_linear)
        k_path_l = k_path_l / nn.Softplus()(self.scale_linear)

        q_path_l_norm = q_path_l.norm(dim=-1, keepdim=True)
        k_path_l_norm = k_path_l.norm(dim=-1, keepdim=True)

        q_path_l = q_path_l ** self.focusing_factor
        k_path_l = k_path_l ** self.focusing_factor

        q_path_l = (q_path_l / q_path_l.norm(dim=-1, keepdim=True)) * q_path_l_norm
        k_path_l = (k_path_l / k_path_l.norm(dim=-1, keepdim=True)) * k_path_l_norm

        i, j, c, d = q_path_l.shape[-2], k_path_l.shape[-2], k_path_l.shape[-1], v_path.shape[-1]
        z = 1 / (torch.einsum("b i c, b c -> b i", q_path_l, k_path_l.sum(dim=1)) + 1e-6)

        kv = torch.einsum("b j c, b j d -> b c d", k_path_l, v_path)
        out_s = torch.einsum("b i c, b c d, b i -> b i d", q_path_l, kv, z)

        w_c, h_c = Find_closest_factors(int(v_path.shape[1]))
        feature_map = rearrange(v_path, "b (w h) c -> b c w h", w=w_c, h=h_c)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        out_self_path = out_s + feature_map
        out_self_path = self.feamap_linear(out_self_path)

        A_cross_celltypes = self.gated_att_net(out_cross_celltypes)
        A_cross_path = self.gated_att_net(out_cross_path)

        A_self_celltypes = self.gated_att_net(out_self_celltypes)
        A_self_path = self.gated_att_net(out_self_path)

        fusion_cross, out_cross_celltypes, out_cross_path = self.att_fusion(A_cross_celltypes, A_cross_path,
                                                                            out_cross_celltypes, out_cross_path)
        fusion_self, out_self_celltypes, out_self_path = self.att_fusion(A_self_celltypes, A_self_path,
                                                                         out_self_celltypes, out_self_path)

        out = torch.cat((fusion_cross, fusion_self), dim=1)

        if return_attention:
            return out, selfattn_celltypes.squeeze().detach().cpu(), cross_attn_celltypes.squeeze().detach().cpu(), cross_attn_path.squeeze().detach().cpu()
        else:
            return out

