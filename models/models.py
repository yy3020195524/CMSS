import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from functools import partial
from segment_anything.modeling import ImageEncoderViT3D, MaskDecoder3D
import torchio as tio

class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT3D(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 16,
            in_chans: int = 1,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            # nn.LayerNorm(out_chans),
        )
        self.neck_64 = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                128,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(128),
            # nn.MaxPool3d(2),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.Conv3d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(64),
            # nn.LayerNorm(out_chans),
        )
        self.neck_128 = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                128,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(128),
            # nn.MaxPool3d(2),
            # nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.Conv3d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(128),
            # nn.LayerNorm(out_chans),
        )

        self.neck_256 = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                256,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(256),
            nn.MaxPool3d(2),
            # nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
            nn.Conv3d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(256),
            # nn.LayerNorm(out_chans),
        )
        self.neck_512 = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                512,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(512),
            nn.MaxPool3d(4),
            nn.Conv3d(
                512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(512),
            # nn.LayerNorm(out_chans),
        )

    # def load_params(self, model_dict):
    #     encoder_store_dict = self.image_encoder.state_dict()
    #     for key in model_dict.keys():
    #         if "image_encoder.block" in key:
    #             # encoder_store_dict[key.replace("module.backbone.", "")] = model_dict[key]
    #             encoder_store_dict[key] = model_dict[key]
    #         elif "image_encoder.patch_embed" in key:
    #             encoder_store_dict[key] = model_dict[key]
    #         else:
    #             print("")
    #     self.image_encoder.load_state_dict(encoder_store_dict, strict=False)
    #
    #     print('Use encoder pretrained weights')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_size = [1,1,256,256,256]
        # import IPython; IPython.embed()
        x = self.patch_embed(x)
        # x = [1,16,16,16,768]
        # import pdb; pdb.set_trace()
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x_64 = self.neck_64(x.permute(0, 4, 1, 2, 3))
        x_128 = self.neck_128(x.permute(0, 4, 1, 2, 3))
        x_256 = self.neck_256(x.permute(0, 4, 1, 2, 3))
        x_512 = self.neck_512(x.permute(0, 4, 1, 2, 3))
        # x = self.neck(x.permute(0, 4, 1, 2, 3))
        # output_size = [1,256,16,16,16]
        return x_512, [x_64,x_128,x_256,x_512]


class Block3D(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            D, H, W = x.shape[1], x.shape[2], x.shape[3]
            x, pad_dhw = window_partition3D(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition3D(x, self.window_size, pad_dhw, (D, H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (D, H, W), (D, H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        x = self.proj(x)

        return x


def window_partition3D(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)


def window_unpartition3D(
        windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_d: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int, int],
        k_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size

    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)

    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)

    attn = (
            attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w) + rel_d[:, :, :, :, None, None] + rel_h[:, :, :, None, :,
                                                                                         None] + rel_w[:, :, :, None,
                                                                                                 None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16, 16),
            stride: Tuple[int, int] = (16, 16, 16),
            padding: Tuple[int, int] = (0, 0, 0),
            in_chans: int = 1,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1)
        return x

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)

    def forward(self, x):
        return F.softmax(self.final_conv(x), dim=1)  # 使用 softmax


class MIA_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(MIA_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X z*y*x)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, depth, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma * out + x
        return out

class Encoder(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, act='relu'):
        super(Encoder, self).__init__()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        return self.out512,(self.skip_out64,self.skip_out128,self.skip_out256,self.skip_out512)

class Decoder(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=16, act='relu'):
        super(Decoder, self).__init__()
        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        self.out_tr = OutputTransition(64, n_class)

    def forward(self, x, skips):
        self.out_up_256 = self.up_tr256(x, skips[2])
        self.out_up_128 = self.up_tr128(self.out_up_256, skips[1])
        self.out_up_64 = self.up_tr64(self.out_up_128, skips[0])
        self.out = self.out_tr(self.out_up_64)

        return self.out
class FusionLayer(nn.Module):
    def __init__(self, in_channel, outChans, depth,act):

        super(FusionLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = LUConv(1024, 512,act)
        self.layer2 = LUConv(512, 512,act)
    def forward(self, x1,x2):
        concat = torch.cat((x1,x2),1)
        cov_layer1 = self.layer1(concat)
        cov_layer2 = self.layer2(cov_layer1)
        out = self.sigmoid(cov_layer2)
        return out
class multimodal_segmentation(nn.Module):
    def __init__(self, n_class=4) -> None:
        super(multimodal_segmentation, self).__init__()
        # self.conv3d = nn.Conv3d(2,1,kernel_size=1, stride=1,padding=0)
        # self.encoder_depth = 12
        self.encoder_depth = 12
        self.encoder_embed_dim = 384
        self.image_size = 16
        self.encoder_num_heads=12
        # self.encoder_num_heads=6
        self.vit_patch_size = 2
        # self.vit_patch_size = 16
        self.image_embedding_size = self.image_size // self.vit_patch_size
        self.prompt_embed_dim = 384
        self.encoder_global_attn_indexes=[2, 5, 8, 11]
        self.encoder = Encoder()
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        self.image_encoder = ImageEncoderViT3D(
            depth=self.encoder_depth,
            embed_dim=self.encoder_embed_dim,
            img_size=self.image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.encoder_num_heads,
            patch_size=self.vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
        )
        self.decoder = Decoder()
        self.mask_decoder = MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.MIA_module = MIA_Module(16)
        # self.fusion_layer =fusionLayer(512,512, 1,act='relu')
        self.fusion_layer =FusionLayer(512,512, 1,act='relu')
        self.conv3d_convert = nn.Sequential(
            # nn.GroupNorm(16, 768),
            nn.GroupNorm(16, 1024),
            nn.ReLU(inplace=True),
            # nn.Conv3d(768,512, kernel_size=1, stride=1,padding=0)
            nn.Conv3d(1024,512, kernel_size=1, stride=1,padding=0)
        )

    def load_params(self, model_dict):
        encoder_store_dict = self.image_encoder.state_dict()
        decoder_store_dict = self.mask_decoder.state_dict()
        for key in model_dict.keys():
            if "image_encoder.block" in key:
                # encoder_store_dict[key.replace("module.backbone.", "")] = model_dict[key]
                encoder_store_dict[key] = model_dict[key]
            elif "image_encoder.patch_embed" in key:
                encoder_store_dict[key] = model_dict[key]
            else:
                print("zxg:")

            if "mask_decoder" in key:
                # decoder_store_dict[key.replace("module.backbone.", "")] = model_dict[key]
                decoder_store_dict[key] = model_dict[key]
        self.image_encoder.load_state_dict(encoder_store_dict, strict=False)
        self.mask_decoder.load_state_dict(decoder_store_dict, strict=False)

        print('Use encoder pretrained weights')
    def forward(self, CT_img,MRI_img):
        # CT_img_F_ds, CT_Skips = self.image_encoder(CT_img)
        # CT_img_F_ds, CT_Skips = self.encoder(CT_img)
        CT_img = self.norm_transform(CT_img.squeeze(dim=1))  # (N, C, W, H, D)
        CT_img = CT_img.unsqueeze(dim=1)

        CT_img_F_ds,CT_Skips = self.image_encoder(CT_img)
        # MRI_img_F_ds, MRI_Skips = self.image_encoder(MRI_img)
        MRI_img_F_ds, MRI_Skips= self.image_encoder(MRI_img)

        CT_img_F_mia = self.MIA_module(CT_img_F_ds)
        MRI_img_F_mia = self.MIA_module(MRI_img_F_ds)

        out_fuse = self.fusion_layer(CT_img_F_mia, MRI_img_F_mia)
        CT_F_z = torch.cat([out_fuse, CT_img_F_mia],dim=1)
        MRI_F_z = torch.cat([out_fuse, MRI_img_F_mia],dim=1)
        CT_F_z = self.conv3d_convert(CT_F_z)
        MRI_F_z = self.conv3d_convert(MRI_F_z)

        # CT_F_z [1, 512,4,4,4]
        CT_seg_out = self.decoder(CT_F_z, CT_Skips)
        # CT_seg_out = self.mask_decoder(CT_F_z, CT_Skips)
        # MRI_seg_out = self.mask_decoder(MRI_F_z, MRI_Skips)
        MRI_seg_out = self.decoder(MRI_F_z, MRI_Skips)

        return CT_img_F_ds, MRI_img_F_ds, CT_seg_out, MRI_seg_out

class Enhancement_texture_LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Enhancement_texture_LDC, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        self.center_mask = torch.tensor([[[[[0, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 0]]]]], dtype=torch.float32).cuda()

        self.base_mask = nn.Parameter(torch.ones_like(self.conv.weight), requires_grad=False)

        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1), 1, 1, 1]),
                                           requires_grad=True)

        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, :, None, None]

        out_diff = F.conv3d(
            x = x.as_tensor(),
            weight=self.conv.weight * mask,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

        return out_diff

class Differential_enhance(nn.Module):
    def __init__(self, nf=48):
        super(Differential_enhance, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.global_maxpool = nn.AdaptiveMaxPool3d(1)
        self.conv1x1 = nn.Conv3d(nf, nf // 2, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(nf, nf // 4, bias=False)
        self.fc2 = nn.Linear(nf // 4, nf, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fuse, x1, x2):
        b, c, d, h, w = x1.shape
        sub_1_2 = x1 - x2
        sub_2_1 = x2 - x1
        avg_w_1_2 = self.global_avgpool(sub_1_2).view(b, c)
        max_w_1_2 = self.global_maxpool(sub_1_2).view(b, c)
        avg_w_2_1 = self.global_avgpool(sub_2_1).view(b, c)
        max_w_2_1 = self.global_maxpool(sub_2_1).view(b, c)
        attn_w_1_2 = self.fc2(F.relu(self.fc1(avg_w_1_2 + max_w_1_2)))
        attn_w_2_1 = self.fc2(F.relu(self.fc1(avg_w_2_1 + max_w_2_1)))
        w_1_2 = self.sigmoid(attn_w_1_2).view(b, c, 1, 1, 1)
        w_2_1 = self.sigmoid(attn_w_2_1).view(b, c, 1, 1, 1)
        D_F1 = torch.multiply(w_1_2, fuse)
        D_F2 = torch.multiply(w_2_1, fuse)
        F_1 = D_F1 + x1
        F_2 = D_F2 + x2
        return F_1, F_2
class Cross_layer(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0
    ):
        super().__init__()
        self.d_model = hidden_dim
        self.texture_enhance1 = Enhancement_texture_LDC(self.d_model,self.d_model)
        self.texture_enhance2 = Enhancement_texture_LDC(self.d_model, self.d_model)
        self.Diff_enhance = Differential_enhance(self.d_model)

    def forward(self, Fuse, x1,x2):
        TX_x1 = self.texture_enhance1(x1)
        TX_x2 = self.texture_enhance2(x2)
        DF_x1, DF_x2 = self.Diff_enhance(Fuse, x1,x2)
        F_1 = TX_x1 +DF_x1
        F_2 = TX_x2 +DF_x2
        return F_1, F_2