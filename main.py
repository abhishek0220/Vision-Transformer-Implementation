import torch
import torch.nn as nn
from typing import Callable


class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        """
        Parameters
        ----------
        img_size : int
            Size of the input image
        patch_size : int
            Size of the patches
        in_channels : int
            Number of channels in the input image
        embed_dim : int
            Dimension of the embedding
        """
        if img_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size")
        
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(  # Convolutional layer to split into patches and embed them
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, img_size, img_size)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, embed_dim, n_patches)
        """
        x = self.proj(x)            # shape: (batch_size, embed_dim, img_size // patch_size, img_size // patch_size)
        x = x.flatten(2)            # shape: (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)       # shape: (batch_size, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p=0.0, proj_p=0.0):
        """
        Parameters
        ----------
        dim : int
            Dimension of features per token
        n_heads : int
            Number of attention heads
        qkv_bias : bool
            Whether to add bias to the query, key, value
        attn_p : float
            Dropout probability for the attention
        proj_p : float
            Dropout probability for the projection
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1/ (self.head_dim ** 0.5)
        self.qkv: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop: Callable[[torch.Tensor], torch.Tensor]  = nn.Dropout(attn_p)
        self.proj: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(dim, dim)
        self.proj_drop: Callable[[torch.Tensor], torch.Tensor] = nn.Dropout(proj_p)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_patches+1, dim)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_patches+1, dim)
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        # Linear Tranformation
        qkv= self.qkv(x)         # shape: (batch_size, n_patches+1, 3*dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )                                       # shape: (batch_size, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # shape: (3, batch_size, n_heads, n_patches+1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]        # shape: (batch_size, n_heads, n_patches+1, head_dim)
        k_t = k.transpose(-2, -1)               # shape: (batch_size, n_heads, head_dim, n_patches+1)
        dp = q.matmul(k_t) * self.scale         # shape: (batch_size, n_heads, head_dim, n_patches+1)
        attn = dp.softmax(dim=-1)               # shape: (batch_size, n_heads, head_dim, n_patches+1)
        attn = self.attn_drop(attn)             # shape: (batch_size, n_heads, head_dim, n_patches+1)

        weighted_avg = attn.matmul(v)                # shape: (batch_size, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # shape: (batch_size, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)       # shape: (batch_size, n_patches+1, dim)

        x = self.proj(weighted_avg)             # shape: (batch_size, n_patches+1, dim)
        x = self.proj_drop(x)                   # shape: (batch_size, n_patches+1, dim)

        return x



    