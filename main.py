import torch
import torch.nn as nn


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
            in_channels=in_channels,
            out_channels=embed_dim,
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

