import torch
import math
from einops import einsum
import torch.nn as nn

class Embedding(nn.Module):
    """Implement the Embedding class that inherits from torch.nn.Module and performs an
    embedding lookup.
    """
    def __init__(self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device|None = None,
        dtype: torch.dtype|None = None, 
    ) -> None:
        """
        Args:
            num_embeddings (int): size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            evice (torch.device | None): Device to store the parameters on
            detype (torch.device | None): Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Init the parameters, N ~ (\mu=0, \sigma^2 = 1.) truncated at [-3\sigma, 3\sigma]
        """
        # sigma = math.sqrt(2 / (self.in_features + self.out_features))
        # nn.init.trunc_normal_(self.weight, std=sigma**2, a = -3 * sigma, b = 3 * sigma)
        sigma = 1.
        nn.init.trunc_normal_(self.embeddings, std=sigma**2, a = -3 * sigma, b = 3*sigma)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

class Linear(nn.Module):
    """Implementing the linear module followed the interface of PyTorch's built-in nn.Linear.Module,
    except for not having a bias argument or parameter.
    """
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        device: torch.device|None = None,
        dtype: torch.dtype|None = None
    ) -> None:
        """
        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None): Device to store the parameters on
            detype (torch.device | None): Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Init the parameters, N ~ (\mu=0, \sigma^2 = \frac{2}{in_feature + out_feature}) truncated at [-3\sigma, 3\sigma]
        """
        sigma = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, std=sigma**2, a = -3 * sigma, b = 3 * sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
