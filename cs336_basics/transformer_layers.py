import torch
import math
from einops import einsum
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """RMSNorm layer which is compute-efficient simplified variant of LayerNorm."""
    def __init__(self, 
        d_model: int,
        eps: float = 1e-5,
        device: torch.device|None = None,
        dtype: torch.dtype|None = None,
    ) -> None:
        """
        Args:
            d_mode (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability
            device (torch.device | None): Device to store the parameters on
            detype (torch.device | None): Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.scale = d_model ** 0.5
        self.g = nn.Parameter(torch.ones(d_model))
    
    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, squence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMSNorm(a_i) = \frac{a_i}{RMS(a)} * g_i
        # RMS(a) = \sqrt{\frac{1}{d_model} \sum_{i=1}^{d_model} a_i^2 + \epsilon}
        # equals to: RMSNorm(a) = F.normalize(a, dim=-1) * g * \sqrt{d_model}
        result = F.normalize(x, dim=-1) * self.g * self.scale # F.normalize默认dim=1，序列维
        return result.to(in_dtype)

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
            device (torch.device | None): Device to store the parameters on
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

class SwiGLU(nn.Module):
    """ Implement the SwiGLU feed-forward network, 
    composed of a SiLU activation function and a GLU.
    SiLU(x) = x \sigma{x} = \frac{x}{1+e^{-x}}
    GLU(x, W_1, W_2) = \sigma{W_1 x} \odot (W_2 x)
    FFN(x) = SwiGLU(x, W_1, W_2, W_3) = W_2(SiLU(W_1 x) \odot (W_3 x)), where x \in R^{d_model}, W_1, W_3 \in R^{d_ff x d_model}, W_2 \in R^{d_model x d_ff}, d_ff = \frac{8}{3}d_model nearby 64
    """
    def __init__(self, 
        d_model: int,
        d_ff: int,
        device: torch.device|None = None,
        dtype: torch.dtype|None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.w1 = Linear(d_ff, d_model)
        self.w2 = Linear(d_model, d_ff)
        self.w3 = Linear(d_ff, d_model)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_output = self.w1(x)
        silu_output = torch.sigmoid(w1_output) * w1_output
        return self.w2(silu_output * self.w3(x))