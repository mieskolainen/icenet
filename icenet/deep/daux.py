# MLPs, Transformers, Hutchinson etc
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn
import numpy as np
import tqdm
import math

from torch.utils.data import TensorDataset, random_split, DataLoader

def split_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    frac: float = 0.9,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Split (X, y) into train/val TensorDatasets and optional DataLoaders
    
    Returns:
        train_ds, val_ds, train_loader, val_loader
    """
    # 1) Create full dataset
    full_ds = TensorDataset(X, y)

    # 2) Compute split sizes
    n = len(full_ds)
    n_train = int(frac * n)
    n_val   = n - n_train
    
    # 3) Do the split with a fixed seed for reproducibility
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 4) Wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_wrapper(model, optimizer, scheduler, train_loader, val_loader=None,
                  n_epochs=10, gradient_max=0.5):
    """
    Training wrapper that returns the model state with lowest validation loss.
    """
    best_val_loss = math.inf
    best_state    = None
    
    print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):0.1E}')
    
    train_losses = []
    val_losses   = []

    epoch_bar = tqdm.trange(n_epochs, desc="Training", unit="epoch")
    for epoch in epoch_bar:
        # --------------------
        # Training phase
        # --------------------
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch_x, batch_c in train_loader:
            optimizer.zero_grad()
            loss = model.loss(batch_x, batch_c).mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_max)
            optimizer.step()

            bsize = batch_x.size(0)
            epoch_loss  += loss.item() * bsize
            total_samples += bsize

        epoch_loss /= total_samples
        train_losses.append(epoch_loss)

        # --------------------
        # Validation phase
        # --------------------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for batch_x, batch_c in val_loader:
                    loss = model.loss(batch_x, batch_c).mean()
                    bsize = batch_x.size(0)
                    val_loss += loss.item() * bsize
                    total_val_samples += bsize

            val_loss /= total_val_samples
            val_losses.append(val_loss)

            # check for new best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k:v.cpu() for k,v in model.state_dict().items()}
        else:
            val_loss = None
            val_losses.append(None)

        # --------------------
        # Scheduler and logging
        # --------------------
        scheduler.step(epoch_loss)
        current_lr = scheduler.get_last_lr()[0]

        postfix = {
            "train_loss": f"{epoch_loss:.4f}",
            "lr":         f"{current_lr:.6f}",
        }
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.4f}"
        epoch_bar.set_postfix(postfix)
    
    # restore best model (if any validation was done)
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return (train_losses, val_losses), model


def expand_dim(c, num_samples: int, device='cpu'):
    """
    Expandim dimension of c into a Tensor of shape (n_samples, cond_dim)
    
    If c is a scalar (0‐D), builds a (n_samples, 1) tensor filled with c
    If c is a 1‐D tensor of length cond_dim, broadcasts to (n_samples, cond_dim)
    If c is already 2‐D of shape (n_samples, cond_dim), just returns it
    """
    
    c_t = torch.as_tensor(c, device=device, dtype=torch.float32)
    
    if c_t.ndim == 0:   # scalar -> (num_samples, 1)
        return c_t.expand(num_samples, 1).clone()
    elif c_t.ndim == 1: # vector -> broadcast to every sample
        return c_t.unsqueeze(0).expand(num_samples, -1).clone()
    elif c_t.ndim == 2:
        if c_t.shape[0] != num_samples:
            raise ValueError(f"number of condition vectors ({c_t.shape[0]}) "
                             f"doesn't match n_samples ({num_samples})")
        return c_t
    else:
        raise ValueError(f"c must be scalar, 1‑D or 2‑D tensor, got ndim={c_t.ndim}")

def div_sandwich(v, x, t, cond, e, create_graph=False):
    """
    Helper function for the Hutchinson trace estimator
    """
    out = (v(x, t, cond) * e).sum()
    grad = torch.autograd.grad(out, x, create_graph=create_graph, retain_graph=True)[0]
    return (grad * e).sum(dim=1)

def div_hutchinson(v, x, t, cond=None,
                   n_samples=10,
                   noise_type="rademacher",
                   create_graph=False):
    """
    Hutchinson's estimator for vector field v(x) divergence, using Rademacher or Gaussian noise.
    """
    estimates = []
    # enable_grad once
    with torch.enable_grad():
        for _ in range(n_samples):
            if noise_type == "rademacher":
                e = (torch.randint(0, 2, x.shape, device=x.device, 
                                  dtype=x.dtype) * 2 - 1)
            elif noise_type == "gaussian":
                e = torch.randn_like(x)
            else:
                raise ValueError(f"Unsupported noise_type: {noise_type}")
            estimates.append(div_sandwich(v=v, x=x, t=t, cond=cond, e=e,
                                          create_graph=create_graph))
    return torch.stack(estimates, dim=0).mean(dim=0)


def sinusoidal_embedding(timesteps, dim):
    """
    Create sinusoidal embeddings for normalized timesteps in [0, 1].

    Args:
        timesteps: Tensor of shape (batch_size, 1)
        dim: Embedding dimension (should be even)

    Returns:
        Tensor of shape (batch_size, dim)
    """
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
    )
    angles = timesteps * freqs  # [batch_size, half_dim]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb  # [batch_size, dim]

class GatedMLPBlock(nn.Module):
    """
    Gated MLP
    """
    def __init__(self, in_dim, out_dim, layer_norm=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.linear(x)
        if self.layer_norm:
            x = self.norm(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)

def get_activation(act="relu"):
    
    if   act == 'silu':
        return nn.SiLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise Exception("Unknown activation chosen")

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, layer_norm=False, act='silu', dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = get_activation(act)
        
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        
        # Dropout layer (only active if dropout > 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x):
        x = self.linear(x)
        if self.layer_norm:
            x = self.norm(x)
        x = self.act(x)
        # After activation typical choice
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[128], layer_norm=False, act='silu',
                 dropout=0.0, apply_tanh=False, apply_sigmoid=False):
        super().__init__()
        assert len(hidden_dim) >= 1, "hidden_dim must have at least one element"
        
        self.input_proj = nn.Linear(in_dim, hidden_dim[0])
        self.layer_norm    = layer_norm
        self.dropout       = nn.Dropout(dropout) if dropout > 0.0 else None
        self.apply_tanh    = apply_tanh
        self.apply_sigmoid = apply_sigmoid
        
        if self.layer_norm:
            self.norm_in = nn.LayerNorm(hidden_dim[0])
        
        self.act_fn = get_activation(act) if act != "gated" else nn.Identity()
        
        # Build hidden layers
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            
            if act != 'gated':
                block = MLPBlock(in_dim=hidden_dim[i], out_dim=hidden_dim[i + 1], layer_norm=layer_norm, dropout=dropout)
            else:
                block = GatedMLPBlock(in_dim=hidden_dim[i], out_dim=hidden_dim[i + 1], layer_norm=layer_norm)
            self.blocks.append(block)
        
        self.output_proj = nn.Linear(hidden_dim[-1], out_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        if self.layer_norm:
            x = self.norm_in(x)
        x = self.act_fn(x)
        
        for block in self.blocks:
            x = x + block(x)  # residual connection
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.output_proj(x)
        
        if self.apply_tanh:
            x = torch.tanh(x)
        
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        
        return x

class VectorTransformer(nn.Module):
    """
    A minimal Transformer model that maps input vectors of dimension
    "in_dim" to output vectors of dimension "out_dim"
    
    Treats each component of the input vector as a "token"
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.05,
        act: str = "relu"
    ):
        super().__init__()
        
        # Embed each scalar token into a d_model-dimensional vector
        self.embed = nn.Linear(1, d_model)
        
        # Learnable positional embeddings for sequence length (in_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_dim, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = act
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to output dimension
        self.fc_out = nn.Linear(d_model, out_dim)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x:   Tensor of shape (batch_size, in_dim)
        Returns:
            out: Tensor of shape (batch_size, out_dim)
        """
        # x -> (batch_size, in_dim, 1) so each feature is a "token"
        x = x.unsqueeze(-1)
        
        # Embed tokens
        x = self.embed(x)  # (batch_size, in_dim, d_model)
        # Add positional embeddings
        x = x + self.pos_embed  # (batch_size, in_dim, d_model)
        
        # Transformer expects shape (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        # Pass through Transformer encoder
        x = self.transformer_encoder(x)  # (in_dim, batch_size, d_model)
        
        # Pool over sequence dimension
        x = x.mean(dim=0)  # (batch_size, d_model)
        
        # Final projection
        return self.fc_out(x)  # (batch_size, out_dim)
