import torch
import torch.nn as nn
import torch.nn.functional as F


def prune_sparsegpt_layer(layer, X: torch.Tensor, sparsity: float, damp: float = 0.01):
    """SparseGPT pruning for both Linear and Conv2d layers.
    
    For Conv2d: unfolds input using im2col and reshapes weight to 2D.
    """
    W = layer.weight.data.clone()
    is_conv = isinstance(layer, nn.Conv2d)
    
    if is_conv:
        # Conv2d: reshape weight [O, I, kH, kW] -> [O, I*kH*kW]
        d_out = W.shape[0]
        W_2d = W.view(d_out, -1)
        d_in = W_2d.shape[1]
        
        # im2col: unfold input patches to match weight columns
        # X shape: [B, C_in, H, W]
        X_unf = F.unfold(X, kernel_size=layer.kernel_size,
                         padding=layer.padding, stride=layer.stride)
        # X_unf shape: [B, C_in*kH*kW, L] -> [B*L, C_in*kH*kW]
        X_2d = X_unf.permute(0, 2, 1).reshape(-1, d_in)
    else:
        # Linear
        d_out, d_in = W.shape
        W_2d = W
        X_2d = X.view(-1, d_in) if X.dim() > 2 else X

    # Hessian
    H = (X_2d.t() @ X_2d) / X_2d.shape[0]
    H[torch.arange(d_in), torch.arange(d_in)] += damp * torch.diag(H).mean()

    H_inv = torch.linalg.inv(H)
    C = torch.linalg.cholesky(H_inv, upper=True)

    pruned = torch.zeros_like(W_2d, dtype=torch.bool)

    for j in range(d_in):
        w_j = W_2d[:, j].clone()
        c_jj = C[j, j]

        scores = (w_j ** 2) / (c_jj ** 2 + 1e-12)
        thresh = torch.quantile(scores, sparsity)
        col_mask = scores < thresh

        W_2d[col_mask, j] = 0.0
        pruned[col_mask, j] = True

        err = (w_j - W_2d[:, j]) / (c_jj + 1e-12)
        W_2d[:, j:] -= torch.ger(err, C[j, j:])
        W_2d[pruned] = 0.0

    if is_conv:
        layer.weight.data.copy_(W_2d.view_as(layer.weight))
    else:
        layer.weight.data.copy_(W_2d)