from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    print(f"freqs_cis.shape: {freqs_cis.shape}, x.shape: {x.shape}")
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    # query (batch_size, seqlen, n_local_heads, head_dim) -> (batch_size, seqlen, n_local_heads, head_dim, 2)
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)) # theta_i (head_dim/2)
    m = torch.arange(0, seqlen, device=device).float() # m (seqlen)
    freqs = torch.outer(m, freqs) # m * theta_i (seqlen, head_dim/2)
    cos_freqs, sin_freqs = torch.cos(freqs), torch.sin(freqs) # (seqlen, head_dim/2)
    cos_freqs, sin_freqs = reshape_for_broadcast(cos_freqs, query_real), reshape_for_broadcast(sin_freqs, query_real) # (bs, seqlen, n_local_heads, head_dim/2)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_real_rot = cos_freqs * query_real - sin_freqs * query_imag # (bs, seqlen, n_local_heads, head_dim/2)
    query_imag_rot = sin_freqs * query_real + cos_freqs * query_imag # (bs, seqlen, n_local_heads, head_dim/2)
    key_real_rot = cos_freqs * key_real - sin_freqs * key_imag # (bs, seqlen, n_local_kv_heads, head_dim/2)
    key_imag_rot = sin_freqs * key_real + cos_freqs * key_imag # (bs, seqlen, n_local_kv_heads, head_dim/2)

    # concatenate the real and imaginary parts of the rotated query and key tensors
    query_out = torch.stack((query_real_rot, query_imag_rot), dim=-1).reshape(query.shape) # reshape performs interleaving by default
    key_out = torch.stack((key_real_rot, key_imag_rot), dim=-1).reshape(key.shape)

    return query_out, key_out