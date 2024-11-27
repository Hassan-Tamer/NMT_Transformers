import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 is_causal_attention: bool = False,
                 is_cross_attention: bool = False):
        """Defines a flexible multi-head attention layer.

        This layer should define parameters for the query, key and value projections, as well as the output projection,
        and implement the following steps:
        (1) Project the input vectors using query projection and key projection matrices.
        (2) Compute the head-wise attention scores scaled by 1/sqrt(head_dim)
        (3) Perform appropriate masking to the attention scores using key_padding_mask and optionally causal attention.
        (4) Normalize the head-wise attention scores using softmax.
        (5) Compute the value projections and then aggregate using the normalized attention scores.
        (6) Use the output projection to obtain the final output vectors.
        When is_cross_attention is True, the key and value projections are computed from the encoder outputs.
        Note that we do not use attention weight dropout in this implementation.

        Args:
            hidden_size: The dimensionality of the input vectors.
            num_attention_heads: The number of attention heads.
            is_causal_attention: Whether to use causal masking,
                    where tokens cannot attend to the future tokens on their right.
            is_cross_attention: Whether to use cross attention,
                    where we use different inputs for the key/value vs. query vectors.
        """
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "The hidden size must be divisible by the number of attention heads."
        self.dk = hidden_size // num_attention_heads  # embedding dimension of query and key vectors per head
        self.num_attention_heads = num_attention_heads
        self.is_cross_attention = is_cross_attention
        self.is_causal_attention = is_causal_attention
        self.d_model = hidden_size

        # hidden size = d_model

        # self.QKV_projection = nn.Linear(hidden_size, 3 * hidden_size)
        self.Q_projection = nn.Linear(self.d_model, self.d_model)
        self.K_projection = nn.Linear(self.d_model, self.d_model)
        self.V_projection = nn.Linear(hidden_size, self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model)



        # TODO Initialize the module and its parameters here.
        # This module should be able to handle both full self-attention, causal masked self-attention and cross-attention.
        # IMPORTANT: You are not allowed to use `nn.MultiheadAttention` or `nn.functional.scaled_dot_product_attention`!

        # raise NotImplementedError("The __init__ function in TransformerAttention is not implemented yet.")

    def causal_attention_mask(self,
                              sequence_length: int,
                              device: Optional[torch.device] = None) -> torch.FloatTensor:
        """Return a Float tensor that can be added to the (un-normalized) attention scores for causal masking.

        Args:
            sequence_length: width and height of the attention mask tensor.
            device: which torch device the resulting tensor should be on (important if you use GPU).

        Returns:
            A Float tensor of shape (1, 1, sequence_length, sequence_length) on device `device`,
            where the entries above the diagonal contain large negative values,
            which means that a query at position i can't attend to a key at position j>i.
        """

        # TODO Implement the forward function.
        # IMPORTANT: For full credit, you should not use python loops.
        #
        # Hint 1: You can pick an arbitrary large value (e.g., -10^{6}), but note that
        #         using `float("-inf")` might lead to numerical issues and 'nan' values during training.
        #
        # Hint 2: Useful pytorch functions for this are `torch.arange` or `torch.triu`.
        #
        # Hint 3: You can move the tensor you create to a device by calling `tensor.to(device)`
        #
        # You should use this function in `forward` and use the returned tensor to implement causal masking
        # by adding it to the un-normalized attention scores of shape (batch_size, num_heads, sequence_length, sequence_length),
        # as torch will handle broadcasting and expand the first two dimensions to batch size and num_heads.
        #
        # You will the masking tensor to be on the same device as the attention scores's device,
        # which you can via the attribute `tensor.device`.
        x = torch.ones(sequence_length,sequence_length)
        x = torch.triu(x, diagonal=1)
        y = torch.ones(sequence_length,sequence_length)
        y.fill_(-1e9)
        mask = x*y
        mask = mask.to(device)
        return mask


        # raise NotImplementedError("The forward function in TransformerAttention is not implemented yet.")


    def forward(self,
                hidden_states: torch.FloatTensor,
                key_padding_mask: torch.BoolTensor,
                encoder_outputs: Optional[torch.FloatTensor] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Computes scaled dot-product attention and returns the output of the attention layer.

        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_size) - the input vectors to the layer.
            key_padding_mask: Tensor of shape (batch_size, sequence_length) indicating which tokens are padding tokens.
                    A `True` entry means that this token should be ignored for the purpose of attention.
                    In the case of cross-attention, the tensor has shape (batch_size, encoder_sequence_length).
            encoder_outputs: Optional tensor of shape (batch_size, encoder_sequence_length, hidden_size).
                    The output vectors of the encoder and only passed if the layer performs cross-attention.

        Returns:
            A (layer_output, attention_weights) where layer_output is a tensor of shape (batch_size, sequence_length, hidden_size)
            and attention_weights are the normalized attention scores in the form of
            a tensor of shape (batch_size, num_attention_heads, number_of_query_tokens, number_of_key_tokens).
        """

        # TODO Implement the forward function!
        # Hint 1: Use `torch.reshape` to add a new axis for the attention head,
        #         which will allow you to process all attention heads in parallel.
        #
        # Hint 2: You can use `torch.transpose` to swap the order of two axes,
        #         As the attention head dimension should be next to the batch size,
        #         see the shape of the output attention weights.
        #
        # Hint 3: `torch.bmm(matrix1, matrix2)` is useful for computing batched matrix multiplications
        #         If matrix1 has shape (B, M, N) and matrix2 has shape (B, N, P),
        #         it performs `B` matrix multiplications and outputs a tensor of shape (B, M, P).
        #         Alternatively, `torch.einsum` should be very useful.
        #         (We really encourage you to check out the documentation of `torch.einsum`,
        #         it can really make your life easier here.)
        # raise NotImplementedError("The forward function in TransformerAttention is not implemented yet.")


        batch_size, sequence_length, hidden_size = hidden_states.size()
        self.Q = self.Q_projection(hidden_states)

        if self.is_cross_attention:
            self.K = self.K_projection(encoder_outputs)
            self.V = self.V_projection(encoder_outputs)


        else:
            self.K = self.K_projection(hidden_states)
            self.V = self.V_projection(hidden_states)

        key_seq_length = encoder_outputs.size(1) if self.is_cross_attention else sequence_length

        self.Q = self.Q.view(batch_size, sequence_length, self.num_attention_heads, self.dk).transpose(1, 2)
        self.K = self.K.view(batch_size, key_seq_length, self.num_attention_heads, self.dk).transpose(1, 2)
        self.V = self.V.view(batch_size, key_seq_length, self.num_attention_heads, self.dk).transpose(1, 2)

        scores = torch.einsum('bhqd,bhkd->bhqk', self.Q,self.K)
        scores = scores / math.sqrt(self.dk)


        if self.is_causal_attention:
            causal_mask = self.causal_attention_mask(sequence_length, device=hidden_states.device)
            scores = scores + causal_mask

        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_padding_mask, -1e9)

        attention_weights = F.softmax(scores,dim=-1)

        x = torch.matmul(attention_weights, self.V)  # Compute weighted sum of value vectors
        x = x.transpose(1, 2).contiguous().view(x.shape[0], sequence_length, self.d_model)  # Reshape and combine heads
        output = self.output_projection(x)

        return output, attention_weights


def plot_attention_matrix(attention_matrix, title):
    """Creates a new figure and plots the normalized attention weights as a heatmap.

    This should provide a colorbar for the scale of the heatmap and label the axes "query token position" and "key token position".
    Args:
        attention_matrix: A numpy array of shape (number_of_query_tokens, number_of_key_tokens)
        title: The title of the plot.
    """
    sns.heatmap(attention_matrix, annot=True, fmt=".2f", cbar=True)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    embed_dim = 8
    num_heads = 2
    batch_size = 1
    encoder_seq_length = 5
    decoder_seq_length = 7

    encoder_outputs = torch.randn(batch_size, encoder_seq_length, embed_dim)
    decoder_inputs = torch.randn(batch_size, decoder_seq_length, embed_dim)

    encoder_padding_mask = torch.zeros(batch_size, encoder_seq_length, dtype=torch.bool)
    encoder_padding_mask[:, -1] = True # The last encoder token is a padding tokens

    decoder_padding_mask = torch.zeros(batch_size, decoder_seq_length, dtype=torch.bool)
    decoder_padding_mask[:, -2:] = True # The last two decoder tokens are padding token


    cross_attention = MultiHeadAttention(embed_dim, num_heads, is_cross_attention=True)
    causal_attention = MultiHeadAttention(embed_dim, num_heads, is_causal_attention=True)

    cross_attention_out, cross_attention_weights = cross_attention(decoder_inputs, encoder_padding_mask, encoder_outputs)
    causal_attention_out, causal_attention_weights = causal_attention(decoder_inputs, decoder_padding_mask)

    # Make sure your outputs have the right hapes
    assert cross_attention_out.shape == (batch_size, decoder_seq_length, embed_dim)
    assert cross_attention_weights.shape == (batch_size, num_heads, decoder_seq_length, encoder_seq_length)
    assert causal_attention_out.shape == (batch_size, decoder_seq_length, embed_dim)
    assert causal_attention_weights.shape == (batch_size, num_heads, decoder_seq_length, decoder_seq_length)

    # Check that the attention weights are normalized
    assert torch.isclose(cross_attention_weights.sum(dim=-1), torch.tensor(1.0)).all()
    assert torch.isclose(causal_attention_weights.sum(dim=-1), torch.tensor(1.0)).all()

    # Check if the attention masking works
    assert torch.isclose(cross_attention_weights[:,:,:,-1], torch.tensor(0.0)).all()
    assert torch.isclose(causal_attention_weights[:,:,:,-2:], torch.tensor(0.0)).all()
    assert torch.isclose(causal_attention_weights[:,:,2,3:], torch.tensor(0.0)).all()

    # plot_attention_matrix(cross_attention_weights[0,0].detach().numpy(), "cross-attention, head 1")
    # plot_attention_matrix(cross_attention_weights[0,1].detach().numpy(), "cross-attention, head 2")
    # plot_attention_matrix(causal_attention_weights[0,0].detach().numpy(), "causal self-attention, head 1")
    # plot_attention_matrix(causal_attention_weights[0,1].detach().numpy(), "causal self-attention, head 2")

