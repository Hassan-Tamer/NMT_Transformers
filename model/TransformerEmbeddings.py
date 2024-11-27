import torch
import torch.nn as nn

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_sequence_length: int):
        """Defines the embedding layer with learnt positional embeddings.

        This layer defines both the token embeddings and positional embeddings,
        which are added together to form the final embedding.

        Args:
            vocab_size: The size of the vocabulary,
                    used to define the size of the token embedding table.
            hidden_size: The dimensionality of the embedding space for both token embeddings and positional embeddings.
            max_sequence_length: The maximum sequence length of the input sequences,
                    used to define the size of the position embedding table.

        Note that this implementation does not use dropout on the embeddings
        and uses learnt positional embeddings instead of sinusoidal embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_size)

        # TODO Initialize the module and its parameters here.
        # You should use `nn.Embedding` for both token embeddings and positional embeddings

    def compute_logits(self, decoder_output: torch.FloatTensor) -> torch.FloatTensor:
        """Computes the logits for the next token prediction given the decoder output.

        Args:
            decoder_output: Tensor of shape (batch_size, sequence_length, hidden_size) - the output of the decoder.

        Returns:
            Tensor of shape (batch_size, sequence_length, vocab_size) containing the logits for the next token prediction.
        """

        # TODO Implement this function
        # Hint: you can access the weight parameter matrix via .weight of an nn.Embedding module:
        # Example:
        # ```embeddings = nn.Embedding(num_embeddings, embedding_dim)
        # assert list(embeddings.weight.shape) == [num_embeddings, embedding_dim]```
        # torch.matmul or F.linear may also be useful here.

        logits = torch.matmul(decoder_output, self.token_embeddings.weight.T)
        return logits

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Computes the embeddings for the input tokens.

        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) containing the input token ids.

        Returns:
            Tensor of shape (batch_size, sequence_length, hidden_size) containing
                    the sum of token embeddings and position embeddings for the input tokens.
        """

        # TODO Implement the forward pass of the embedding layer.

        embedding = self.token_embeddings(input_ids)
        pos = self.position_embeddings(torch.arange(input_ids.shape[1]))
        embedding = embedding + pos
        # print(embedding.shape)
        return embedding
    

if __name__ == '__main__':
    vocab_size = 10
    hidden_size = 5
    max_sequence_length = 7
    batch_size = 3
    trans = TransformerEmbeddings(vocab_size=vocab_size, hidden_size=hidden_size, max_sequence_length=max_sequence_length)
    input_ids = torch.randint(0, vocab_size, (batch_size, max_sequence_length))
    out = trans(input_ids)
    assert out.shape == (batch_size, max_sequence_length, hidden_size)

    decoder_output = torch.randn(batch_size, max_sequence_length, hidden_size)
    logits = trans.compute_logits(decoder_output)
    assert logits.shape == (batch_size, max_sequence_length, vocab_size)

