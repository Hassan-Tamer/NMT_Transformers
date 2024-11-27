import sys
sys.path.append('model')
import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
from TransformerEmbeddings import TransformerEmbeddings

class EncoderDecoderModel(nn.Module):
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 hidden_size: int,
                 intermediate_size: int,
                 num_attention_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 max_sequence_length: int,
                 hidden_dropout_prob: float):
        """A encoder-decoder transformer model which can be used for NMT.

        Args:
            source_vocab_size: The size of the source vocabulary.
            target_vocab_size: The size of the target vocabulary.
            hidden_size: The dimensionality of all input and output embeddings.
            intermediate_size: The intermediate size in the feedforward layers.
            num_attention_heads: The number of attention heads in each multi-head attention modules.
            num_encoder_layers: The number of transformer blocks in the encoder.
            num_decoder_layers: The number of transformer blocks in the decoder.
            max_sequence_length: The maximum sequence length that this model can handle.
            hidden_dropout_prob: The dropout probability in the hidden state in each block.
        """

        super().__init__()
        # TODO Register the input embedding modules and the encoder and decoder blocks.
        # You should use the TransformerBlock and TransformerEmbeddings sub-modules.
        #
        # Hint: Check out `nn.ModuleList` to register a variable number of sub-modules.
        # Input embedding modules
        # self.norm=torch.nn.LayerNorm(normalized_shape=(hidden_size,))
        self.source_embeddings = TransformerEmbeddings(source_vocab_size, hidden_size, max_sequence_length)
        self.target_embeddings = TransformerEmbeddings(target_vocab_size, hidden_size, max_sequence_length)
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([TransformerBlock(hidden_size, intermediate_size, num_attention_heads, hidden_dropout_prob, is_decoder=False)
                                              for _ in range(num_encoder_layers)])
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([TransformerBlock(hidden_size, intermediate_size, num_attention_heads, hidden_dropout_prob, is_decoder=True)
                                              for _ in range(num_decoder_layers)])

    def forward_encoder(self, input_ids: torch.LongTensor, padding_mask: torch.BoolTensor) -> torch.FloatTensor:
        """Implement the forward pass of the encoder.

        Args:
            input_ids: tensor of shape (batch_size, sequence_length) containing the input token ids to the encoder.
            padding_mask: tensor of shape (batch_size, sequence_length) indicating which encoder tokens are padding tokens (`True`)
                    and should be ignored in self-attention computations.

        Returns:
            Tensor of shape (batch_size, sequence_length, hidden_size) containing the output embeddings of the encoder.
        """

        # TODO Implement this function
    # Forward pass through source embeddings
        encoder_embeddings = self.source_embeddings(input_ids)
        for eb in self.encoder_blocks:
            encoder_embeddings = eb(encoder_embeddings, padding_mask)
        return encoder_embeddings

    def forward_decoder(self,
                        input_ids: torch.LongTensor,
                        padding_mask: torch.BoolTensor,
                        encoder_outputs: torch.FloatTensor,
                        encoder_padding_mask: torch.BoolTensor) -> torch.FloatTensor:
        """Implement the forward pass of the decoder.

        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) containing the input token ids to the decoder.
            padding_mask: Tensor of shape (batch_size, sequence_length) indicating which decoder tokens are padding tokens (`True`)
                    and should be ignored in self-attention computations.
            encoder_outputs: Tensor of shape (batch_size, encoder_sequence_length, hidden_size) containing the output embeddings of the encoder.
            encoder_padding_mask: Tensor of shape (batch_size, encoder_sequence_length) indicating which encoder tokens are padding tokens (`True`)
                    and should be ignored in cross-attention computations.

        Returns:
            Tensor of shape (batch_size, sequence_length, target_vocabulary_size)
            containing the logits for predicting the next token in the target sequence.
        """

        # TODO Implement this function
        decoder_embeddings=self.target_embeddings(input_ids)
        for db in self.decoder_blocks:
          decoder_embeddings = db(decoder_embeddings, padding_mask, encoder_outputs, encoder_padding_mask)

        logits = self.target_embeddings.compute_logits(decoder_embeddings)
        return logits

    def forward(self, encoder_input_ids, encoder_padding_mask, decoder_input_ids, decoder_padding_mask):

        encoder_outputs = self.forward_encoder(encoder_input_ids, encoder_padding_mask)
        decoder_logits = self.forward_decoder(decoder_input_ids, decoder_padding_mask, encoder_outputs, encoder_padding_mask)
        return decoder_logits



if __name__ == "__main__":
    # Test your EncoderDecoderModel implementation
    model = EncoderDecoderModel(source_vocab_size=100,
                                target_vocab_size=100,
                                hidden_size=64,
                                intermediate_size=256,
                                num_attention_heads=4,
                                num_encoder_layers=2,
                                num_decoder_layers=2,
                                max_sequence_length=128,
                                hidden_dropout_prob=0.1)

    encoder_input_ids = torch.randint(0, 100, (32, 128))
    encoder_padding_mask = torch.randint(0, 2, (32, 128), dtype=torch.bool)
    decoder_input_ids = torch.randint(0, 100, (32, 128))
    decoder_padding_mask = torch.randint(0, 2, (32, 128), dtype=torch.bool)

    decoder_logits = model(encoder_input_ids, encoder_padding_mask, decoder_input_ids, decoder_padding_mask)
    assert decoder_logits.shape == (32, 128, 100)
    print("Model test passed")