import torch
from torch import nn
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer



class AbstractEmbedder(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def encode(self, *args, **kwargs):
        raise NotImplementedError

class T5TokenEmbedder(AbstractEmbedder):

    def __init__(self, tokenizer_version='google/t5-v1_1-small',
                 device='cuda', tokenizer_max_legth=77,**kwargs):

        super().__init__()
        self.tknz = T5Tokenizer.from_pretrained(tokenizer_version)
        self.vocab_size = self.tknz.vocab_size
        self.device = device
        self.max_len = tokenizer_max_legth


    def forward(self,text, return_lengths=False,**kwargs):
        batch_encoding = self.tknz(text, return_length=True, max_length=self.max_len, truncation=True,
                                   return_overflowing_tokens=False)
        tokens = batch_encoding['input_ids']
        len_ = batch_encoding['length']

        if return_lengths:
            return tokens, len_
        else:
            return tokens, None

    @torch.no_grad()
    def encode(self, text, **kwargs):
        return self(text,**kwargs)




__TEXT_EMBEDDERS__ = {
    'T5Tokenizer': T5TokenEmbedder
}