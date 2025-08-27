from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Model, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import Tensor
import numpy as np

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class TokenEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        return self.norm(self.projection(x))

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len=3, stride=1, dropout=0):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        # self.value_embedding = nn.Linear(patch_len, d_model)
        
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x = x.permute(-1,  x.shape[2], x.shape[3])
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        # x = self.value_embedding(x)
        # print("x: ", x.shape)
        # raise ValueError
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.patch_embedding = PatchEmbedding(d_model=hidden_dim)
        
    def forward(self, x):
        x = self.patch_embedding(x.permute(0, 2, 1))  # (B, 1, D)
        
        return x  # (B, 1, D)


class TimeSeriesEncoder1(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.value_embedding(x.permute(0, 2, 1))  # (B, 1, D)
        return x  # (B, 1, D)

class TimeSeriesEncoder2(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, num_layers=3):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        value_embed = self.value_embedding(x)  # (B, T, D)
        
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        position_embed = self.position_embedding(positions)  # (B, T, D)
        
        embeddings = value_embed + position_embed
        
        embeddings = embeddings.permute(1, 0, 2)  # (T, B, D) for transformer
        encoded = self.transformer_encoder(embeddings)
        return encoded.permute(1, 0, 2)  # (B, T, D)


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, text_features, time_features):
        """
        text_features: (seq_len, batch, embed_dim)
        time_features: (time_steps, batch, embed_dim)
        """
        attn_output, _ = self.attn(
            query=text_features,
            key=time_features,
            value=time_features,
            need_weights=False
        )
        return self.norm(text_features + attn_output)

class TimeAwareGPT2(GPT2LMHeadModel):
    def __init__(self, config, hidden_dim, d_model=768, num_layer=4, num_heads=8):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.tokenizer = GPT2Tokenizer.from_pretrained('/root/daye/gpt2')
        self.tokenizer.add_special_tokens({
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'pad_token': '[PAD]'
        })
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(d_model, num_heads) for _ in range(num_layer)
        ])
        self.time_proj = nn.Linear(hidden_dim, d_model)
        self.resize_token_embeddings(len(self.tokenizer))
        
    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        time_features = kwargs.pop("time_features", None)
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        use_cache = kwargs.get("use_cache", self.config.use_cache)
        output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        inputs_embeds = inputs_embeds.permute(1, 0, 2)  # (T, B, D)

        if time_features is not None:
            time_features = self.time_proj(time_features)
            time_features = time_features.permute(1, 0, 2)
        else:
            time_features = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for idx, (transformer_layer, cross_attn) in enumerate(zip(self.transformer.h, self.cross_attn_layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states.permute(1, 0, 2),)  # 恢复(B, T, D)

            layer_past = past_key_values[idx] if past_key_values is not None else None

            transformer_outputs = transformer_layer(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = transformer_outputs[0]

            hidden_states = cross_attn(hidden_states, time_features)

            if use_cache:
                presents += (transformer_outputs[1],)

            if output_attentions:
                all_self_attentions += (transformer_outputs[2],)

        hidden_states = self.transformer.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states.permute(1, 0, 2),)

        logits = self.lm_head(hidden_states).permute(1, 0, 2)  # (B, T, vocab_size)

        if not return_dict:
            return tuple(v for v in [logits, presents, all_hidden_states, all_self_attentions] if v is not None)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = TimeSeriesEncoder(input_dim=args.input_dim, hidden_dim=args.time_dim)
        base_config = GPT2Config.from_pretrained('gpt2')
        self.decoder = TimeAwareGPT2(base_config, hidden_dim=args.time_dim, d_model=args.text_dim, num_layer=args.num_layer_cross, num_heads=args.n_heads_cross)
        
        pretrained_dict = torch.load('gpt2/pytorch_model.bin')
        model_dict = self.decoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.decoder.load_state_dict(model_dict, strict=False)


    def forward(self, x, captions):
        time_features = self.encoder(x)  # (batch, seq_len, 256)
        
        outputs = self.decoder(
            input_ids=captions,
            time_features=time_features
        )
        return outputs.logits

    def generate(self, x, max_length, temperature=0.9, top_k=50):
        self.decoder.eval()
        device = x.device
        batch_size = x.size(0)
        
        forbidden_words = ["end", "middle", "beginning", "is", "at", "the", "in", "from", "ends", "starts"]
        # forbidden_words = ["in"]
        forbidden_tokens = self.decoder.tokenizer.convert_tokens_to_ids(forbidden_words)
        
        forbidden_tokens += [
            self.decoder.tokenizer.eos_token_id,
            self.decoder.tokenizer.pad_token_id
        ]
        
        input_ids = torch.full(
            (batch_size, 1), 
            self.decoder.tokenizer.bos_token_id,
            device=device,
            dtype=torch.long
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length - 1):
            if finished.all():
                break
                
            attention_mask = torch.ones_like(input_ids) * (~finished).unsqueeze(1)
            
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                time_features=self.encoder(x)
            )
            
            logits = outputs.logits[:, -1, :] / temperature
            
            if step == 0:
                forbid_mask = torch.zeros_like(logits, dtype=torch.bool)
                for token in forbidden_tokens:
                    forbid_mask |= (torch.arange(logits.size(-1), device=device) == token)
                
                logits = logits.masked_fill(forbid_mask, -float('inf'))
            
            if top_k > 0:
                top_k_logits = torch.topk(logits, top_k)
                indices_to_remove = logits < top_k_logits.values[..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            eos_mask = (next_tokens == self.decoder.tokenizer.eos_token_id)
            finished = finished | eos_mask
            
            next_tokens = torch.where(finished, self.decoder.tokenizer.pad_token_id, next_tokens)
            
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)

        return input_ids
