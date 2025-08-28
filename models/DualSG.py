import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from exp.exp_caption import TSCaption
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoConfig, AutoModel
import numpy as np

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class AdaptiveImportanceMask(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.linear_alpha = nn.Linear(d_model, 1)

    def forward(self, Ei):
        # Ei: [bs * nvars, patch_num, d_model]

        # Compute temporal gradient norm: ∇_t E_i
        grad = Ei[:, 1:, :] - Ei[:, :-1, :]  # [bs * nvars, patch_num - 1, d_model]
        grad_norm = torch.norm(grad, p=2, dim=-1)  # [bs * nvars, patch_num - 1]
        grad_norm = F.pad(grad_norm, (1, 0), mode='constant', value=0)  # -> [bs * nvars, patch_num]

        # Normalize gradient
        grad_norm = grad_norm / (grad_norm.max(dim=1, keepdim=True)[0] + 1e-8)

        # Multi-head self-attention
        attn_out, _ = self.msa(Ei, Ei, Ei)  # [bs * nvars, patch_num, d_model]

        # Linear projection to attention logits
        attn_score = self.linear_alpha(attn_out).squeeze(-1)  # [bs * nvars, patch_num]

        # Combine with normalized gradient
        combined_score = grad_norm * attn_score  # [bs * nvars, patch_num]

        # Pass through sigmoid to get importance scores
        alpha = torch.sigmoid(combined_score)  # [bs * nvars, patch_num]

        return alpha


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        patch_len = configs.patch_len
        self.args = configs

        if configs.patch_adaptive == 0:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, patch_len, stride, padding, configs.dropout)
        else:
            # patching and embedding
            self.num_patches1 = int((configs.seq_len - 8) / 4 + 2)
            self.num_patches2 = int((configs.seq_len - 16) / 8 + 2)
            self.num_patches3 = int((configs.seq_len - 32) / 16 + 2)
            
            total_patches = self.num_patches2  # Total should match the (16,8) setting
            if self.pred_len == 96:
                ratios = [0.75, 0.25, 0]  # More from (8,4)
            elif self.pred_len == 192:
                ratios = [5/12, 7/12, 0]  # Balanced
            elif self.pred_len == 336:
                ratios = [1/6, 1/2, 1/3]  # More from (32,16)
            else:  # 720
                ratios = [0, 7/12, 5/12]  # Mostly from (32,16)

            self.k1 = int(ratios[0] * total_patches)
            self.k2 = int(ratios[1] * total_patches)
            self.k3 = total_patches - self.k1 - self.k2  # Ensure sum is total_patches

            # Ensure we don't exceed available patches for each configuration
            self.k3 = min(self.k3, self.num_patches3)
            self.k2 = min(self.k2, self.num_patches2)
            self.k1 = min(self.k1, self.num_patches1)
            
            self.patch_embedding1 = PatchEmbedding(
                configs.d_model, patch_len=8, stride=4, padding=2, dropout=configs.dropout)
            self.patch_embedding2 = PatchEmbedding(
                configs.d_model, patch_len=16, stride=8, padding=4, dropout=configs.dropout)
            self.patch_embedding3 = PatchEmbedding(
                configs.d_model, patch_len=32, stride=16, padding=8, dropout=configs.dropout)

            # Adjust remaining if any of the k's were capped
            remaining = total_patches - (self.k1 + self.k2 + self.k3)
            # Distribute remaining to the largest ratio component (heuristic)
            if remaining > 0:
                if ratios[2] >= max(ratios):
                    self.k3 += remaining
                elif ratios[0] >= max(ratios):
                    self.k1 += remaining
                else:
                    self.k2 += remaining
        
        self.num_channels = configs.enc_in 
        self.weight = nn.Parameter(
            torch.ones(self.num_channels, configs.pred_len) * 0.5,
            requires_grad=True
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        self.caption = TSCaption(max_length=configs.max_length, data=configs.data_path[:-4], seq_len=configs.seq_len)
        self.pool_type = configs.pool_type

        self.aim = AdaptiveImportanceMask(d_model=configs.d_model, num_heads=configs.n_heads)
        
        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        
        # llm model
        if configs.llm_model == 'LLAMA2':
            llama2_path = 'Llama-2-7b-hf'
            self.llama_config = LlamaConfig.from_pretrained(llama2_path, attn_implementation="eager")
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    llama2_path,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    llama2_path,
                    # trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # device_map="auto"
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    llama2_path,
                    # trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    llama2_path,
                    # trust_remote_code=True,
                    local_files_only=False
                )
            llm_dim = self.llm_model.config.hidden_size
        elif configs.llm_model == 'LLAMA3':
            # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
            llama3_path = "Llama-3.2-3B"
            cache_path = "./"

            # Load the configuration with custom adjustments
            self.config =  LlamaConfig.from_pretrained(llama3_path)

            self.config.num_hidden_layers = configs.llm_layers
            self.config.output_attentions = True
            self.config.output_hidden_states = True
            
            self.llm_model  = LlamaModel.from_pretrained(
                llama3_path,
                # device_map="auto",
                # config=self.config,
                # token=self.hug_token,cache_dir=cache_path
            )
            llm_dim = self.llm_model.config.hidden_size
            # self.tokenizer = AutoTokenizer.from_pretrained(llama3_path,use_auth_token=self.hug_token,cache_dir=cache_path)
            self.tokenizer = AutoTokenizer.from_pretrained(llama3_path)
        elif configs.llm_model == 'GPT2':
            gpt2_path = 'gpt2'
            self.gpt2_config = GPT2Config.from_pretrained(gpt2_path, attn_implementation="eager")

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            llm_dim = self.gpt2_config.hidden_size
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    gpt2_path,
                    # device_map="auto",
                    # trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    gpt2_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                    # device_map="auto"
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    gpt2_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    # device_map="auto"
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    gpt2_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    # device_map="auto"
                )
        elif configs.llm_model == 'Deepseek':
            qwen_path = 'DeepSeek-R1-Distill-Qwen-1.5B'
            self.qwen_config = AutoConfig.from_pretrained(qwen_path)

            self.qwen_config.num_hidden_layers = configs.llm_layers
            self.qwen_config.output_attentions = True
            self.qwen_config.output_hidden_states = True

            llm_dim = self.qwen_config.hidden_size

            try:
                self.llm_model = AutoModel.from_pretrained(
                    qwen_path,
                    config=self.qwen_config,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local Qwen model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    qwen_path,
                    config=self.qwen_config,
                    local_files_only=False
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen_path,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local Qwen tokenizer files not found. Attempting to download...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen_path,
                    local_files_only=False
                )
        elif configs.llm_model == 'QWEN':
            qwen25_path = 'Qwen2.5-3B'
            self.qwen25_config = AutoConfig.from_pretrained(qwen25_path)

            self.qwen25_config.num_hidden_layers = configs.llm_layers
            self.qwen25_config.output_attentions = True
            self.qwen25_config.output_hidden_states = True

            llm_dim = self.qwen25_config.hidden_size

            try:
                self.llm_model = AutoModel.from_pretrained(
                    qwen25_path,
                    config=self.qwen25_config,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local Qwen2.5-3B model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    qwen25_path,
                    config=self.qwen25_config,
                    local_files_only=False
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen25_path,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local Qwen2.5-3B tokenizer files not found. Attempting to download...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen25_path,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            bert_path = 'bert-base-uncased'
            self.bert_config = BertConfig.from_pretrained(bert_path)

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True

            llm_dim = self.bert_config.hidden_size

            try:
                self.llm_model = BertModel.from_pretrained(
                    bert_path,
                    local_files_only=True,
                    config=self.bert_config
                )
            except EnvironmentError:
                print("Local BERT model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    bert_path,
                    local_files_only=False,
                    config=self.bert_config
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    bert_path,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local BERT tokenizer files not found. Attempting to download them...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    bert_path,
                    local_files_only=False
                )
        else:
            llm_dim = 768
            raise Exception('LLM model is not defined')
        
        if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        self.caption_proj = nn.Sequential(
            nn.Linear(llm_dim, configs.pred_len * 2),
            nn.ReLU(),
            nn.LayerNorm(configs.pred_len * 2),
            nn.Linear(configs.pred_len * 2, configs.pred_len),
            nn.ReLU(),
            nn.LayerNorm(configs.pred_len)
        )
        
        for param in self.llm_model.parameters():
            param.requires_grad = False
        if self.args.text_cd:
            self.channel_fusion = ChannelMoE(channels=self.num_channels, k=configs.top_k)

    def apply_patch_embeddings(self, x_enc):
        if self.args.patch_adaptive == 0:
            enc_out, n_vars = self.patch_embedding(x_enc)
            return enc_out, n_vars
        else:
            """Applies all three patch embeddings and returns concatenated result"""
            # x_enc: [bs, n_vars, seq_len]
            enc1, n_vars1 = self.patch_embedding1(x_enc)
            enc2, n_vars2 = self.patch_embedding2(x_enc)
            enc3, n_vars3 = self.patch_embedding3(x_enc)
            assert n_vars1 == n_vars2 == n_vars3, "Mismatch in number of variables"

            # Select appropriate patches from each
            enc3_part = enc3[:, :self.k3, :]  # Front-most
            start_idx = (self.num_patches2 - self.k2) // 2
            enc2_part = enc2[:, start_idx:start_idx+self.k2, :]  # Middle
            enc1_part = enc1[:, -self.k1:, :]  # End

            # Concatenate in temporal order: front (32,16) -> middle (16,8) -> end (8,4)
            if self.k1 == 0:
                combined = torch.cat([enc3_part, enc2_part], dim=1)
            elif self.k3 == 0:
                combined = torch.cat([enc2_part, enc1_part], dim=1)
            else:
                combined = torch.cat([enc3_part, enc2_part, enc1_part], dim=1)
            return combined, n_vars1

    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.apply_patch_embeddings(x_enc)
        alpha = self.aim(enc_out)
        enc_out = enc_out * alpha.unsqueeze(-1)  

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
    
    def prompt_embed(self, x_enc):
        unfolded_x = x_enc.permute(0, 2, 1).unfold(dimension=-1, size=x_enc.size(1) // 12, step=x_enc.size(1) // 12).mean(dim=-1)
        min_vals = torch.min(unfolded_x, dim=2, keepdim=True)[0]
        max_vals = torch.max(unfolded_x, dim=2, keepdim=True)[0]
        denominator = max_vals - min_vals
        denominator = torch.where(denominator == 0, torch.tensor(1e-8, dtype=denominator.dtype, device=denominator.device), denominator)
        unfolded_x = (unfolded_x - min_vals) / denominator

        captions = []
        for idx in range(unfolded_x.size(1)):
            slice_tensor = unfolded_x[:, idx:idx+1, :]
            # print("slice_tensor: ", slice_tensor.shape)
            
            output_text = self.caption.generate(slice_tensor.permute(0, 2, 1))
            # print("output_text: ", output_text)
            # print("output_text: ", len(output_text))
            # raise KeyboardInterrupt
            captions.append(output_text)

        # captions = [[str(sublist) for sublist in batch] for batch in zip(*captions)]
        # add information
        captions = [
            [f"<|start_prompt|Make predictions about the future based on the past time series caption: {caption}<|<end_prompt>|>" for caption in row]
            for row in captions
        ]
        captions = [[str(sublist) for sublist in batch] for batch in zip(*captions)]
        return np.array(captions)

    def prompt_embed1(self, x_enc):
        prompt = "The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. This dataset consists of 2 years data from two separated counties in China. To explore the granularity on the Long sequence time-series forecasting (LSTF) problem, different subsets are created, {ETTh1, ETTh2} for 1-hour-level and ETTm1 for 15-minutes-level. Each data point consists of the target value ”oil temperature” and 6 power load features. The train/val/test is 12/4/4 months."
        captions = [prompt for i in range(x_enc.size(0))]
        return captions

    def prompt_embed2(self, x_enc):
        prompt = "Predict the future time step given the trend, season and residual."
        captions = [prompt for i in range(x_enc.size(0))]
        return captions
    
    def prompt_embed3(self, x_mark_enc):
        batchsize, l, channel = x_mark_enc.shape
        
        # 确保通道数足够
        if channel < 4:
            raise ValueError("Input tensor must have at least 4 channels")
        
        output = []
        for b in range(batchsize):
            first = x_mark_enc[b, 0, :4]
            last = x_mark_enc[b, -1, :4]
            
            first_str = ", ".join([f"{v:.4f}" for v in first.tolist()])
            last_str = ", ".join([f"{v:.4f}" for v in last.tolist()])
            combined_str = f"{first_str}; {last_str}"
            
            batch_row = [combined_str] * 7
            output.append(batch_row)
        
        return output  # shape: (batchsize, 7)

    def pearson_corr(self, a, b):
        if len(a) < 2 or len(b) < 2:
            return torch.tensor(0.0)
        a_mean = torch.mean(a)
        b_mean = torch.mean(b)
        a_centered = a - a_mean
        b_centered = b - b_mean
        cov = torch.sum(a_centered * b_centered)
        std_a = torch.sqrt(torch.sum(a_centered ** 2))
        std_b = torch.sqrt(torch.sum(b_centered ** 2))
        if std_a == 0 or std_b == 0:
            return torch.tensor(0.0)
        return cov / (std_a * std_b)

    def compute_input_statistics(self, input_tensor):
        batchsize, l, channel = input_tensor.shape
        output_list = []
        
        for b in range(batchsize):
            batch_stats = []
            for c in range(channel):
                x = input_tensor[b, :, c]
                
                min_val = x.min().item()
                max_val = x.max().item()
                median_val = x.median().item()
                
                trend = "upward" if x[-1] > x[0] else "downward"
                
                stats_str = (
                    f"Input statistics: min value {min_val:.4f}, max value {max_val:.4f}, "
                    f"median value {median_val:.4f}, the trend is {trend}, "
                )
                batch_stats.append(stats_str)
            
            output_list.append(batch_stats)
        
        return output_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            
            if self.args.prompt == 1:
                captions = self.prompt_embed(x_enc)
            elif self.args.prompt == 2:
                captions = self.compute_input_statistics(x_enc)
            elif self.args.prompt == 3:
                captions = self.prompt_embed1(x_enc)
            elif self.args.prompt == 4:
                captions = self.prompt_embed2(x_enc)
            elif self.args.prompt == 5:
                captions = self.prompt_embed3(x_mark_enc)

            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # process text
            batch_size, num_channels = dec_out.size(0), dec_out.size(2)
            flattened_captions = [captions[b][c] for b in range(batch_size) for c in range(num_channels)]
            
            if self.args.use_fullmodel:
                # Tokenization
                inputs = self.tokenizer(
                    flattened_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(x_enc.device)
                with torch.no_grad():
                    outputs = self.llm_model(
                        **inputs,
                        use_cache=False
                    )
                last_hidden = outputs.last_hidden_state
                
                
            else:
                inputs = self.tokenizer(
                    flattened_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).input_ids.to(x_enc.device)
                last_hidden = self.llm_model.get_input_embeddings()(inputs.to(x_enc.device))            
            
            if self.pool_type == "avg":
                pooled = last_hidden.mean(dim=1)
            elif self.pool_type == "max":
                pooled = last_hidden.max(dim=1).values
            elif self.pool_type == "min":
                pooled = last_hidden.min(dim=1).values
            else:
                raise ValueError(f"Unsupported pool type: {self.pool_type}")

            caption_emb = pooled.view(batch_size, num_channels, -1)  # [B, C, d_model]
            
            # project time dim
            caption_vectors = self.caption_proj(caption_emb)  # [B, C, L]
            caption_vectors = norm(caption_vectors)
            
            if self.args.text_cd:
                caption_vectors = self.channel_fusion(caption_vectors)
                caption_vectors = norm(caption_vectors)
            
            dec_out_transposed = dec_out.permute(0, 2, 1)  # [B, C, L]
            
            # learn weight
            weight = torch.sigmoid(self.weight).unsqueeze(0)  # [1, C, L]

            dec_out = (1 - weight) * dec_out_transposed + weight * caption_vectors

            dec_out = dec_out.permute(0, 2, 1)

            dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
            return dec_out[:, -self.pred_len:, :]
        return None



class ChannelMoE(nn.Module):
    def __init__(self, channels, k=4):
        super().__init__()
        self.channels = channels
        self.k = k
        
        self.weight_gen = nn.Sequential(
            nn.Conv1d(channels, channels*4, 1, groups=channels),
            nn.ReLU(),
            nn.Conv1d(channels*4, channels*channels, 1, groups=channels)
        )
        
        for conv in self.weight_gen:
            if isinstance(conv, nn.Conv1d):
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.norm = nn.LayerNorm(channels)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        B, C, L = x.shape
        
        x_pool = F.adaptive_avg_pool1d(x, 1)  # [B, C, 1]
        
        scores = self.weight_gen(x_pool)
        scores = scores.view(B, C, C)
        scores = self.norm(scores)  # normlayer
        scores = scores / self.temperature  # scale
        
        topk_values, topk_indices = torch.topk(scores, self.k, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_values)
        
        weights = F.softmax(mask, dim=-1)
        
        fused = torch.bmm(weights, x)
        
        return fused + x


def norm(input_emb):

    mean = input_emb.mean(dim=1, keepdim=True).detach()  # [B, 1, L]
    var = torch.var(input_emb, dim=1, keepdim=True, unbiased=False)  # [B, 1, L]
    return (input_emb - mean) / torch.sqrt(var + 1e-5)
