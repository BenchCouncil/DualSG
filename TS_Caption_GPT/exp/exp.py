import re
from model.time_caption_model import Model
from data_provider.data_factory import data_provider
import torch.nn as nn
import warnings
import os
import torch 
from torch import optim
import numpy as np
from bert_score import BERTScorer
from utils.tools import EarlyStopping
warnings.filterwarnings('ignore')

class Exp_time_caption(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.expert = BERTScorer(lang="en", model_type="/root/daye/bert-base-uncased", num_layers=12)
        
        self.model = self.model.to(self.device)        
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _build_model(self):
        model = Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag=flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss(ignore_index=self.model.decoder.tokenizer.pad_token_id)
        return criterion
    
    # def preprocess_caption(self, text):
    #     text = text.lower().replace(",", " ,").replace(".", " .")
    #     return f"[BOS] {text} [EOS]"
    
    def preprocess_caption(self, text):
        # 规范化处理
        text = text.strip().lower()
        text = re.sub(r'[^a-zA-Z0-9,.!?]', ' ', text)
        # 处理标点空格（与GPT的tokenization规则一致）
        text = text.replace(',', ' ,').replace('.', ' .') 
        # 使用GPT自带的特殊标记格式
        return f"{self.model.decoder.tokenizer.bos_token} {text} {self.model.decoder.tokenizer.eos_token}"

    def train(self, setting):
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_data, train_loader = self._get_data(flag='train')
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()
            print("Epoch {} :".format(epoch))
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.unsqueeze(-1).to(self.device)  # (batch, seq_len) -> (batch, seq_len, 1)
                
                # print(y[0])
                # raise KeyError
                captions = self.model.decoder.tokenizer(
                    [self.preprocess_caption(ann) for ann in y], 
                    padding="max_length",
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                # bos_tokens = torch.full((x.size(0), 1), self.model.decoder.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
                # captions = torch.cat([bos_tokens, captions], dim=1)
                # print(captions)
                # raise KeyboardInterrupt
                # 模型前向
                input_caption = captions.input_ids[:, :-1]
                output_caption = captions.input_ids[:, 1:]
                # print("input_caption: ", input_caption.shape)
                # print("output_caption: ", output_caption.shape)
                # print("output_caption: ", output_caption.reshape(-1).shape)
                # raise KeyboardInterrupt
                outputs = self.model(x, input_caption)
                # print("Input Caption Shape:", input_caption.shape)  # 应为 (B, T)
                # print("Outputs Shape:", outputs.shape)
                # 计算损失
                seq_len = x.size(1)  # 获取时间序列的序列长度T
                # print("seq_len: ", seq_len)
                # logits = outputs.logits[:, seq_len:, :].contiguous()  # 截取后L个位置的logits
                logits = outputs
                # print(f"output_caption min: {output_caption.min()}, max: {output_caption.max()}")
                # print(f"logits shape: {logits}")
                # # raise KeyboardInterrupt
                
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    output_caption.reshape(-1)
                )
                
                # print("outputs: ", logits.shape)
                # print("outputs: ", captions.input_ids.shape)
                
                # 反向传播
                self.model_optim.zero_grad()
                train_loss.append(loss.item())
                loss.backward()
                self.model_optim.step()
                
                if (batch_idx + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_idx + 1, epoch + 1, loss.item()))
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(flag = 'val')
            test_loss = self.vali(flag = 'test')

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def vali(self, flag):
        val_data, val_loader = self._get_data(flag=flag)
        self.model.eval()
        total_loss = 0.0
        self.criterion = self._select_criterion()
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.unsqueeze(-1).to(self.device)
                
                # 文本预处理
                captions = self.model.decoder.tokenizer(
                    [self.preprocess_caption(ann) for ann in y], 
                    padding="max_length",
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                # bos_tokens = torch.full((x.size(0), 1), self.model.decoder.tokenizer.bos_token_id, dtype=torch.long, device=self.model.device)
                # captions = torch.cat([bos_tokens, captions], dim=1)
                
                input_caption = captions.input_ids[:, :-1]
                output_caption = captions.input_ids[:, 1:]
                
                # 模型前向
                outputs = self.model(x, input_caption)
                
                # 计算损失（对齐维度）
                seq_len = x.size(1)
                # logits = outputs.logits[:, seq_len:, :].contiguous()
                logits = outputs
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    output_caption.reshape(-1)
                )
                total_loss += loss.item()
                
        
        avg_loss = total_loss / len(val_loader)
        print(f'Average Loss: {avg_loss:.4f}')
        return avg_loss

    
    def test(self, setting, test=1):
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            
        self.model.eval()
        
        test_data, test_loader = self._get_data(flag='test')
        total_loss = 0.0
        generated_captions = []
        P_all = []
        R_all = []
        F_all = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x = x.unsqueeze(-1).to(self.device)

                captions = self.model.decoder.tokenizer(
                    [self.preprocess_caption(ann) for ann in y], 
                    padding="max_length",
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                output_caption = captions.input_ids[:, 1:]
                # 添加检查
                assert output_caption.min() >= 0 and output_caption.max() < len(self.model.decoder.tokenizer), "Token IDs 越界!"
                # bos_tokens = torch.full((x.size(0), 1), self.model.decoder.tokenizer.bos_token_id, dtype=torch.long, device=self.model.device)
                # captions = torch.cat([bos_tokens, captions], dim=1)
                
                input_caption = captions.input_ids[:, :-1]
                output_caption = captions.input_ids[:, 1:]
                
                outputs = self.model(x, input_caption)
                # print(self.generate(x))
                
                seq_len = x.size(1)
                # logits = outputs.logits[:, seq_len:, :].contiguous()
                logits = outputs
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    output_caption.reshape(-1)
                )
                total_loss += loss.item()
                
                batch_x = x.clone()
                batch_captions = self.generate(batch_x, max_length=self.args.max_length, setting=setting)
                P, R, F1 = self.expert.score(batch_captions, y)
                P_all.append(P.mean().item())
                R_all.append(R.mean().item())
                F_all.append(F1.mean().item())
                generated_captions.extend(batch_captions)        
                
        avg_loss = total_loss / len(test_loader)
        print(f'[Test] Average Loss: {avg_loss:.4f}')
        print(f'[Test] Average BertScore: {sum(F_all) / len(F_all):.4f}')
        
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
            
        with open(os.path.join(self.args.save_dir, 'generated_captions.txt'), 'w') as f:
            for caption in generated_captions:
                f.write(caption + '\n')
        
        return avg_loss, generated_captions
    
    def generate(self, x_batch, max_length, test=0, setting=""):
        
        x_batch = x_batch.to(self.device)
        
        if test==2:
            print('loading Time Series Caption model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            
        self.model.eval()
        with torch.no_grad():
            # 生成token_ids
            token_ids = self.model.generate(x_batch, max_length)
            
            # 解码为文本
            captions = []
            for ids in token_ids:
                # 跳过特殊token并解码
                caption = self.model.decoder.tokenizer.decode(
                    ids.cpu().numpy(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                captions.append(caption)
                
            return captions

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device