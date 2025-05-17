import re
import sys
from models.time_caption_model import Model
from data_provider.data_factory_caption import data_provider
import torch.nn as nn
import warnings
import os
import torch 
from torch import optim
import numpy as np
from bert_score import BERTScorer
from utils.tools_caption import EarlyStopping
import hashlib
import json
import threading


warnings.filterwarnings('ignore')

class Exp_time_caption(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # self.expert = BERTScorer(lang="en", model_type="/root/daye/bert-base-uncased", num_layers=12)
        
        self.model = self.model.to(self.device)        
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _build_model(self):
        model = Model(self.args).float()
        
        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
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
                # P, R, F1 = self.expert.score(batch_captions, y)
                # P_all.append(P.mean().item())
                # R_all.append(R.mean().item())
                # F_all.append(F1.mean().item())
                generated_captions.extend(batch_captions)        
                
        avg_loss = total_loss / len(test_loader)
        print(f'[Test] Average Loss: {avg_loss:.4f}')
        # print(f'[Test] Average BertScore: {sum(F_all) / len(F_all):.4f}')
        
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
            
        with open(os.path.join(self.args.save_dir, 'generated_captions.txt'), 'w') as f:
            for caption in generated_captions:
                f.write(caption + '\n')
        
        return avg_loss, generated_captions
    
    def generate(self, x_batch, max_length, test=0, setting=""):
        
        x_batch = x_batch.to(self.device)
        
        if test==2:
            # print('loading Time Series Caption model')
            self.model.load_state_dict(torch.load('/root/daye/TSCaption/TS_Caption_GPT/checkpoints/timecaption_batch_size32_epochs100_dim768/checkpoint.pth'))
            
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

def parser_caption():
    # 定义默认参数
    args = {
        'root_path': './datasets',
        'data_path': 'timecaption.json',
        'data': 'timecaption',
        'batch_size': 32,
        'num_workers': 4,
        'epochs': 100,
        'patience': 10,
        'learning_rate': 1e-4,
        'scale': False,
        'save_dir': './results',
        'use_multi_gpu': False,
        'devices': '0,1',
        'use_gpu': True,
        'gpu_type': 'cuda',
        'gpu': 0,
        'n_heads_cross': 8,
        'num_layer_cross': 8,
        'input_dim': 12,
        'time_dim': 256,
        'text_dim': 768,
        'is_training': 1,
        'itr': 1,
        'max_length': 40,
        'checkpoints': '/root/daye/TSCaption/TS_Caption_GPT/checkpoints/'
    }

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--'):
            arg_name = arg[2:]
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                value = sys.argv[i + 1]
                if arg_name in ['scale', 'use_multi_gpu']:
                    args[arg_name] = True
                elif arg_name in ['batch_size', 'num_workers', 'epochs', 'patience', 'n_heads_cross', 'num_layer_cross', 'input_dim', 'time_dim', 'text_dim', 'is_training', 'itr', 'max_length', 'gpu']:
                    args[arg_name] = int(value)
                elif arg_name in ['learning_rate']:
                    args[arg_name] = float(value)
                else:
                    args[arg_name] = value
                i += 2
            else:
                if arg_name in ['scale', 'use_multi_gpu']:
                    args[arg_name] = True
                i += 1
        else:
            i += 1

    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return Args(**args)

# class TSCaption():
#     def __init__(self, max_length):
#         self.args = parser_caption()
#         self.max_length = max_length
#         if self.args.use_gpu and self.args.use_multi_gpu:
#             self.args.devices = self.args.devices.replace(' ', '')
#             device_ids = self.args.devices.split(',')
#             self.args.device_ids = [int(id_) for id_ in device_ids]
#             self.args.gpu = self.args.device_ids[0]
    
#         self.setting = '{}_batch_size{}_epochs{}_dim{}'.format('timecaption', self.args.batch_size, self.args.epochs, self.args.text_dim)
#         self.exp = Exp_time_caption(self.args)

#     def generate(self, x):
#         with torch.no_grad():
#             x = self.exp.generate(x, max_length=self.max_length, test=2, setting=self.setting)
#         return x


# class TSCaption():
#     def __init__(self, max_length, data):
#         self.args = parser_caption()
#         self.max_length = max_length
#         if self.args.use_gpu and self.args.use_multi_gpu:
#             self.args.devices = self.args.devices.replace(' ', '')
#             device_ids = self.args.devices.split(',')
#             self.args.device_ids = [int(id_) for id_ in device_ids]
#             self.args.gpu = self.args.device_ids[0]
    
#         self.setting = '{}_batch_size{}_epochs{}_dim{}'.format(
#             'timecaption', self.args.batch_size, self.args.epochs, self.args.text_dim)
#         self.exp = Exp_time_caption(self.args)

#         self.cache_file = data + 'tscaption_cache.json'
#         self.lock = threading.Lock()
#         self.cache = {}
#         if os.path.exists(self.cache_file):
#             try:
#                 with open(self.cache_file, 'r') as f:
#                     self.cache = json.load(f)
#             except (json.JSONDecodeError, IOError):
#                 self.cache = {}

#     def _compute_md5(self, x_tensor):
#         x_contiguous = x_tensor.contiguous()
#         x_np = x_contiguous.cpu().detach().numpy()
#         shape = np.array(x_np.shape, dtype=np.int32).tobytes()
#         data = x_np.tobytes()
#         return hashlib.md5(shape + data).hexdigest()

#     def generate(self, x):
#         cache_key = self._compute_md5(x)
        
#         with self.lock:
#             if cache_key in self.cache:
#                 return self.cache[cache_key]
        
#         with torch.no_grad():
#             generated = self.exp.generate(
#                 x, 
#                 max_length=self.max_length,
#                 test=2,
#                 setting=self.setting
#             )
        
#         with self.lock:
#             if cache_key not in self.cache:
#                 self.cache[cache_key] = generated
#                 try:
#                     with open(self.cache_file, 'w') as f:
#                         json.dump(self.cache.copy(), f)
#                 except IOError:
#                     pass
        
#         return generated

#     def __del__(self):
#         with self.lock:
#             try:
#                 with open(self.cache_file, 'w') as f:
#                     json.dump(self.cache, f)
#             except:
#                 pass

class TSCaption():
    def __init__(self, max_length, data, seq_len):
        self.args = parser_caption()
        self.max_length = max_length
        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(' ', '')
            device_ids = self.args.devices.split(',')
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]
    
        self.setting = '{}_batch_size{}_epochs{}_dim{}'.format(
            'timecaption', self.args.batch_size, self.args.epochs, self.args.text_dim)
        self.exp = Exp_time_caption(self.args)

        # 初始化缓存
        self.cache_file = data + str(seq_len) + 'tscaption_cache.json'
        self.lock = threading.Lock()
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = {}

    def _compute_sample_md5(self, sample_tensor):
        """计算单个样本的MD5哈希"""
        # 输入维度为 [1, L]
        sample_contiguous = sample_tensor.contiguous()
        sample_np = sample_contiguous.cpu().detach().numpy()
        return hashlib.md5(sample_np.tobytes()).hexdigest()

    def generate(self, x):
        """样本级缓存生成"""
        batch_size = x.size(0)
        results = []
        need_generate = []

        # 阶段1：缓存查询
        with self.lock:
            for i in range(batch_size):
                sample = x[i]  # 获取第i个样本 [1, L]
                cache_key = self._compute_sample_md5(sample)
                
                if cache_key in self.cache:
                    results.append(self.cache[cache_key])
                else:
                    results.append(None)
                    need_generate.append((i, sample, cache_key))

        # 全部命中直接返回
        if not need_generate:
            return results

        # 阶段2：批量生成未命中样本
        with torch.no_grad():
            # 构建子批次输入 [n_samples, 1, L]
            sub_batch = torch.stack([item[1] for item in need_generate])
            
            # 调用生成接口，假设支持任意batch_size
            generated = self.exp.generate(
                sub_batch,
                max_length=self.max_length,
                test=2,
                setting=self.setting
            )

        # 阶段3：更新缓存和结果
        with self.lock:
            for (idx, _, key), output in zip(need_generate, generated):
                if key not in self.cache:
                    self.cache[key] = output
                results[idx] = output
            
            # 异步保存缓存
            self._save_cache_async()

        return results

    def _save_cache_async(self):
        """异步安全保存缓存"""
        try:
            # 使用副本避免长时间锁
            cache_copy = self.cache.copy()
            with open(self.cache_file, 'w') as f:
                json.dump(cache_copy, f)
        except Exception as e:
            print(f"Cache save error: {str(e)}")

    def __del__(self):
        with self.lock:
            self._save_cache_async()