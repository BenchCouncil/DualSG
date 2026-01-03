# DualSG

[![arXiv](https://img.shields.io/badge/arXiv-2507.21830-8bc34a.svg)](https://arxiv.org/abs/2507.21830)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**[<a href="https://mp.weixin.qq.com/s/3eitZo9SEmZ01QxeXKGhPw">中文解读1</a>]**
**[<a href="https://mp.weixin.qq.com/s/zry87cssdiVhpNmxZdATHg">中文解读2</a>]**
**[<a href="https://mp.weixin.qq.com/s/XxjTpkGfJcJDWP0r19R6-Q">中文解读3</a>]**

This is the official implementation of our paper:  **DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework**

![Framework](./idea.png)

> Multivariate Time Series Forecasting (MTSF) is fundamental to many real-world applications. Recent studies leverage Large Language Models (LLMs) for MTSF, but treating them as end-to-end forecasters often sacrifices numerical precision, while latent-space alignment between text and time series remains challenging. We propose **DualSG**, a dual-stream framework that uses LLMs as **semantic guidance modules** rather than standalone predictors. DualSG introduces **Time Series Caption**, an explicit natural-language summary of temporal patterns, providing interpretable context for LLMs. A caption-guided fusion module further captures inter-variable dependencies with reduced noise and computation. Experiments across diverse datasets show consistent improvements over 15 state-of-the-art methods.

![Idea](./framework.png)



## Project Structure
```
├── models/ 
├── exp/ # Experiment runners
├── data_provider/ # Dataset loaders
├── layers/ # Transformer components
├── utils/ # Tools, losses, metrics
├── TS_Caption_GPT/ # Time-aware GPT2 decoder and checkpoints
├── scripts/ # Shell scripts for reproducibility
├── run.py # Entry point for numerical forecasting
├── requirements.txt # Python dependencies
└── README.md
```

## Setup

```bash
conda create -n dualsg python=3.9
conda activate dualsg
pip install -r requirements.txt
```
Place your datasets under ./dataset/. See data_provider/ for supported formats.

process data can be download in : https://drive.google.com/file/d/15hsRcUyx79uJ5aiXFK1LA-zxbMY3aw8W/view?usp=sharing

# Usage
Long-term Forecasting
```bash
bash DualSG/scripts/DualSG/DualSG_ETTh1.sh
```



# Citation
If you find our work helpful, please consider citing us:

```bibtex
@inproceedings{ding2025dualsg,
  title={DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework},
  author={Ding, Kuiye and Fan, Fanda and Wang, Yao and Jian, Ruijie and Wang, Xiaorui and Gong, Luqi and Jiang, Yishan and Luo, Chunjie and Zhan, Jianfeng},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={508--517},
  year={2025}
}
```
# Acknowledgements

All the experiment datasets are public, and we obtain them from the following links:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library/.

- TFB: https://github.com/decisionintelligence/TFB.

- TimeMixer: https://github.com/kwuking/TimeMixer.
