# DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework

[![arXiv](https://img.shields.io/badge/arXiv-2507.21830-8bc34a.svg)](https://arxiv.org/abs/2507.21830)

## 🚀 Highlights

- 💡 **Dual-stream forecasting** decouples numerical and semantic modeling to resolve LLM precision bottlenecks.
- 🔁 **Time series captioning** generates intermediate semantic representations to guide numerical forecasting.
- 🤖 **TimeAwareGPT2 decoder** enhances LLM alignment with temporal patterns using temporal position control.

---

## 🗂️ Project Structure
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

## 📦 Setup

```bash
conda create -n dualsg python=3.9
conda activate dualsg
pip install -r requirements.txt
```
Place your datasets under ./dataset/. See data_provider/ for supported formats.

# 🧪 Usage
🔢 Long-term Forecasting (PatchTST + DualSG)
```bash
bash DualSG/scripts/DualSG/DualSG_ETTh1.sh
```



# 📎 Citation
If you find our work helpful, please consider citing us:

```bibtex
@inproceedings{da2025dualsg,
  title={DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework},
  author={Kuiye Ding and Fanda Fan and Yao Wang and Ruijie jian and Xiaorui Wang and Luqi Gong and Yishan Jiang and Chunjie Luo an Jianfeng Zhan},
  booktitle={ACM MM},
  year={2025}
}
```
# 🙌 Acknowledgements

All the experiment datasets are public, and we obtain them from the following links:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library/.

- TFB: https://github.com/decisionintelligence/TFB.

- TimeMixer: https://github.com/kwuking/TimeMixer.
