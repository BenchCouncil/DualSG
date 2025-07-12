# DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework

This is the official PyTorch implementation of our **ACM MM 2025** paper:

> **DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework**  
> 📄 [Paper](./TODO_LINK_TO_PAPER.pdf)

---

## 🚀 Highlights

- 💡 **Dual-stream forecasting** decouples numerical and semantic modeling to resolve LLM precision bottlenecks.
- 🔁 **Time series captioning** generates intermediate semantic representations to guide numerical forecasting.
- 🤖 **TimeAwareGPT2 decoder** enhances LLM alignment with temporal patterns using temporal position control.

---

## 🗂️ Project Structure
```
├── models/ # Core model logic (DualSG, GPT2, etc.)
├── exp/ # Experiment runners
├── data_provider/ # Dataset loaders (ETT, PEMS, M4, etc.)
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
  author={},
  booktitle={ACM MM},
  year={2025}
}
```
# 🙌 Acknowledgements

