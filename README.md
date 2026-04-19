# Self-Pruning Neural Network (Tredence AI Engineering Case Study)

## 📌 Overview
This project implements a **self-pruning neural network** that dynamically learns to remove less important weights during training using learnable gating mechanisms.

Unlike traditional pruning (post-training), this approach enables the model to **adapt its structure during training**, resulting in a more efficient and compact architecture.

---

## ⚙️ Key Features
- Custom `PrunableLinear` layer with learnable gates
- End-to-end differentiable pruning mechanism
- L1-based sparsity regularization
- Trade-off analysis between accuracy and sparsity
- Gate distribution visualization
- FastAPI-based inference API (production-oriented)

---

## 🧠 Core Idea

Each weight is associated with a learnable gate:

- Gate ≈ 1 → Important weight (kept)
- Gate ≈ 0 → Unimportant weight (pruned)

### Loss Function

Total Loss = Classification Loss + λ × Sparsity Loss

- Classification Loss → Ensures accuracy  
- Sparsity Loss (L1 on gates) → Encourages pruning  

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.01   | 50.99       | 1.75         |
| 0.10   | 51.59       | 7.78         |
| 1.00   | 50.82       | 24.82        |

### Observations
- Increasing λ increases sparsity significantly  
- Accuracy remains relatively stable despite pruning  
- At λ = 1.0, the model removes ~25% of weights with minimal accuracy drop  
- Demonstrates effective self-pruning behavior with controlled trade-off    

---

## 📈 Visualizations

Gate distribution plots are generated to show:
- Spike near 0 → successful pruning  
- Remaining active weights → important connections  

---

## 🏗️ Project Structure

```text
tredence-self-pruning-nn/
│
├── main.py          # Training pipeline
├── model.py         # Prunable model definition
├── train.py         # Training + evaluation logic
├── utils.py         # Plotting utilities
├── config.py        # Hyperparameters
├── api.py           # FastAPI inference service
│
├── results/         # Artifacts (metrics, plots, saved models)
│   ├── metrics.csv
│   ├── plots/
│   └── model_*.pt
│
└── report/          # Documentation
    └── report.md

```

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python main.py
```

## 🌐 Run API (FastAPI)
```bash
uvicorn api:app --reload
```

Open in browser:
```bash
http://127.0.0.1:8000/docs
```

## 📦 API Endpoint

POST /predict/

Input: Image file (CIFAR-10 format)

Output:
```bash
{
  "prediction": "cat"
}
```

## 🎯 Key Learnings

- Implementing differentiable pruning mechanisms
- Understanding sparsity-inducing regularization
- Managing trade-offs in ML systems
- Building production-ready ML APIs

## 👤 Author
Madhumita P (RA2311026010866) - B.Tech CSE (AIML), SRM Institute of Science and Technology

