import torch

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

LAMBDA_VALUES = [0.01, 0.1, 1.0]

THRESHOLD = 1e-2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"