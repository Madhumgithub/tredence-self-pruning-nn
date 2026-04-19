import matplotlib.pyplot as plt
import os

def plot_gate_distribution(gates, lambda_val):
    os.makedirs("results/plots", exist_ok=True)

    plt.figure()
    plt.hist(gates.detach().cpu().numpy(), bins=50)
    plt.title(f"Gate Distribution (lambda={lambda_val})")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    plt.savefig(f"results/plots/gate_distribution_{lambda_val}.png")
    plt.close()