import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

from model import PrunableNN
from train import train_model
import config

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.BATCH_SIZE)

    results = []

    from utils import plot_gate_distribution   

    for lam in config.LAMBDA_VALUES:
        print(f"\nTraining with lambda = {lam}")

        model = PrunableNN()

        acc, sparsity = train_model(model, trainloader, testloader, lam, config.DEVICE, config.EPOCHS)

        gates = model.get_all_gates()
        plot_gate_distribution(gates, lam)

        results.append([lam, acc, sparsity])

    df = pd.DataFrame(results, columns=["Lambda", "Accuracy", "Sparsity"])
    df.to_csv("results/metrics.csv", index=False)

    print(df)


if __name__ == "__main__":
    main()