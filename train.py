import os

import model
import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, trainloader, testloader, lambda_val, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            classification_loss = criterion(outputs, labels)

            # Sparsity loss
            gates = model.get_all_gates()
            sparsity_loss = torch.mean(torch.abs(gates))

            loss = classification_loss + lambda_val * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    acc = evaluate(model, testloader, device)
    sparsity = calculate_sparsity(model)

    import os
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/model_lambda_{lambda_val}.pt")

    return acc, sparsity


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def calculate_sparsity(model, threshold=0.1):
    gates = model.get_all_gates()

    pruned = (gates < threshold).sum().item()
    total = gates.numel()

    return 100 * pruned / total