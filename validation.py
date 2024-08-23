import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from tqdm import tqdm
from translob_pytorch import TransLOB  # type: ignore
from loaders.fi2010_loader import Dataset_fi2010


def validate():
    T = 100
    k = 4
    stock = [0, 1, 2, 3, 4]
    train_test_ratio = 0.7
    batch_size = 128
    num_workers = 1
    auction = False
    normalization = "Zscore"
    lighten = False

    days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_day_length = round(len(days) * train_test_ratio)
    train_days = days[:train_day_length]
    test_days = days[train_day_length:]

    dataset_test = Dataset_fi2010(
        auction, normalization, stock, test_days, T, k, lighten
    )

    dataset_val = dataset_test

    epochs = 1

    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = torch.load("loggers/results/translob_2024-08-23_16_36_55/best_val_model.pt")
    model.to(model.device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    for iter in tqdm(range(epochs)):
        val_loss = []
        val_acc = []
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(model.device, dtype=torch.float), targets.to(
                model.device, dtype=torch.int64
            )

            if model.name == "translob":
                inputs = torch.squeeze(inputs, 1)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss.append(loss.item())
            tmp_acc = torch.count_nonzero(
                torch.argmax(outputs, dim=1) == targets
            ).item() / targets.size(0)
            val_acc.append(tmp_acc)

        val_loss = np.mean(val_loss)
        val_acc = np.mean(val_acc)

    print(f"Validation Loss: {val_loss}")
    print(f"Validation Acc: {val_acc}")


if __name__ == "__main__":
    validate()
