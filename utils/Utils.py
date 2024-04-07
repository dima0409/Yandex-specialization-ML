import csv
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import clear_output
from torch import device, inference_mode
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


# Method for learning callback
def plot_stats(
        train_loss: list[float],
        valid_loss: list[float],
        train_accuracy: list[float],
        valid_accuracy: list[float],
        title: str,
        save: str = None
):
    plt.clf()
    plt.subplot(2, 1, 1)

    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.subplot(2, 1, 2)

    plt.title(title + ' accuracy')
    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(valid_accuracy, label='Valid accuracy')
    plt.legend()
    plt.tight_layout()
    plt.pause(0.001)
    if save:
        plt.savefig(save)


# def plot_multi_processing(q):
#     sns.set(style='darkgrid')
#     info = q.get()
#     plot_stats(**info)


def train(model: nn.Module, data_loader: DataLoader, x_field_name: str, y_field_name: str, optimizer: Optimizer,
          loss_fn, device: device):
    model.train()

    total_loss = 0
    total_correct = 0

    for sample in tqdm(data_loader):
        x, y = sample[x_field_name].to(device), sample[y_field_name].to(device)
        x = x.to(torch.float32)
        # y = y.to(torch.float64)
        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output, y)

        loss.backward()

        total_loss += loss.item()

        total_correct += (output.argmax(dim=1) == y).sum().item()

        optimizer.step()

    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)


@inference_mode()
def evaluate(model: nn.Module, data_loader: DataLoader, x_field_name: str, y_field_name: str, loss_fn, device: device):
    model.eval()

    total_loss = 0
    total_correct = 0

    for sample in tqdm(data_loader):
        x, y = sample[x_field_name].to(device), sample[y_field_name].to(device)
        x = x.to(torch.float32)
        # y = y.to(torch.float32)
        output = model(x)

        loss = loss_fn(output, y)

        total_loss += loss.item()

        total_correct += (output.argmax(dim=1) == y).sum().item()

    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)


def fit(model, train_loader, valid_loader, x_field_name: str, y_field_name: str, optimizer, loss_fn, num_epochs,
        title, device: device):
    model.to(device)
    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []
    base_path = f"../model_results/{title}_variants/"
    create_path_if_not_exists(base_path + "chart")
    create_path_if_not_exists(base_path + "tables")

    # csv field names
    fields = ['Train loss', 'Test loss', 'Train accuracy', 'Test accuracy']

    # name of csv file
    filename = base_path + "tables/metrics.csv"
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

    for epoch in range(num_epochs):
        print(f"Start epoch {epoch}")
        train_loss, train_accuracy = train(model, train_loader, x_field_name, y_field_name, optimizer, loss_fn, device)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, x_field_name, y_field_name, loss_fn, device)

        # writing to csv file
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)

            writer.writerow({'Train loss': train_loss, 'Test loss': valid_loss, 'Train accuracy': train_accuracy,
                             'Test accuracy': valid_accuracy})

        torch.save(model.state_dict(), base_path + title + "_" + str(epoch) + "_epoch")
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(valid_accuracy)

        if len(train_loss_history) >= 3:
            plot_stats(train_loss_history[1:], valid_loss_history[1:], train_accuracy_history[1:],
                       valid_accuracy_history[1:], title)
    plot_stats(train_loss_history[1:], valid_loss_history[1:], train_accuracy_history[1:],
               valid_accuracy_history[1:], title, save=base_path + "chart/callback.pdf")


def create_path_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
