import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from pathlib import Path

from network import VAE, Encoder, Decoder
from config import (MSE_LOSS_SCALE, DEVICE, TRAIN_BATCH_SIZE,
                    TRAIN_EPOCHS, MODELS_PATH, DATA_PATH)


def train_fn(
        model: VAE, optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader, test_dataloader: DataLoader = None):
    train_loop = tqdm(train_dataloader)
    train_loop.set_description("Train")
    running_mse_loss = 0
    running_kl_loss = 0

    for i, (x, y) in enumerate(train_loop):
        x = x.to(model.device)

        mean, log_var, pred = model.forward(x)

        mse_loss_value = F.mse_loss(pred, x) * MSE_LOSS_SCALE
        running_mse_loss += mse_loss_value.item()

        kl_loss_value = model.kl_loss(mean, log_var)
        running_kl_loss += kl_loss_value.item()

        total_loss = mse_loss_value + kl_loss_value
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loop.set_postfix_str(
            f"mse loss: {running_mse_loss / (i + 1):.5f}; kl loss: {running_kl_loss / (i + 1):.5f}")

    if test_dataloader is None:
        return

    test_loop = tqdm(test_dataloader)
    test_loop.set_description("Eval")
    running_mse_loss = 0
    running_kl_loss = 0

    for i, (x, y) in enumerate(test_loop):
        x = x.to(model.device)

        with torch.no_grad():
            mean, log_var, pred = model.forward(x)

        mse_loss_value = F.mse_loss(pred, x) * MSE_LOSS_SCALE
        running_mse_loss += mse_loss_value.item()

        kl_loss_value = model.kl_loss(mean, log_var)
        running_kl_loss += kl_loss_value.item()

        test_loop.set_postfix_str(
            f"mse loss: {running_mse_loss / (i + 1):.5f}; kl loss: {running_kl_loss / (i + 1):.5f}")


def main():
    train_data = MNIST(DATA_PATH, train=True,
                       transform=ToTensor(), download=True)
    test_data = MNIST(DATA_PATH, train=False,
                      transform=ToTensor(), download=True)

    train_dataloader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=64)

    encoder = Encoder()
    decoder = Decoder()
    vae = VAE(encoder, decoder, DEVICE)

    optimizer = torch.optim.AdamW(vae.parameters())

    for epoch in range(TRAIN_EPOCHS):
        train_fn(vae, optimizer, train_dataloader, test_dataloader)

    encoder = vae.encoder
    decoder = vae.decoder
    torch.save(encoder.state_dict(), Path(MODELS_PATH, "vae_encoder.pt"))
    torch.save(decoder.state_dict(), Path(MODELS_PATH, "vae_decoder.pt"))


if __name__ == "__main__":
    main()