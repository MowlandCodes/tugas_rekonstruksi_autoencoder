import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].values
        label = int(row[0])
        image = row[1:].astype(np.uint8).reshape(28, 28)
        image = np.expand_dims(image, axis=2)
        image = self.transform(image)
        return image, image, label


class EncoderVAE(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            plt.cm.register_cmap if False else nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = EncoderVAE(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        out = self.decoder(z)
        return out, mu, logvar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI Tool untuk Rekonstruksi Citra VAE - Opsi A"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path ke file weights model (.pth)"
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Indeks data Fashion-MNIST yang mau diekstrak",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=2,
        help="Ukuran latent dimension model (2, 8, atau 32)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/fashion-mnist_test.csv",
        help="Path ke file dataset CSV",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[-] Error: File model '{args.model}' tidak ditemukan!")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[+] Loading dataset dari {args.dataset}...")
    dataset = FashionMNISTDataset(args.dataset)
    if args.index >= len(dataset) or args.index < 0:
        print(
            f"[-] Error: Indeks {args.index} out of bounds! Maksimal data cuma {len(dataset) - 1}"
        )
        exit(1)

    orig_tensor, _, _ = dataset[args.index]
    orig_tensor = orig_tensor.unsqueeze(0).to(device)

    print(f"[+] Memuat arsitektur VAE dengan Latent Dimension: {args.latent_dim}...")
    model = VAE(args.latent_dim).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.eval()

    print(f"[+] Memproses indeks gambar ke-{args.index}...")
    with torch.no_grad():
        recon_tensor, _, _ = model(orig_tensor)

    img_orig = orig_tensor.squeeze().cpu().numpy()
    img_recon = recon_tensor.squeeze().cpu().numpy()

    plt.imsave("original.png", img_orig, cmap="gray")
    plt.imsave("reconstructed.png", img_recon, cmap="gray")

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img_orig, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_recon, cmap="gray")
    axes[1].set_title(f"Reconstructed (Dim {args.latent_dim})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    print(
        "[+] SUCCESS: File 'original.png', 'reconstructed.png', dan 'comparison.png' berhasil di-dump!"
    )
