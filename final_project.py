# imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# encoder architecture
def make_encoder():
    layers = []
    in_channels = 3
    channels = [32, 64, 128, 128, 256, 256]
    for c in channels:
        layers.append(nn.Conv2d(in_channels, c, 3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = c
    layers.append(nn.Flatten())
    layers.append(nn.Linear(256 * 1 * 1, 2))
    return nn.Sequential(*layers)

# generate synthetic test images
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 165, 0), (128, 0, 128)
]

def random_color():
    return COLORS[np.random.randint(len(COLORS))]

def generate_shape(shape, color, size=64):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = int(0.90 * size)
    x = (size - s) // 2
    y = (size - s) // 2
    if shape == "circle":
        draw.ellipse((x, y, x + s, y + s), fill=color, outline=(0, 0, 0), width=2)
    elif shape == "square":
        draw.rectangle((x, y, x + s, y + s), fill=color, outline=(0, 0, 0), width=2)
    elif shape == "triangle":
        draw.polygon([(x+s//2, y), (x, y+s), (x+s, y+s)], fill=color, outline=(0, 0, 0), width=2)
    return img

class SyntheticShapesDataset(Dataset):
    def __init__(self, n_per_class=10):
        self.classes = ["circle", "square", "triangle"]
        self.n = n_per_class
        self.total = len(self.classes) * n_per_class

    def __len__(self): return self.total

    def __getitem__(self, idx):
        label = idx // self.n
        shape = self.classes[label]
        img = generate_shape(shape, random_color())
        return img, label

# plot embeddings
def plot_embeddings(encoder1, encoder2, dataset, title1="Shape-Sensitive (Ts)", title2="Color-Sensitive (Tc)"):
    # set encoders to evaluation mode
    encoder1.eval()
    encoder2.eval()
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    zs1, zs2, imgs = [], [], []
    for img, label in dataset:
        # preprocess and move image to device
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            # get embeddings from both encoders
            z1 = encoder1(x).cpu().numpy()[0]
            z2 = encoder2(x).cpu().numpy()[0]
        zs1.append(z1 / np.linalg.norm(z1))
        zs2.append(z2 / np.linalg.norm(z2))
        imgs.append(img)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, zs, title in zip(axs, [zs1, zs2], [title1, title2]):
        ax.set_title(title)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.grid(True)
        for (x, y), img in zip(zs, imgs):
            # remove white background from images
            img = img.convert("RGBA").resize((20, 20))
            data = np.array(img)
            r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
            white_mask = (r > 240) & (g > 240) & (b > 240)
            data[white_mask, 3] = 0
            img_rgba = Image.fromarray(data)
            imagebox = OffsetImage(img_rgba, zoom=1.0)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

    plt.tight_layout()
    plt.show()

# main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained models
    shape_encoder = make_encoder().to(device)
    color_encoder = make_encoder().to(device)
    shape_encoder.load_state_dict(torch.load("shape_encoder6.pth", map_location=device))
    color_encoder.load_state_dict(torch.load("color_encoder.pth", map_location=device))

    # create small test set
    test_dataset = SyntheticShapesDataset(n_per_class=30)

    # plot embeddings
    plot_embeddings(
        shape_encoder, color_encoder,
        test_dataset,
        title1="Shape-Sensitive Embedding (Ts)",
        title2="Color-Sensitive Embedding (Tc)"
    )