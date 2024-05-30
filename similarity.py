import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def transform_(img_pil):
    transforms_all = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()])  
    try:
        for img_index in range(img_pil.shape[0]):
            img_pil[img_index,:,:,:]  = transforms_all(img_pil[img_index,:,:,:])
    except:
        try:
            img_pil     =   np.asarray(img_pil)
            for img_index in range(img_pil.shape[0]):
                img_pil[img_index,:,:,:]  = transforms_all(img_pil[img_index,:,:,:])
        except: return transforms_all(img_pil)
    return img_pil

class ImagePairDataset(Dataset):
    def __init__(self, image_pairs, transform_apply=None):
        self.image_pairs = image_pairs
        self.transform = transform_apply

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32768, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = BaseNetwork()

    def forward(self, img1, img2):
        feat1 = self.base_network(transform_(img1))
        feat2 = self.base_network(transform_(img2))
        return feat1, feat2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

def train_siamese_network(image_pairs, num_epochs=20, batch_size=16, learning_rate=0.001):
    transform__ = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = ImagePairDataset(image_pairs, transform_apply=transform__)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, label in dataloader:
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    return model


image_pairs = [
    ("car_white_0.jpg", "car_white_1.jpg", 1),
    ("car_white_0.jpg", "car_negative_example_0.jpg", 0),
    # Add more image pairs
]

model = train_siamese_network(image_pairs)


def compute_similarity(model, img1_path, img2_path, transform=transform_):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    img1 = transform_(img1).unsqueeze(0).cuda()
    img2 = transform_(img2).unsqueeze(0).cuda()

    model.eval()
    with torch.no_grad():
        feat1, feat2 = model(img1, img2)
        similarity = nn.functional.pairwise_distance(feat1, feat2).item()
    return similarity

similarity = compute_similarity(model, "car_white_0.jpg", "car_negative_example_0.jpg")
print(f"Similarity score: {similarity}")

