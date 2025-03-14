import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from VQ_VAE import VQVAE

IMG_PATH = '/home/yukino/Documents/Code/Datasets/CelebA/Img/img_align_celeba'
BATCH_SIZE = 64

class CelebADataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    dataset = CelebADataset(IMG_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    vqvae_model = VQVAE(3, 256, 64, 4, 512, 0.25)

    for i, data in enumerate(dataloader):
        output = vqvae_model(data)
        print(output.shape)
        break
