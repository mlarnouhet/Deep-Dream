from PIL import Image
from fastai.vision.all import show_image as imshow
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import cv2 as cv
import numpy as np



def pseudo_loss(feature_maps):
    maps = torch.flatten(feature_maps)
    return torch.sum(maps ** 2) / len(maps)  # If using optimizer, add minus sign

def pre_process(img):
    tensor = transforms.ToTensor()(img).to(DEVICE).unsqueeze(0)
    tensor.requires_grad = True
    return tensor

def post_process(img):
    return np.moveaxis(img.to('cpu').detach().numpy()[0], 0, 2)


class Vgg16(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.features = models.vgg16(pretrained=True, progress=True).features
        self.layer_id = layer_id
        self.network = nn.Sequential(*[self.features[i] for i in range(self.layer_id)])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.network(x)

LEARNING_RATE = 1e-4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
N_iter = 1000
layer_id = 24
layer_ids_to_use = [4]
pyramid_levels = 5
ratio = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Im_path = "data/Original_images/sky.jpeg"
image = np.array(Image.open(Im_path))
image = cv.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
model = Vgg16(layer_id).to(device=DEVICE)


for level in list(reversed(range(pyramid_levels))):
    size = (int(IMAGE_HEIGHT / (ratio ** level)), int(IMAGE_WIDTH / (ratio ** level)))
    image = cv.resize(image, size)
    image = pre_process(image)

    for i in range(N_iter):
        feature_maps = model(image)
        loss = pseudo_loss(feature_maps)
        loss.backward()
        g_std = torch.std(image.grad.data)
        g_mean = torch.mean(image.grad.data)
        image.grad.data = image.grad.data - g_mean
        image.grad.data = image.grad.data / g_std
        image.data += LEARNING_RATE * image.grad.data
        image.grad.data.zero_()

    image = post_process(image)

imshow(image)

print("end")