import PIL
import time

import torch

import Jetbot import Camera
import cv2
from torchvision.transforms import transforms

from vae.vae import VAE

image_channels = 3
VARIANTS_SIZE = 32

def load_vae(model_path, device):
    vae = VAE(image_channels=image_channels, z_dim=VARIANTS_SIZE)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    vae.to(torch.device(device))
    vae.eval()
    return vae

def preprocess(image):
    observe = PIL.Image.fromarray(image)
    observe = observe.resize((160,120))
    croped = observe.crop((0, 40, 160, 120))
    tensor = transforms.ToTensor()(croped)
    return tensor

if __name__ == '__main__':


    vae = load_vae('./vae.path', device='cuda')

    camera = Camera(width=320,height=240, fps=31)
    camera.start()



    while True:
        start = time.time()


        image = camera.value
        tensor = preprocess(image)
        tensor.to('cuda')
        z, _, _ = vae.encode(torch.stack((tensor,tensor),dim=0)[:-1].to('cuda'))

        elapsed_time = time.time() - start
        print("Elapsed {}".format(elapsed_time))

