import PIL
import torch
import torchvision.transforms

transform_image = torchvision.transforms.Compose([
    torchvision.transforms.Resize((120, 160)),
    torchvision.transforms.Lambda(lambda x: x.crop((0, 40, 160, 120))),
    torchvision.transforms.ToTensor()
])


def pre_process_image(image_nb_array):
    """
    Pre-processes an image before it is fed to the neural network.
    :param image_nb_array: The image as a numpy array.
    :return: Tensor of shape (1, 120, 160)
    """
    pil_image = PIL.Image.fromarray(image_nb_array)
    t_image = transform_image(pil_image)
    t_image = torch.unsqueeze(t_image, dim=0)
    return t_image
