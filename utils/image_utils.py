import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# standard values used while training vgg19 on imagenet_1k dataset
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


## Utils for loading and displaying images

def loader(image_name, imsize):
    image = Image.open(image_name)
    loader = transforms.Compose(
        [
            transforms.Resize(imsize),  # scale imported image
            transforms.ToTensor(),  # transform it into a torch tensor
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # transforms.Lambda(lambda x: x.mul_(255)),
        ]
    )
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def unloader(tensor):
    unloader = transforms.Compose(
        [
            # transforms.Lambda(lambda x: x.mul_((1./255))),
            transforms.Normalize(-IMAGENET_MEAN / IMAGENET_STD, 1 / IMAGENET_STD),
            transforms.Lambda(lambda x: x.clamp_(0, 1)),
            transforms.ToPILImage(),  # reconvert into PIL image
        ]
    )

    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    return image


# function to recolorize the output image according to the content image
def recolorize(output_img, content_img, strength=1):
    # convert PIL images to numpy ndarrays
    output_img = transforms.ToTensor()(output_img).numpy()
    content_img = transforms.ToTensor()(content_img).numpy()

    output_img = np.moveaxis(output_img, 0, -1)
    content_img = np.moveaxis(content_img, 0, -1)

    # extract only the luminance (using value of HSV) of the produced image
    content_hsv = cv2.cvtColor(content_img, cv2.COLOR_RGB2HSV)
    output_hsv = cv2.cvtColor(output_img, cv2.COLOR_RGB2HSV)

    # prseve the hue and saturation of the content image by specified strength
    # lower strength means more color of style image
    output_hsv[:, :, 0] = output_hsv[:, :, 0] * (1 - strength) + content_hsv[:, :, 0] * strength
    output_hsv[:, :, 1] = output_hsv[:, :, 1] * (1 - strength) + content_hsv[:, :, 1] * strength

    # convert back to RGB and then to PIL image
    final_rgb = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2RGB)
    # return transforms.ToPILImage()(final_rgb)
    return final_rgb

def imshow(image, title=None):

    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.imshow(image)
