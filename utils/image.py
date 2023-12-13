import torch
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# standard values used while training vgg19 on imagenet_1k dataset
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

## Utils for loading and displaying images

def image_loader(image_name, imsize):
    loader = transforms.Compose(
        [
            transforms.Resize(imsize),  # scale imported image
            transforms.ToTensor(),  # transform it into a torch tensor
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # transforms.Lambda(lambda x: x.mul_(255)),
        ]
    )
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
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

    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.imshow(image)
