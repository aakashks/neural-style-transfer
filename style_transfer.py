import torch.optim as optim

from utils.image import *
from utils.losses import *
from utils.fm_extractor import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# desired depth layers to compute style/content losses -
# manually picked up looking to the cnn layers
# for style these are all ReLUs just after maxpool2d ie. relu_1_1, 2_2 ...
style_layers = ["1", "6", "11", "20", "29"]

# for content use relu_4_2
content_layers = ["22"]


def run_style_transfer(
    model,
    content_img,
    style_img,
    imsize,
    style_weight=1e12,
    content_weight=1e0,
    num_steps=200,
    step_size=50,
):
    content_img = image_loader(content_img, imsize)
    style_img = image_loader(style_img, imsize)
    input_img = content_img.clone().requires_grad_(True)

    model.eval().requires_grad_(False)
    model = model.to(device)

    style_loss_modules = [
        StyleLoss(resp.to(device))
        for _, resp in FeatureExtractor(model, style_layers)(style_img)
    ]
    content_loss_modules = [
        ContentLoss(resp.to(device))
        for _, resp in FeatureExtractor(model, content_layers)(content_img)
    ]

    # these are good weights settings recommended by Leon Gatys in his implementation
    style_weights = [style_weight / n**2 for n in [64, 128, 256, 512, 512]]
    content_weights = [content_weight]

    loss_modules = style_loss_modules + content_loss_modules
    weights = style_weights + content_weights

    optimizer = optim.LBFGS(params=[input_img])
    i = 0
    while i < num_steps:

        def closure():
            optimizer.zero_grad()

            style_fms = FeatureExtractor(model, style_layers)(input_img)
            content_fms = FeatureExtractor(model, content_layers)(input_img)
            fms = style_fms + content_fms
            losses = [weights[i] * mod(fms[i][1]) for i, mod in enumerate(loss_modules)]

            loss = sum(losses)
            loss.backward()
            nonlocal i
            if i % step_size == 0:
                print(f"Step {i}: Total Loss: {loss.detach().item():.4e} All layer losses: ", end="")
                [print(f'{loss.item():.2e}, ', end="") for loss in losses]
                print()
                # imshow(input_img.detach())
                # plt.show()

            i += 1
            return loss

        optimizer.step(closure)

    return input_img.detach()
