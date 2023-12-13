from torchvision.models import vgg19, VGG19_Weights
import argparse
import matplotlib.pyplot as plt

from utils import image
from style_transfer import run_style_transfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--content", type=str, default="images/neckarfront.jpg")
    parser.add_argument("--style", type=str, default="images/vangogh-starry-night.jpg")
    parser.add_argument("--style_weight", type=float, default=1e12)
    parser.add_argument("--content_weight", type=float, default=1e0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="results/result.png")
    args = parser.parse_args()

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    output_img = run_style_transfer(
        cnn,
        args.content,
        args.style,
        args.imsize,
        args.style_weight,
        args.content_weight,
        args.steps,
        args.step_size,
    )

    image.imshow(output_img)
    plt.savefig(args.output_path, dpi=1000, bbox_inches="tight")
