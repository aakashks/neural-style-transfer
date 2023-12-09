from style_transfer import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='images/neckarfront.jpg')
    parser.add_argument('--style', type=str, default='images/vangogh-starry-night.jpg')
    parser.add_argument('--style_weight', type=float, default=1e7)
    parser.add_argument('--content_weight', type=float, default=1e2)
    parser.add_argument('--tv_weight', type=float, default=1e-6)
    parser.add_argument('--steps', type=int, default=600)
    args = parser.parse_args()
    
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    
    output_img = run_style_transfer(
        cnn, args.content, args.style, 
        style_weight=args.style_weight, content_weight=args.content_weight, tv_weight=args.tv_weight, 
        num_steps=args.steps
    )
    
    imshow(output_img, title='Output Image')
    plt.savefig('output/result.png')