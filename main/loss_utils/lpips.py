import torch
import torch.nn.functional as F
import dnnlib


class NvidiaVGG16:
    def __init__(self, device='cuda'):
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.model = torch.jit.load(f).eval().to(device)

    def __call__(self, img):
        img = (img + 1) / 2 * 255
        return self.model(img, resize_images=False, return_lpips=True)


def perc(
        target: torch.tensor,
        image: torch.tensor,
        vgg: torch.nn.Module,
        downsampling: bool,
):
    # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
    if image.shape[2] > 256 and downsampling:
        image = F.interpolate(image, size=(256, 256), mode='area')
    if target.shape[2] > 256 and downsampling:
        target = F.interpolate(target, size=(256, 256), mode='area')

    # Features for synth images.
    image_features = vgg(image)
    target_features = vgg(target)

    diff = (image_features - target_features).square()
    return diff.sum().mean()