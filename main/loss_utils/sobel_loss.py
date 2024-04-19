import torch
import torch.nn.functional as F

sobel_y = [
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
]
sobel_x = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
]
sobel_kernel_x = torch.tensor(sobel_x, dtype=torch.float32, device="cuda").unsqueeze(0).expand(1, 3, 3, 3)
sobel_kernel_y = torch.tensor(sobel_y, dtype=torch.float32, device="cuda").unsqueeze(0).expand(1, 3, 3, 3)



def sobel_loss(render, target):
    render_x_sob = F.conv2d(render.unsqueeze(0), sobel_kernel_x, stride=1, padding=1)
    target_x_sob = F.conv2d(target.unsqueeze(0), sobel_kernel_x, stride=1, padding=1)

    render_y_sob = F.conv2d(render.unsqueeze(0), sobel_kernel_y, stride=1, padding=1)
    target_y_sob = F.conv2d(target.unsqueeze(0), sobel_kernel_y, stride=1, padding=1)

    diff_x = torch.square(render_x_sob - target_x_sob)
    diff_y = torch.square(render_y_sob - target_y_sob)
    diff = diff_x + diff_y
    return diff.mean(), diff
