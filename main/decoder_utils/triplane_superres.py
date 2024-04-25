import torch.nn

from torch_utils import persistence
from training.superresolution import SuperresolutionHybrid4XCustom, SuperresolutionHybrid2XCustom


@persistence.persistent_class
class TriplaneSuperres1024(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.superres = SuperresolutionHybrid4XCustom(channels=96 * 3, img_resolution=1024, sr_num_fp16_res=0, sr_antialias=False)

    def forward(self, x, ws):
        x = x.reshape(1, 96, 512, 512)
        x = self.superres(rgb=None, x=x, ws=ws)
        x = x.reshape(1, 3, 32, 512, 512)
        return x


@persistence.persistent_class
class TriplaneSuperres512(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.superres = SuperresolutionHybrid2XCustom(channels=96 * 3, img_resolution=512, sr_num_fp16_res=0, sr_antialias=False)

    def forward(self, x, ws):
        x = x.reshape(1, 96 * 3, 512, 512)
        x = self.superres(rgb=None, x=x, ws=ws)
        x = x.reshape(1, 3, 32, 512, 512)
        return x
