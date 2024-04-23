import torch
from torch import nn

from dnnlib import EasyDict
from main.decoder_models.base_decoder import Decoder
from main.decoder_utils.pos_encoding import Embedder
from torch_utils import persistence
from training.volumetric_rendering.renderer import sample_from_planes


@persistence.persistent_class
class SequentialDecoder(nn.Module):
    def __init__(self, G, hidden_dim=128, use_xyz_embedding=True, use_gen_finetune=True, device="cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_xyz_embedding = use_xyz_embedding
        self.use_gen_finetune = use_gen_finetune
        self.device = device

        position_dim = 3
        if use_xyz_embedding:
            self.embedder = Embedder(include_input=True, input_dims=3, num_freqs=10)
            position_dim = self.embedder.out_dim

        # trainable networks
        self.G = G
        self.xyz_decoder = Decoder(n_features=32 + position_dim, out_features=3, hidden_dim=hidden_dim).to(device)
        self.scale_decoder = Decoder(n_features=32 + position_dim + 3, out_features=3, hidden_dim=hidden_dim).to(device)
        self.rotation_decoder = Decoder(n_features=32 + position_dim + 6, out_features=4, hidden_dim=hidden_dim).to(device)
        self.opacity_decoder = Decoder(n_features=32 + position_dim + 10, out_features=1, hidden_dim=hidden_dim).to(device)
        self.color_decoder = Decoder(n_features=32 + position_dim + 11, out_features=3, hidden_dim=hidden_dim).to(device)

        self.scale_activation = torch.nn.Softplus()

    def activate_scale(self, scale):
        return - self.scale_activation(scale + 5) - 2

    def forward(self, z, gan_camera_params, init_position, truncation_psi):
        ws = self.G.mapping(z, gan_camera_params, truncation_psi=truncation_psi)
        synth = self.G.synthesis(ws, torch.zeros_like(gan_camera_params), noise_mode="const")

        if "triplane_depth" in self.G.rendering_kwargs:
            plane_features = sample_from_planes(
                self.G.renderer.plane_axes,
                synth["feature_planes"],
                init_position.unsqueeze(0),
                padding_mode="zeros",
                box_warp=self.G.rendering_kwargs["box_warp"],
                triplane_depth=self.G.rendering_kwargs["triplane_depth"],
            )
        else:
            plane_features = sample_from_planes(
                self.G.renderer.plane_axes,
                synth["feature_planes"],
                init_position.unsqueeze(0),
                padding_mode="zeros",
                box_warp=self.G.rendering_kwargs["box_warp"],
            )
        plane_features = plane_features[0]

        result = EasyDict()
        if self.use_xyz_embedding:
            current_info = self.embedder(init_position)
        else:
            current_info = init_position
        xyz = self.xyz_decoder(plane_features, current_info) * 0.01 + init_position
        result.xyz = xyz

        current_info = torch.concat([current_info, xyz], dim=-1)
        scale = self.activate_scale(self.scale_decoder(plane_features, current_info))
        result.scale = scale

        current_info = torch.concat([current_info, scale], dim=-1)
        rotation = self.rotation_decoder(plane_features, current_info)
        result.rotation = rotation

        current_info = torch.concat([current_info, rotation], dim=-1)
        opacity = self.opacity_decoder(plane_features, current_info)
        result.opacity = opacity

        current_info = torch.concat([current_info, opacity], dim=-1)
        color = self.color_decoder(plane_features, current_info)
        result.color = color
        return result

    def get_params_custom(self):
        params = list(self.xyz_decoder.parameters())
        params += list(self.scale_decoder.parameters())
        params += list(self.rotation_decoder.parameters())
        params += list(self.opacity_decoder.parameters())
        params += list(self.color_decoder.parameters())

        if self.use_gen_finetune:
            params += list(self.G.parameters())

        return params
