

def load_from_pkl_new_G(network_pkl, device, model):
    if model == "panohead":
        from PanoHead.training.triplane import TriPlaneGenerator
        from PanoHead.torch_utils import misc
    elif model == "eg3d":
        from eg3d.training.triplane import TriPlaneGenerator
        from eg3d.torch_utils import misc

    G = load_from_pkl(network_pkl, device, model)
    print("Reloading Modules!")
    kwargs = dict(G.init_kwargs)
    G_new = TriPlaneGenerator(*G.init_args, **kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new
    return G


def load_from_pkl(network_pkl, device, model):
    if model == "panohead":
        from PanoHead import dnnlib, legacy
    elif model == "eg3d":
        from eg3d import dnnlib, legacy
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device)
