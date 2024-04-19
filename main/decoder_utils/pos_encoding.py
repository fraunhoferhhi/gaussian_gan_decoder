import torch


class Embedder:
    def __init__(
        self, include_input=True, input_dims=3, num_freqs=10, log_sampling=False, periodic_fns=[torch.sin, torch.cos]
    ):
        self.periodic_fns = periodic_fns
        self.log_sampling = log_sampling
        self.num_freqs = num_freqs
        self.input_dims = input_dims
        self.include_input = include_input
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(self.identity)
            out_dim += d

        max_freq = self.num_freqs - 1
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(self.frequency_activation(p_fn=p_fn, freq=freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def frequency_activation(p_fn, freq):
        def func(x_in):
            return p_fn(x_in * freq)
        return func

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


if __name__ == "__main__":
    num_freqs = 10
    embedder = Embedder(
        include_input=True, input_dims=3, num_freqs=num_freqs, log_sampling=False, periodic_fns=[torch.sin, torch.cos]
    )
    res = embedder(torch.tile(torch.linspace(0, 1, 100)[:, None], [1, 3]))
    print()
