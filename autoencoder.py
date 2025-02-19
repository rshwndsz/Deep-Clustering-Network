from collections import OrderedDict

import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(args.latent_dim)
        self.dims_list = args.hidden_dims + args.hidden_dims[:-1][::-1]  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == self.latent_dim

        # Encoder Network
        print(self.__repr__())
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update({"linear0": nn.Linear(int(self.input_dim), int(hidden_dim)), "activation0": nn.ReLU()})
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(int(self.hidden_dims[idx - 1]), int(hidden_dim)),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(int(self.hidden_dims[idx])),
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(int(hidden_dim), int(self.output_dim)),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(int(hidden_dim), int(tmp_hidden_dims[idx + 1])),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(int(tmp_hidden_dims[idx + 1])),
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = "[Structure]: {}-".format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += "{}-".format(dim)
        repr_str += str(self.output_dim) + "\n"
        repr_str += "[n_layers]: {}".format(self.n_layers) + "\n"
        repr_str += "[n_clusters]: {}".format(self.n_clusters) + "\n"
        repr_str += "[input_dims]: {}".format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)
