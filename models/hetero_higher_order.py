import torch
from torch_geometric.utils import to_dense_batch

from models.hetero_base_nn import MPNN
from models.util import need_padding


class HigherOrder(MPNN):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm,
                 act):
        super().__init__(hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         num_mlp_layers,
                         norm,
                         act)

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict = self.init_embedding(data)

        if need_padding(batch_dict['_vals']):
            _, real_x_mask = to_dense_batch(x_dict['vals'].new_empty(batch_dict['_vals'].shape[0]),
                                        batch_dict['_vals'])  # B x Nmax x F
            real_x_x_mask = torch.einsum('bn,bm->bnm', real_x_mask, real_x_mask)  # B x Nmax x Nmax
        else:
            real_x_x_mask = None
            B = batch_dict['_vals'].max() + 1
            N = batch_dict['_vals'].shape[0] // B

        feature_dim = x_dict['vals'].shape[-1]
        device = x_dict['vals'].device

        for i, layer in enumerate(self.gcns):
            if real_x_x_mask is not None:
                x_x_dense = torch.zeros(*real_x_x_mask.shape + (feature_dim,), device=device, dtype=torch.float)
                x_x_dense[real_x_x_mask] = x_dict['vals']
            else:
                x_x_dense = x_dict['vals'].reshape(B, N, N, -1)

            x_x_dense = self.higher_orders[i](x_x_dense, real_x_x_mask)

            if real_x_x_mask is not None:
                x_x_dense = x_x_dense[real_x_x_mask]
            else:
                x_x_dense = x_x_dense.reshape(B * N * N, feature_dim)

            x_dict['vals'] = x_x_dense
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        return self.predictor(x_dict['vals']).squeeze()

    def predict_single(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict = self.init_embedding(data)

        N = batch_dict['_vals'].shape[0]
        feature_dim = x_dict['vals'].shape[-1]

        for i, layer in enumerate(self.gcns):
            x_x_dense = x_dict['vals'].reshape(1, N, N, feature_dim)
            x_x_dense = self.higher_orders[i](x_x_dense, None)

            x_dict['vals'] = x_x_dense.reshape(N * N, feature_dim)
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        return self.predictor(x_dict['vals']).squeeze()
