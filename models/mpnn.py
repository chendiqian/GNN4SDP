import torch
from torch_geometric.utils import to_dense_batch

from models.hetero_base_nn import BaseModel


class MPNN(BaseModel):
    def __init__(self,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm,
                 act,
                 force_psd):
        super().__init__(hid_dim,
                         num_encode_layers,
                         num_conv_layers,
                         num_pred_layers,
                         num_mlp_layers,
                         norm,
                         act,
                         force_psd)

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict = self.init_embedding(data)

        for i, layer in enumerate(self.gcns):
            # now we do message passing
            x_dict = layer(x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        if self.ign_2to1 is not None:
            x = x_dict['vals']
            _, real_x_mask = to_dense_batch(x_dict['vals'].new_empty(batch_dict['_vals'].shape[0]),
                                            batch_dict['_vals'])  # B x Nmax x F
            real_x_x_mask = torch.einsum('bn,bm->bnm', real_x_mask, real_x_mask)  # B x Nmax x Nmax
            x_x_dense = torch.zeros(*real_x_x_mask.shape + (x.shape[-1],), device=x.device, dtype=torch.float)
            x_x_dense[real_x_x_mask] = x

            x_dense = self.ign_2to1(x_x_dense)
            x_dense = self.predictor(x_dense).squeeze()
            x_x_dense = torch.einsum('bn,bm->bnm', x_dense, x_dense)
            x_x_dense = x_x_dense[real_x_x_mask]
            return x_x_dense
        else:
            return self.predictor(x_dict['vals']).squeeze()
