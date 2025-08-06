from .ppgn import PPGN
from .ign import IGN
from .hetero_base_nn import MPNN
from .two_wl import TwoWL
from .edge_gt import EdgeGT


def get_model(args):
    if args.type == 'mpnn':
        model = MPNN(hid_dim=args.hidden,
                     num_encode_layers=args.num_encode_layers,
                     num_gnn_layers=args.num_gnn_layers,
                     num_pred_layers=args.num_pred_layers,
                     num_mlp_layers=args.num_mlp_layers,
                     norm=args.norm,
                     act=args.act)
    elif args.type == 'ign':
        model = IGN(hid_dim=args.hidden,
                    num_encode_layers=args.num_encode_layers,
                    num_gnn_layers=args.num_gnn_layers,
                    num_conv_layers=args.num_conv_layers,
                    num_pred_layers=args.num_pred_layers,
                    num_mlp_layers=args.num_mlp_layers,
                    norm=args.norm,
                    act=args.act)
    elif args.type == 'ppgn':
        model = PPGN(hid_dim=args.hidden,
                     num_encode_layers=args.num_encode_layers,
                     num_gnn_layers=args.num_gnn_layers,
                     num_conv_layers=args.num_conv_layers,
                     num_pred_layers=args.num_pred_layers,
                     num_mlp_layers=args.num_mlp_layers,
                     block_mlp_layers=args.block_mlp_layers,
                     layernorm=args.layernorm,
                     norm=args.norm,
                     act=args.act)
    elif args.type == '2wl':
        model = TwoWL(hid_dim=args.hidden,
                      num_encode_layers=args.num_encode_layers,
                      num_gnn_layers=args.num_gnn_layers,
                      num_conv_layers=args.num_conv_layers,
                      num_pred_layers=args.num_pred_layers,
                      num_mlp_layers=args.num_mlp_layers,
                      block_mlp_layers=args.block_mlp_layers,
                      norm=args.norm,
                      act=args.act)
    elif args.type == 'edge_gt':
        model = EdgeGT(
            hid_dim=args.hidden,
            num_encode_layers=args.num_encode_layers,
            num_gnn_layers=args.num_gnn_layers,
            num_conv_layers=args.num_conv_layers,
            num_pred_layers=args.num_pred_layers,
            num_mlp_layers=args.num_mlp_layers,
            num_head=args.num_head,
            norm=args.norm,
            act=args.act
        )
    else:
        raise NotImplementedError

    return model
