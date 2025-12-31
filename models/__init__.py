from .ppgn import PPGN
from .ign import IGN
from .two_wl import TwoWL
from .delta_two_wl import DeltaTwoWL
from .edge_gt import EdgeGT
from .hetero_higher_order import HigherOrder


def get_model(args):
    if args.type == 'mpnn':
        model = HigherOrder(
            no_mp=False,
            no_wl=True,
            no_dual=args.no_dual,
            hid_dim=args.hidden,
            num_encode_layers=args.num_encode_layers,
            num_conv_layers=args.num_conv_layers,
            gnn_mlp_layers=args.gnn_mlp_layers,
            num_pred_layers=args.num_pred_layers,
            norm=args.norm,
            act=args.act
        )
    elif args.type == 'ign':
        model = IGN(no_mp=args.no_mp,
                    no_wl=False,
                    no_dual=args.no_dual,
                    hid_dim=args.hidden,
                    num_encode_layers=args.num_encode_layers,
                    num_conv_layers=args.num_conv_layers,
                    gnn_mlp_layers=args.gnn_mlp_layers,
                    num_pred_layers=args.num_pred_layers,
                    norm=args.norm,
                    act=args.act)
    elif args.type == 'ppgn':
        model = PPGN(no_mp=args.no_mp,
                     no_wl=False,
                     no_dual=args.no_dual,
                     hid_dim=args.hidden,
                     num_encode_layers=args.num_encode_layers,
                     num_conv_layers=args.num_conv_layers,
                     gnn_mlp_layers=args.gnn_mlp_layers,
                     num_pred_layers=args.num_pred_layers,
                     block_mlp_layers=args.block_mlp_layers,
                     layernorm=args.layernorm,
                     norm=args.norm,
                     act=args.act)
    elif args.type == '2wl':
        model = TwoWL(no_mp=args.no_mp,
                      no_wl=False,
                      no_dual=args.no_dual,
                      hid_dim=args.hidden,
                      num_encode_layers=args.num_encode_layers,
                      num_conv_layers=args.num_conv_layers,
                      gnn_mlp_layers=args.gnn_mlp_layers,
                      num_pred_layers=args.num_pred_layers,
                      block_mlp_layers=args.block_mlp_layers,
                      norm=args.norm,
                      act=args.act)
    elif args.type == 'delta_2wl':
        model = DeltaTwoWL(no_mp=args.no_mp,
                           no_wl=False,
                           no_dual=args.no_dual,
                           hid_dim=args.hidden,
                           num_encode_layers=args.num_encode_layers,
                           num_conv_layers=args.num_conv_layers,
                           gnn_mlp_layers=args.gnn_mlp_layers,
                           num_pred_layers=args.num_pred_layers,
                           block_mlp_layers=args.block_mlp_layers,
                           norm=args.norm,
                           act=args.act)
    elif args.type == 'edge_gt':
        model = EdgeGT(no_mp=args.no_mp,
                       no_wl=False,
                       no_dual=args.no_dual,
                       hid_dim=args.hidden,
                       num_encode_layers=args.num_encode_layers,
                       num_conv_layers=args.num_conv_layers,
                       gnn_mlp_layers=args.gnn_mlp_layers,
                       num_pred_layers=args.num_pred_layers,
                       num_head=args.num_head,
                       norm=args.norm,
                       act=args.act)
    else:
        raise NotImplementedError

    return model
