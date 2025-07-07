from .ppgn import PPGN
from .ign import IGN
from .mpnn import MPNN
from .two_wl import TwoWL


def get_model(args):
    if args.type == 'mpnn':
        model = MPNN(hid_dim=args.hidden,
                     num_encode_layers=args.num_encode_layers,
                     num_conv_layers=args.num_conv_layers,
                     num_pred_layers=args.num_pred_layers,
                     num_mlp_layers=args.num_mlp_layers,
                     norm=args.norm,
                     act=args.act,
                     force_psd=args.force_psd)
    elif args.type == 'ign':
        model = IGN(hid_dim=args.hidden,
                    num_encode_layers=args.num_encode_layers,
                    num_conv_layers=args.num_conv_layers,
                    num_pred_layers=args.num_pred_layers,
                    num_mlp_layers=args.num_mlp_layers,
                    norm=args.norm,
                    act=args.act,
                    force_psd=args.force_psd)
    elif args.type == 'ppgn':
        model = PPGN(hid_dim=args.hidden,
                     num_encode_layers=args.num_encode_layers,
                     num_conv_layers=args.num_conv_layers,
                     num_pred_layers=args.num_pred_layers,
                     num_mlp_layers=args.num_mlp_layers,
                     block_mlp_layers=args.block_mlp_layers,
                     norm=args.norm,
                     act=args.act,
                     force_psd=args.force_psd)
    elif args.type == '2wl':
        model = TwoWL(hid_dim=args.hidden,
                      num_encode_layers=args.num_encode_layers,
                      num_conv_layers=args.num_conv_layers,
                      num_pred_layers=args.num_pred_layers,
                      num_mlp_layers=args.num_mlp_layers,
                      block_mlp_layers=args.block_mlp_layers,
                      norm=args.norm,
                      act=args.act,
                      force_psd=args.force_psd)
    else:
        raise NotImplementedError

    return model
