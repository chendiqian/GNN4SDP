from .ppgn import PPGN

def get_model(args):
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
    return model
