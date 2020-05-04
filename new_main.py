import torch

from models import GAT
from main import parse_args


args = parse_args()
gat = GAT(
    seed=1,
    nn_args=args.__dict__,
    optim_args=dict(
        lr=args.lr,
        weight_decay_conv=args.weight_decay_conv
    ),
)
if torch.cuda.is_available():
    gat.to_cuda()

gat.train_n_epochs(12)
