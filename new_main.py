import torch

from models import GAT, ConvDecoder
from main import parse_args

args = parse_args()
gat = GAT(
    seed=args.seed,
    nn_args=dict(args=GAT.adapt_args(True, args)),
    optim_args=dict(),
)
gat.set_save_valid_conditions('save', 'every', 600, 'epochs')
gat.train_n_epochs(args.epochs_gat)
gat.load_checkpoint(max_epochs=args.epochs_gat)  # line should be unnecessary once using latest ptutils

conv = ConvDecoder(
    seed=args.seed,
    nn_args=dict(
        args=ConvDecoder.adapt_args(False, args),
        entity_embeddings=gat.final_entity_embeddings,
        relation_embeddings=gat.final_relation_embeddings,
    ),
    optim_args=dict(),
    extra_things_to_use_in_hash=gat.get_path(gat.epochs),
)
conv.set_save_valid_conditions('save', 'every', 10, 'epochs')
conv.train_n_epochs(args.epochs_conv)

# fuck it
conv.conv.n_samples = 5
conv.eval()
with torch.no_grad():
    corpus = conv.train_loader.corpus.corpus
    print(corpus.batch_size)
    corpus.get_validation_pred(
        args, conv.conv, corpus.unique_entities_train
    )
