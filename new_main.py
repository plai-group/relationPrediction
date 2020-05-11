import torch

from models import GAT, ConvDecoder
from main import parse_args


args = parse_args()
gat = GAT(
    seed=1,
    nn_args=dict(args=args),
    optim_args=dict(),
)
gat.train_n_epochs(gat.args.epochs)

conv = ConvDecoder(
    seed=1,
    nn_args=dict(
        args=args,
        entity_embeddings=gat.final_entity_embeddings,
        relation_embeddings=gat.final_relation_embeddings,
    ),
    optim_args=dict(),
)
conv.train_n_epochs(conv.args.epochs)

# fuck it
conv.eval()
with torch.no_grad():
    corpus = conv.train_loader.corpus.corpus
    print(corpus.batch_size)
    corpus.get_validation_pred(
        args, conv.conv, corpus.unique_entities_train
    )
