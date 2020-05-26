import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import math
from layers import SpGraphAttentionLayer, ConvKB
from create_batch import CorpusDataset, get_loaders
from argparse import Namespace

import ptutils as ptu

CUDA = torch.cuda.is_available()  # checking cuda availability


# begin my additions (some copied from other files) -------------------------------

class ConvOrGAT(ptu.CudaCompatibleMixin, ptu.HasDataloaderMixin, ptu.Trainable):

    @staticmethod
    def adapt_args(is_gat, args):
        """
        adapts arguments to select either conv args or gat args, depending on self.is_gat
        """
        new = {}
        for field, value in args.__dict__.items():
            last = field.split('_')[-1].lower()
            rest = '_'.join(field.split('_')[:-1])
            to_use = 'gat' if is_gat else 'conv'
            if last not in ['gat', 'conv']:
                new[field] = value
            elif last == to_use:
                new[rest] = value
            else:
                pass
        new.pop('epochs')  # artifact doesn't use this
        return Namespace(**new)

    def get_optim_state(self):
        return {'optim': self.optim.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()}

    def set_optim_state(self, state):
        self.optim.load_state_dict(state['optim'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

    def init_optim(self):
        # weight_decay = self.args.weight_decay_gat if self.is_gat else self.args.weight_decay_conv
        step_size = 500 if self.is_gat else 25
        self.optim = torch.optim.Adam(
            self.parameters(), lr=self.args.lr,
            weight_decay=self.args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=step_size, gamma=0.5, last_epoch=-1)

    def end_epoch(self):
        self.lr_scheduler.step()
        super().end_epoch()

    def gat_or_conv_init(self, args):

        self.args = args
        self.corpus = CorpusDataset(self.args)
        self.current_batch_2hop_indices = self.corpus.get_current_batch_2hop_indices()

        train_loader, valid_loader, test_loader = get_loaders(self.corpus)
        self.set_dataloaders(train_loader, valid_loader, test_loader)

    def post_init(self):

        if torch.cuda.is_available():
            self.to_cuda()


class GAT(ConvOrGAT):
    is_gat = True

    def init_nn(self, args):

        # args = Namespace(**args)
        self.gat_or_conv_init(args)

        init_ent_emb, init_rel_emb = self.corpus.get_pretrained_embs()
        self.spgat = SpKBGATModified(
            initial_entity_emb=init_ent_emb,
            initial_relation_emb=init_rel_emb,
            entity_out_dim=self.args.entity_out_dim,
            relation_out_dim=self.args.entity_out_dim,
            drop_GAT=self.args.drop,
            alpha=self.args.alpha,
            nheads_GAT=self.args.nheads,
        )

    def loss(self, train_indices, train_values):

        entity_embed, relation_embed = self.spgat(
            self.corpus, self.corpus.train_adj_matrix, train_indices, self.current_batch_2hop_indices)
        loss = self.loss_from_embeddings(train_indices, entity_embed, relation_embed)
        self.log = {'loss': loss.item()}
        self.tqdm_text = str(self.log)
        return loss

    def loss_from_embeddings(self, train_indices, entity_embed, relation_embed):

        len_pos_triples = int(
            train_indices.shape[0] / (int(self.args.valid_invalid_ratio) + 1))
        pos_triples = train_indices[:len_pos_triples].repeat(int(self.args.valid_invalid_ratio), 1)
        neg_triples = train_indices[len_pos_triples:]
        def get_norm(triples):
            source_embeds = entity_embed[triples[:, 0]]
            relation_embeds = relation_embed[triples[:, 1]]
            tail_embeds = entity_embed[triples[:, 2]]
            x = source_embeds + relation_embeds - tail_embeds
            return torch.norm(x, p=1, dim=1)
        pos_norm = get_norm(pos_triples)
        neg_norm = get_norm(neg_triples)
        y = torch.ones(int(self.args.valid_invalid_ratio) * len_pos_triples).to(self.device)
        loss_func = nn.MarginRankingLoss(margin=self.args.margin)
        loss = loss_func(pos_norm, neg_norm, y)
        return loss

    @property
    def final_entity_embeddings(self):
        return self.spgat.final_entity_embeddings

    @property
    def final_relation_embeddings(self):
        return self.spgat.final_relation_embeddings

class ConvDecoder(ConvOrGAT):
    is_gat = False

    def init_nn(self, args, entity_embeddings, relation_embeddings):

        self.gat_or_conv_init(args)
        self.conv = SpKBGATConvOnly(
            final_entity_emb=entity_embeddings,
            final_relation_emb=relation_embeddings,
            entity_out_dim=self.args.entity_out_dim,
            relation_out_dim=self.args.entity_out_dim,
            drop_conv=self.args.drop,
            alpha_conv=self.args.alpha,
            nheads_GAT=self.args.nheads,
            conv_out_channels=self.args.out_channels,
            variational=self.args.variational,
            temperature=self.args.temperature,
            sigma_p=self.args.sigma_p,
        )

    def classifier_loss(self, train_indices, train_values):

        preds = self.conv(
            self.corpus, self.corpus.train_adj_matrix, train_indices)
        return nn.SoftMarginLoss(reduction='sum')(preds.view(-1), train_values.view(-1))

    def prior_logpdf(self):
        entity_embeddings, relation_embeddings = self.conv.get_sampled_embeddings()
        entity_sqr_distance = torch.norm(self.conv.entity_embeddings_from_gat - entity_embeddings, 2)**2
        relation_sqr_distance = torch.norm(self.conv.relation_embeddings_from_gat - relation_embeddings, 2)**2
        sqr_distance = entity_sqr_distance + relation_sqr_distance
        return -0.5 * sqr_distance / self.conv.sigma_p**2

    def temp_entropy_q(self):

        std = torch.cat([self.conv.entity_logstddev.exp(), self.conv.relation_logstddev.exp()], dim=0)
        entropy_q = 0.5 * torch.log(2*math.pi*math.e*std**2).sum()
        return entropy_q * self.conv.temperature

    def begin_epoch(self):

        super().begin_epoch()

    def loss(self, train_indices, train_values):

        likelihood_neg_logpdf = self.classifier_loss(train_indices, train_values)
        if self.conv.variational:
            B = len(train_values)
            D = len(self.corpus.corpus.train_indices)
            data_neg_logpdf = likelihood_neg_logpdf * D/B
            elbo_temp = self.prior_logpdf() - data_neg_logpdf + self.temp_entropy_q()
            loss = -elbo_temp * B/D
        else:
            loss = likelihood_neg_logpdf
        self.log = {'loss': loss.item()}
        self.tqdm_text = str(self.log)
        return loss



# end my additions ------------------------------------------------------------------

class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        edge_embed_nhop = relation_embed[
            edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_nhop = out_relation_1[
            edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        x = F.elu(self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]

        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).to(edge_list.device)
        mask = torch.zeros(self.entity_embeddings.shape[0]).to(edge_list.device)
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(self, final_entity_emb, final_relation_emb, entity_out_dim, relation_out_dim,
                 drop_conv, alpha_conv, nheads_GAT, conv_out_channels, variational,
                 temperature, sigma_p):  # NOTE removed alpha as it doesn't seem to get used
        '''
        Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list
        '''

        super().__init__()

        self.num_nodes = final_entity_emb.shape[0]
        emb_dim = entity_out_dim[0] * nheads_GAT[0]

        # Properties of Relations
        self.num_relation = final_relation_emb.shape[0]
        self.relation_dim = final_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_conv = drop_conv
        # self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.variational = variational
        self.temperature = temperature
        self.sigma_p = sigma_p

        assert final_entity_emb.shape == (self.num_nodes, emb_dim,)
        assert final_relation_emb.shape == (self.num_relation, emb_dim,)

        self.entity_embeddings_from_gat = final_entity_emb.clone()  # requires we always load GAT before initialising this
        self.relation_embeddings_from_gat = final_relation_emb.clone()

        self.final_entity_embeddings_mean = nn.Parameter(final_entity_emb.clone())
        self.final_relation_embeddings_mean = nn.Parameter(final_relation_emb.clone())  # this is learnable more. is this desired?
        if self.variational:
            self.entity_logstddev = nn.Parameter(final_entity_emb*0-2)
            self.relation_logstddev = nn.Parameter(final_relation_emb*0-2)

        self.convKB = ConvKB(emb_dim, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def get_sampled_embeddings(self):
        entity_embeddings = self.final_entity_embeddings_mean
        relation_embeddings = self.final_relation_embeddings_mean
        if self.variational:
            entity_embeddings = entity_embeddings + torch.randn_like(entity_embeddings) * self.entity_logstddev.exp()
            relation_embeddings = relation_embeddings + torch.randn_like(relation_embeddings) * self.relation_logstddev.exp()
        return entity_embeddings, relation_embeddings

    def forward(self, Corpus_, adj, batch_inputs):
        entity_embeddings, relation_embeddings = self.get_sampled_embeddings()
        conv_input = torch.cat((entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):

        def get_probs():
            entity_embeddings, relation_embeddings = self.get_sampled_embeddings()
            conv_input = torch.cat((entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), relation_embeddings[
                batch_inputs[:, 1]].unsqueeze(1), entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
            return torch.sigmoid(self.convKB(conv_input))

        if self.variational:
            if not hasattr(self, 'n_samples'):
                raise Exception('Must set attribute n_samples before testing.')
            return sum(get_probs() for _ in range(self.n_samples)) / self.n_samples
        else:
            return get_probs()
