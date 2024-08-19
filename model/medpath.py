import torch
from torch import nn
import torch.nn.functional as F
import math


class Embedding(nn.Module):
    def __init__(self, num_feat, hidden_dim):
        """
        num_feat: the number of the medical feature
        hidden_dim: embedding size
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_feat, hidden_dim
        )
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        """
        :param x:   (feature_index_num)
        :return:    embeddings B*F*E
        """
        emb = self.embedding(x)

        return emb  # F*E


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        noutfeat,
        alpha,
        dropout=0.3,
        nrelation=12,
        device=torch.device("cuda"),
    ):
        """
        ninfeat: the number of the medical feature
        noutfeat: the number of the target feature
        """
        super().__init__()

        self.W = nn.ParameterList()

        for i in range(nrelation):
            # relation-sepcific weight
            # W_0 is the padding matrix
            if i != 0:
                self.W.append(nn.Parameter(torch.rand(size=(hidden_dim, hidden_dim))))
                
                nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            else:
                self.W.append(nn.Parameter(torch.ones(size=(hidden_dim, hidden_dim)), requires_grad = False))

        # 线性层a
        self.linear = nn.Linear(2 * noutfeat, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.device = device
        self.num_rel = nrelation
        self.hidden_dim = hidden_dim
        self.Linear_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.Linear_i = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.Linear_y = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.Linear = nn.Linear(hidden_dim + noutfeat, 1)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, noutfeat),
            nn.ReLU()
        )
        # self.M_l = nn.Parameter(size=(m, self.num_feat))

    def forward(self, source_embed, rel_index, s):
        #get W
        params = [param for param in self.W]
        params = torch.stack(params, dim=0)
        params = params.view(self.num_rel, -1)
        bs = rel_index.size(0)
        num_visit = rel_index.size(1)
        max_feat = rel_index.size(2)
        max_target = rel_index.size(3)
        num_path = rel_index.size(4)
        num_rel = self.num_rel
        K = rel_index.size(5)
        W = []
        # message calculation
        for bs in range(bs):
            batch_W = []
            for nv in range(num_visit):
                visit_W = []
                for mf in range(max_feat):
                    rels = rel_index[bs][nv][mf]
                    rels = rels.view(-1, num_rel)
                    rels = torch.mm(rels, params)
                    rels = rels.view(max_target, num_path, K, self.hidden_dim, self.hidden_dim)
                    rels = torch.prod(rels, dim=-3)
                    visit_W.append(rels)
                visit_W = torch.stack(visit_W, dim=0)
                batch_W.append(visit_W)
            batch_W = torch.stack(batch_W, dim=0)
            W.append(batch_W)
        W = torch.stack(W, dim=0)
        W = W.to(self.device)

        z_j = W.permute(0, 1, 3, 2, 4, 5, 6)
        z_j = torch.einsum('bvtfpdd, bvtfd -> bvtfpd', z_j, source_embed)
        s = s.unsqueeze(dim = -2).unsqueeze(dim = -2).unsqueeze(dim = -2).repeat(1, 1, max_target, max_feat, num_path, 1)
        attn = self.Linear(torch.cat([s, z_j], dim=-1)).squeeze(dim=-1)
        attn = self.softmax(attn)
        z_j = torch.einsum('bvtfp, bvtfpd -> bvtfd', attn, z_j)
        
        output = torch.sum(self.output_layer(z_j), dim=-2)
        output = torch.sum(output, dim=-2)

        return output


class MedPath(nn.Module):

    def __init__(
        self,
        nfeat,
        nemb,
        hidden_dim,
        gat_layers,
        gat_hid,
        alpha=0.2,
        nrelation=12,
        device=torch.device("cuda")
    ):
        super().__init__()

        self.embedding = Embedding(nfeat, nemb)
        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        self.nrelation = nrelation
        self.device = device
        self.hidden_dim = hidden_dim
        for _ in range(gat_layers):
            self.gats.append(
                GraphAttentionLayer(
                    hidden_dim=hidden_dim,
                    noutfeat=gat_hid,
                    alpha=alpha,
                    nrelation=self.nrelation,
                    device=self.device,
                    hidden_dim=self.hidden_dim
                )
            )
            ninfeat = gat_hid
        self.rnn = nn.GRU(nfeat, gat_hid, 1, bidirectional=True)

    def forward(self, neighbor_index, rel_index, h_time):
        # 1. embedding source
        embeds = self.embedding.embedding.weight
        source_embed = torch.einsum('bvtmn, nd -> bvtmd', neighbor_index.permute(0, 1, 3, 2, 4), embeds)

        # 2. Graph Attention Layers
        for l in range(self.gat_layers):
            # print(f"----------------No.{l}.gat_layers-----------------------")
            output = self.gats[l](
                source_embed=source_embed,
                rel_index=rel_index,
                s=h_time
            )  # tensor(bs, num_visit, output_dim)

            output = torch.mean(output, dim=1)

        return output