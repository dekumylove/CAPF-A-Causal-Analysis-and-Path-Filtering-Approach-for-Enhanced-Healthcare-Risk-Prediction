import torch
from torch import nn
import torch.nn.functional as F


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
        ninfeat,
        noutfeat,
        dropout,
        alpha,
        hidden_dim,
        num_path,
        threshold,
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
                self.W.append(nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim))))
                
                nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            else:
                self.W.append(nn.Parameter(torch.ones(size=(hidden_dim, hidden_dim)), requires_grad = False))

        self.linear = nn.Linear(2 * noutfeat, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.device = device
        self.num_rel = nrelation
        self.num_path = num_path
        self.hidden_dim = hidden_dim
        self.num_feat = ninfeat
        self.W_p = nn.Parameter(torch.ones(noutfeat + hidden_dim))
        self.b_p = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold
        self.softmax = nn.Softmax()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim + noutfeat, self.num_feat),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.num_feat, 1),
        )

    def forward(self, feature_embed, h_time, rel_index, cosine_similarity):
        # get W
        params = torch.stack([param for param in self.W], dim = 0)
        params = params.view(self.num_rel, -1)
        max_target = rel_index.size(3)
        num_path = rel_index.size(4)
        K = rel_index.size(5)
        # calculate the message transmitted through paths
        W = []
        for bs in range(rel_index.size(0)):
            sample_W = []
            for nv in range(rel_index.size(1)):
                visit_W = []
                for nf in range(rel_index.size(2)):
                    rels = rel_index[bs][nv][nf]
                    rels = rels.view(-1, self.num_rel)
                    rels = torch.mm(rels, params)
                    rels = rels.view(max_target, num_path, K, self.hidden_dim, self.hidden_dim)
                    rels = torch.prod(rels, dim=-3)
                    visit_W.append(rels)
                visit_W = torch.stack(visit_W, dim=0)
                sample_W.append(visit_W)
            sample_W = torch.stack(sample_W, dim=0)
            W.append(sample_W)
        M_j = torch.stack(W, dim=0)
        M_j = M_j.cuda(1)

        # path filtering
        feat_embed_1 = feature_embed.unsqueeze(dim=-2).unsqueeze(dim=-2).repeat(1, 1, 1, rel_index.size(3), self.num_path, 1)
        M_j = torch.einsum('bvftpdd, bvftpd -> bvftpd', M_j, feat_embed_1)
        h_time_1 = h_time.unsqueeze(dim=-2).unsqueeze(dim=-2).unsqueeze(dim=-2).repeat(1, 1, feature_embed.size(2), rel_index.size(3), self.num_path, 1)
        msg = torch.cat([h_time_1, M_j], dim=-1)
        msg = torch.einsum('bvftpa, a -> bvftp', msg, self.W_p) + self.b_p
        a_j = self.sigmoid(msg)
        mask = F.gumbel_softmax(a_j, hard = True)
        msg = torch.einsum('bvftp, bvftp -> bvftp', mask, msg)
        path_attn = self.softmax(msg, dim=-1)
        M_j = torch.einsum('bvftp, bvftpd -> bvftpd', path_attn, M_j)
        M_j = torch.sum(M_j, dim=-2)
        M_j = torch.sum(M_j, dim=-2)

        # joint impact
        final_M_j = torch.einsum('bvff, bvfd -> bvfd', cosine_similarity, M_j)

        # causal attention
        h_time_2 = h_time.unsqueeze(dim=-2).repeat(1, 1, feature_embed.size(2), 1)
        attn_causal = self.MLP(torch.cat([h_time_2, feature_embed], dim=-1)).squeeze(dim=-1)
        attn_causal = F.softmax(attn_causal, dim=-1)
        attn_trivial = 1 - attn_causal

        # get the graph representation
        h_c = torch.einsum('bvf, bvfd -> bvfd', attn_causal, final_M_j)
        h_c = torch.sum(h_c, dim=-2)
        h_t = torch.einsum('bvf, bvfd -> bvfd', attn_trivial, final_M_j)
        h_t = torch.sum(h_t, dim=-2)
        h_f = h_c.clone()

        # random addition to generate h_f
        M = h_c.size(0)
        seq_len = h_c.size(1)
        II = torch.randint(0, seq_len, (M,), device=h_c.device)
        h_f[torch.arange(M), II] += h_t[torch.arange(M), II]

        return h_f, h_c, h_t, path_attn, attn_causal


class GATModel(nn.Module):

    def __init__(
        self,
        nfeat,
        nemb,
        gat_layers,
        gat_hid,
        dropout,
        num_path,
        threshold,
        alpha=0.2,
        nrelation=12,
        device=torch.device("cuda"),
    ):
        super().__init__()

        self.embedding = Embedding(nfeat, nemb)
        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        self.nrelation = nrelation
        self.device = device
        self.hidden_dim = nemb
        ninfeat = nemb
        for _ in range(gat_layers):
            self.gats.append(
                GraphAttentionLayer(
                    ninfeat=ninfeat,
                    noutfeat=gat_hid,
                    dropout=dropout,
                    alpha=alpha,
                    nrelation=self.nrelation,
                    device=self.device,
                    hidden_dim=self.hidden_dim,
                    num_path=num_path,
                    threshold=threshold
                )
            )
            ninfeat = gat_hid
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim, gat_hid),
            nn.ReLU()
        )

    def forward(self, rel_index, feat_index, h_time):

        # instore the path attention for interpretation
        path_attentions = []
        causal_attentions = []
        # 1. embedding feature
        embeds = self.embedding.embedding.weight
        feature_embed = torch.einsum('bvmn, nd -> bvmd', feat_index, embeds)
        # 2. cosine similarity
        normalized_embeddings = F.normalize(feature_embed, p=2, dim=1)
        normalized_embeddings = normalized_embeddings.view(-1, feat_index.size(-2), self.hidden_dim)
        normalized_embeddings_2 = normalized_embeddings.view(-1, self.hidden_dim, feat_index.size(-2))
        cs = torch.bmm(normalized_embeddings, normalized_embeddings_2)
        cs = cs.view(feat_index.size(0), feat_index.size(1), feat_index.size(2), feat_index.size(2))

        # 2. Graph Attention Layers
        for l in range(self.gat_layers):
            # print(f"----------------No.{l}.gat_layers-----------------------")
            h_f, h_c, h_t, path_attn, attn_causal = self.gats[l](
                feature_embed=feature_embed,
                h_time=h_time,
                cosine_similarity=cs,
                rel_index=rel_index
            )  # (batch_size, 6, hidden_dim)

            path_attentions.append(path_attn)
            causal_attentions.append(attn_causal)

            h_f = self.output_layer(h_f)
            h_c = self.output_layer(h_c)
            h_t = self.output_layer(h_t)
            h_f = torch.mean(h_f, dim=1)
            h_c = torch.mean(h_c, dim=1)
            h_t = torch.mean(h_t, dim=1)

        return h_f, h_c, h_t, path_attentions, causal_attentions