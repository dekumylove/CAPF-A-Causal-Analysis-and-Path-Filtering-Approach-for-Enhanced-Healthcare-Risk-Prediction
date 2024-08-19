import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter
from model.lstm import LSTM_Model
from model.Dipole import Dip_c, Dip_g, Dip_l
from model.retain import Retain
from model.graphcare import GraphCare
from model.medpath import MedPath
from model.PKGAT import GATModel
from model.stageaware import StageAware


class DiseasePredModel(nn.Module):
    def __init__(self, model_type: str, input_dim, output_dim, hidden_dim, embed_dim, num_path, threshold, dropout, gamma_GraphCare, lambda_HAR, bi_direction=False, device=torch.device("cuda")):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.bi_direction = bi_direction
        self.device = device

        self.Wlstm = nn.Linear(output_dim, output_dim, bias=False)
        if model_type == "Dip_l":
            self.dipole = Dip_l(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif model_type == "Dip_c":
            self.dipole = Dip_c(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                max_timesteps=10,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif model_type == "Dip_g":
            self.dipole = Dip_g(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif model_type == "Retain":
            self.dipole = Retain(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                device=self.device
            )
        elif model_type == "LSTM": # LSTM
            self.dipole = LSTM_Model(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                activation="sigmoid",
            )
        else:
            self.dipole = Dip_g(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )

        self.Wkg = nn.Linear(output_dim, output_dim, bias=False)

        if model_type == 'GraphCare':
            self.pkgat = GraphCare(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                dropout=dropout,
                gamma=gamma_GraphCare
            )
        elif model_type == 'MedPath':
            self.pkgat = MedPath(
                nfeat=self.input_dim,
                nemb=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                gat_hid=self.output_dim,
                gat_layers=1
            )
        elif model_type == "StageAware":
            self.pkgat = StageAware(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                Lambda=lambda_HAR,
                device=self.device
            )
        else:
            self.pkgat = GATModel(
                nfeat=input_dim,
                nemb=self.embed_dim,
                gat_layers=1,
                gat_hid=self.output_dim,
                dropout=0.1,
                num_path=num_path,
                threshold=threshold,
                device=self.device
            )

        self.out_linear = nn.Linear(output_dim, output_dim, bias=False)
        self.out_activation = nn.Sigmoid()

    def forward(self, feature_index, rel_index, feat_index, only_dipole, p):
        lstm_out, h_time = self.dipole(feature_index)
        if only_dipole == True:
            return self.out_activation(lstm_out)
        else:
            lstm_out = self.Wlstm(lstm_out)
            h_f, h_c, h_t, path_attentions, causal_attentions = self.pkgat(feature_index, rel_index, feat_index, h_time)
            kg_out_f = self.Wkg(h_f)
            kg_out_c = self.Wkg(h_c)
            kg_out_t = self.Wkg(h_t)

            final_lstm = p * lstm_out
            final_f = final_lstm + (1 - p) * kg_out_f
            final_c = final_lstm + (1 - p) * kg_out_c
            final_t = final_lstm + (1 - p) * kg_out_t
            final_f = self.out_activation(self.out_linear(final_f))
            final_c = self.out_activation(self.out_linear(final_c))
            final_t = self.out_activation(self.out_linear(final_t))

            return final_f, final_c, final_t, path_attentions, causal_attentions
        
    def interpret(self, feature_index, rel_index, only_dipole, p, top_k=2):
        _, _, _, path_attentions, causal_attentions = self.forward(
            feature_index, rel_index, only_dipole, p
        )
        path_attentions = torch.einsum('', causal_attentions)

        visit_top_attn = []
        visit_top_index = []
        batch_bottom_pathattn = []
        batch_bottom_pathindex = []

        for layer_idx, path_attention in enumerate(path_attentions):
            causal_attention = causal_attentions[layer_idx]
            path_attention = torch.einsum('bvf, bvffp -> bvffp', causal_attention, path_attention)

            for batch_idx in range(path_attention.shape[0]):
                for visit_idx in range(path_attention.shape[1]):
                    visit_path_attn = path_attention[batch_idx, visit_idx, :, :, :]
                    visit_path_attn = visit_path_attn.view(visit_path_attn.shape[0], visit_path_attn.shape[1], -1)
                    top_path = torch.topk(visit_path_attn, top_k)
                    bottom_path = torch.topk(visit_path_attn, top_k, largest = False)
                    top_attn = top_path.values
                    top_index = top_path.indices
                    bottom_attn = bottom_path.values
                    bottom_index = bottom_path.indices
                    
                    visit_top_attn.append(top_attn)
                    visit_top_index.append(top_index)
                    batch_bottom_pathattn.append(bottom_attn)
                    batch_bottom_pathindex.append(bottom_index)

        return visit_top_attn, visit_top_index, batch_bottom_pathattn, batch_bottom_pathindex