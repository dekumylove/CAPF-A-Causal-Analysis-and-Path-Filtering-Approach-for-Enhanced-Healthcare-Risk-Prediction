import torch
from torch.utils.data import TensorDataset

class DiseasePredDataset(TensorDataset):
    def __init__(self, features, rel_index, feat_index, targets):
        super().__init__()
        self.data = features
        self.rel_index = rel_index
        self.feat_index = feat_index
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # x: (seq_length, feature_size)
        # y: (dignoses size)
        features = self.data
        features = features[index]
        features = torch.cat(features, dim=0)
        rel_index = self.rel_index
        rel_index = rel_index[index]

        feat_index = self.feat_index
        feat_index = feat_index[index]

        y = self.targets[index]
        y = y.squeeze(dim=0)
        return features, rel_index, feat_index, y