import pickle
import torch
from tqdm import tqdm
import pandas as pd

with open('code_map_10.pkl', 'rb') as f:
    code_map = pickle.load(f)
#----------------------------------------------------------------------load adjacent matrix and convert matrix to adjacent list----------------------------------------------------------------------------
def generate_adjacent_list():
    adj_list = {}
    with open('adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    for i in tqdm(range(adj.shape[0]), total = adj.shape[0], desc = 'generating adjacent list'):
        adj_list[i] = {}
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                adj_list[i][j] = int(adj[i][j].item())
    print(len(adj_list))
    with open('adjacent_list.pkl', 'wb') as e:
        pickle.dump(adj_list, e)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------generate path using the adjacent list-----------------------------------------------------------------------------------------------
def generate_paths():
    K = 3 #path's length
    features = torch.load('features_one_hot.pt')
    with open('adjacent_list.pkl', 'rb') as f:
        adj_list = pickle.load(f)

    top = pd.read_csv('top_diagnoses.csv')
    icd = top['Diagnosis']
    target_idx = []
    for i in icd:
        for k, v in code_map.items():
            key = k.split('_')[0]
            if i == key:
                target_idx.append(v)
                break

    def find_all_paths(start_idx, path=[]):
        path = path + [start_idx]
        paths = []
        if start_idx in target_idx:
            if len(path) == 1:
                paths.append(path)
            else:
                return [path]
        if start_idx not in adj_list or len(path) >= K:
            return []
        for node, rel in adj_list[start_idx].items():
            if node not in path:
                new_paths = find_all_paths(node, path)
                for p in new_paths:
                    paths.append(p)
                    if len(paths) > 3:
                        return paths
        return paths

    paths = []
    for sample in tqdm(features, total = len(features), desc = 'generating paths'):
        sample_paths = []
        for visit in sample:
            visit_paths = {}
            for i in range(visit.shape[1]):
                if visit[0][i] != 0:
                    all_paths = []
                    all_paths = find_all_paths(i)
                    visit_paths[i] = all_paths
                
            sample_paths.append(visit_paths)
        paths.append(sample_paths)
    with open(f'paths_{K}.pkl', 'wb') as e:
        pickle.dump(paths, e)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------generate relation index-----------------------------------------------------------------------------------------------
def generate_rel_index(num_feat = 1992, max_feat = 12, num_rel = 12, K = 2, num_path = 4, max_target = 10):
    with open('paths_3.pkl', 'rb') as f:
        paths3 = pickle.load(f)
    with open('adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    rel_index = []
    feat_index = []
    paths = []
    for sample in tqdm(paths3, total = len(paths3)):
        sample_rel = []
        sample_feat = []
        sample_path = []
        for visit in sample:
            visit_rel = torch.zeros(size=(max_feat, max_target, num_path, K, num_rel))
            visit_feat = []
            real_feat = torch.zeros(size=(max_feat, num_feat))
            visit_path = [[] for i in range(max_feat)]
            for k, v in visit.items():
                ok = k
                if k not in visit_feat:
                    if len(visit_feat) >= max_feat:
                        break
                    visit_feat.append(k)
                k = visit_feat.index(k)
                real_feat[k][ok] = 1
                target_ = []
                for path_idx, path in enumerate(v):
                    if path_idx < num_path:
                        target = path[-1]
                        if target not in target_:
                            target_.append(target)
                            if len(target_) > max_target:
                                break
                        target = target_.index(target)
                        for path_index in range(num_path):
                            slice_tensor = visit_rel[:, :, path_index, :, :]
                            if torch.sum(slice_tensor).item() == 0:
                                visit_path[k].append(path)
                                for i in range(len(path) - 1):
                                    visit_rel[k][target][path_index][i][int(adj[path[i]][path[i+1]])] = 1
                                if len(path) - 1 < K:
                                    for j in range(K - len(path) + 1):
                                        visit_rel[k][target][path_index][len(path) - 1 + j][0] = 1
                                break
                    else:
                        break
            sample_feat.append(real_feat)
            sample_rel.append(visit_rel)
            sample_path.append(visit_path)
        sample_feat = torch.stack(sample_feat, dim=0)
        feat_index.append(sample_feat)
        sample_rel = torch.stack(sample_rel, dim=0)
        rel_index.append(sample_rel)
        paths.append(sample_path)
    torch.save(rel_index, f'rel_index_{3}.pt')
    torch.save(feat_index, f'feat_index_{3}.pt')
                    
                   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------filter label that is imbalanced-----------------------------------------------------------------------------------------------
def filter_label():
    labels = torch.load('label_one_hot.pt')
    labels_C = torch.cat(labels, dim = 0)
    print(labels_C.size())
    labels_C = torch.sum(labels_C, dim = 0)
    print(labels_C.size())
    values, indices = torch.topk(labels_C, 10, largest=False)
    filter_idx = indices.tolist()
    new_labels = []
    for l in labels:
        mask = torch.ones(l.size(1), dtype=torch.bool)
        mask[filter_idx] = False
        filtered_l = l[0][mask]
        new_labels.append(filtered_l)
    print(new_labels[0].size())
    torch.save(new_labels, 'new_label_one_hot.pt')
                                     
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------generate graphcare, har, medpath data -----------------------------------------------------------------------------------------------
def generate_knowledge_driven_data(max_feat = 32, num_feat = 1992, max_target = 8, num_rel = 11):
    with open('adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)
    features = torch.load('features_one_hot.pt')

    # generate feat_index
    feat_index = []
    rel_index = []
    neighbor_index = []
    for sample in tqdm(features, total=len(features), desc="generating feat_index"):
        sample_f_index = []
        sample_n_index = []
        sample_r_index = []
        for visit in sample:
            visit_f_index = torch.zeros(size=(max_feat, num_feat))
            visit_n_index = torch.zeros(size=(max_feat, max_target, num_feat))
            visit_r_index = torch.zeros(size=(max_feat, max_target, num_rel))
            feat_num = 0
            for i in range(visit.shape[1]):
                if visit[0][i] != 0:
                    visit_f_index[feat_num][i] = 1
                    target_num = 0
                    for j in range(adj.shape[1]):
                        if adj[i][j] != 0:
                            visit_n_index[feat_num][target_num][j] = 1
                            visit_r_index[feat_num][target_num][int(adj[i][j]) - 1] = 1
                            target_num += 1
                            if target_num >= max_target:
                                break
                    feat_num += 1
                    if feat_num >= max_feat:
                        break
            sample_f_index.append(visit_f_index)
            sample_n_index.append(visit_n_index)
            sample_r_index.append(visit_r_index)
        sample_f_index = torch.stack(sample_f_index, dim=0)
        sample_n_index = torch.stack(sample_n_index, dim=0)
        sample_r_index = torch.stack(sample_r_index, dim=0)
        feat_index.append(sample_f_index)
        rel_index.append(sample_r_index)
        neighbor_index.append(sample_n_index)
    print(f'feat_index:{len(feat_index)}')
    print(f'rel_index:{len(rel_index)}')
    print(f'neighbor_index:{len(neighbor_index)}')
    with open('feat_index.pkl', 'wb') as f:
        pickle.dump(feat_index, f)
    with open('rel_index.pkl', 'wb') as f:
        pickle.dump(rel_index, f)
    with open('neighbor_index.pkl', 'wb') as f:
        pickle.dump(neighbor_index, f)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------generate_drugrecommendation_data-----------------------------------------------------------------------------------------------
def generate_drugrecommendation_data():
    features = torch.load('features_one_hot.pt')
    drug_count = {}
    for sample in features:
        for visit in sample:
            for i in range(774):
                if visit[0][1218+i] != 0:
                    k = 1218+i
                    if k not in drug_count:
                        drug_count[k] = 1
                    else:
                        drug_count[k] += 1
    sorted_items = sorted(drug_count.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:80]
    keys = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    drugrecommendation_features_one_hot = []
    drugrecommendation_label_one_hot = []
    for sample in tqdm(features, total=len(features), desc="generating drugrecommendation data"):
        sample_feature = []
        label = torch.zeros(size=(1, 80))
        for visit_index, visit in enumerate(sample):
            if visit_index == len(sample) - 1:
                visit_feature = visit.clone()
                for i in range(774):
                    k = i + 1218
                    if visit[0][k] != 0 and k in values:
                        label[0][values.index(k)] = 1
                        visit_feature[0][k] = 0
                sample_feature.append(visit_feature)
            else:
                sample_feature.append(visit)
        drugrecommendation_features_one_hot.append(sample_feature)
        drugrecommendation_label_one_hot.append(label)
    torch.save(drugrecommendation_features_one_hot, 'drugrecommendation_features_one_hot.pt')
    torch.save(drugrecommendation_label_one_hot, 'drugrecommendation_label_one_hot.pt')   

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------