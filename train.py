import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import DiseasePredDataset
from model.DiseasePredModel import DiseasePredModel
from utils import llprint, get_accuracy

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.__version__)


torch.manual_seed(0)
np.random.seed(0)

saved_path = "../saved_model/"
model_name = "Our_model"
path = os.path.join(saved_path, model_name)
if not os.path.exists(path):
    os.makedirs(path)

# the function used to evalutate time-series method
def evaluate_dipole(eval_model, dataloader, device, only_dipole, p):
    eval_model.eval()
    y_label = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            feature_index, rel_index, feat_index, y = batch
            feature_index, rel_index, feat_index, y = (
                feature_index.cuda(1),
                rel_index.cuda(1),
                feat_index.cuda(1),
                y.cuda(1)
            )
            output = eval_model(
                feature_index, rel_index, feat_index, only_dipole, p
            )

            loss = regularization_loss_dipole(output, y)
            y_label.extend(np.array(y.data.cpu()))
            y_pred.extend(np.array(output.data.cpu()))
            # llprint('\rtest step: {} / {}'.format(idx, len(dataloader)))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = get_accuracy(
        y_label, y_pred
    )

    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean

# the function used to evaluate knowledge-driven method
def evaluate(eval_model, dataloader, device, only_dipole, p, adj):
    eval_model.eval()
    y_label = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            feature_index, rel_index, feat_index, y = batch
            feature_index, rel_index, feat_index, y = (
                feature_index.cuda(1),
                rel_index.cuda(1),
                feat_index.cuda(1),
                y.cuda(1)
            )
            output_f, output_c, output_t, path_attentions, causal_attentions = eval_model(
                feature_index, rel_index, feat_index, only_dipole, p
            )
            # Interpretation
            if args.show_interpretation:
                batch_top_pathattn, batch_top_pathindex, batch_bottom_pathattn, batch_bottom_pathindex = eval_model.interpret(path_attentions, causal_attentions)
                for index, sample in enumerate(batch_top_pathattn):
                    for visit_index, visit in enumerate(sample):
                        print(f"Sample {idx * args.batch_size + index}, Visit {visit_index}:")
                        for feat_i in range(visit.size(0)):
                            top_pi = batch_top_pathindex[index][visit_index][feat_i]
                            for num, i in enumerate(top_pi.tolist()):
                                if i < len(batch_top_pathindex[idx * args.batch_size + index][visit_index][feat_i]):
                                    print(f" Top path {i}: Attention Value = {visit[feat_i][num]}")
                                    print(f" Top path is: {batch_top_pathindex[idx * args.batch_size + index][visit_index][feat_i][i]}")
                            bottom_pi = batch_bottom_pathindex[index][visit_index][feat_i]
                            for num, i in enumerate(bottom_pi.tolist()):
                                if i < len(batch_bottom_pathindex[idx * args.batch_size + index][visit_index][feat_i]):
                                    print(f" Bottom path {i}: Attention Value = {batch_bottom_pathattn[index][visit_index][feat_i][num]}")
                                    print(f" Bottom path is: {batch_bottom_pathindex[idx * args.batch_size + index][visit_index][feat_i][i]}")

            loss = regularization_loss(output_f, output_c, output_t, y, args.lambda1, args.lambda2, adj, eval_model, args.beta, args.batch_size)
            y_label.extend(np.array(y.data.cpu()))
            y_pred.extend(np.array(output_f.data.cpu()))
            # llprint('\rtest step: {} / {}'.format(idx, len(dataloader)))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"\nTest average Loss: {avg_loss}")
    macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = get_accuracy(
        y_label, y_pred
    )

    return macro_auc, micro_auc, precision_mean, recall_mean, f1_mean

# the function used to calculate the time-series method's output loss
def regularization_loss_dipole(output, target):
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(output, target)

    return loss

# the function used to calculate the knowledge-driven method's output loss
def regularization_loss(output_f, output_c, output_t, target, Lambda, adj, model, beta, bs):

    hidden_size = model.hidden_dim
    W = model.dipole.rnn.weight_ih_l0  # (3*hidden_size, input_size)
    W_ir = W[0:hidden_size, :]  # (hidden_size, input_size)
    W_iz = W[hidden_size:2 * hidden_size, :]
    W_in = W[2 * hidden_size:3 * hidden_size, :]
    W_matrix = [W_ir, W_iz, W_in]

    relationship_loss = 0
    # A: (input_size , input_size) 01matrix
    for i in range(W_ir.shape[0]):
        for j in range(i, W_ir.shape[0]):
            if adj[i, j] != 0:
                for weight in W_matrix:
                    relationship_loss += torch.norm(weight[i] - weight[j], 2) ** 2

    ce_loss = nn.CrossEntropyLoss()
    loss1 = ce_loss(output_f, target)
    loss2 = ce_loss(output_c, target)

    kl_loss = nn.KLDivLoss()
    even_target = torch.ones_like(target) / target.size(1)
    loss3 = kl_loss(output_t, even_target)

    total_loss = loss1 + Lambda * loss2 + Lambda * loss3 + (beta / bs) * relationship_loss
    return total_loss

def main(args, features, rel_index, feat_index, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    split_train_point = int(len(features) * 6.7 / 10)
    split_test_point = int(len(features) * 8.7 / 10)

    train_features, train_rel_index, train_feat_index, train_labels = (
        features[:split_train_point],
        rel_index[:split_train_point],
        feat_index[:split_train_point],
        labels[:split_train_point]
    )
    test_features, test_rel_index, test_feat_index, test_labels = (
        features[split_train_point:split_test_point],
        rel_index[split_train_point:split_test_point],
        feat_index[split_train_point:split_test_point],
        labels[split_train_point:split_test_point]
    )
    valid_features, valid_rel_index, valid_feat_index, valid_labels = (
        features[split_test_point:],
        rel_index[split_test_point:],
        feat_index[split_test_point:],
        labels[split_test_point:]
    )

    print("train_features: ", len(train_features), "train_rel_index: ", len(train_rel_index), "train_feat_index: ", len(train_feat_index), "train_labels: ", len(train_labels))
    print("test_features: ", len(test_features), "test_rel_index: ", len(test_rel_index), "test_feat_index: ", len(test_feat_index), "test_labels: ", len(test_labels))
    print("valid_features: ", len(valid_features), "valid_rel_index: ", len(valid_rel_index), "valid_feat_index: ", len(valid_feat_index), "valid_labels: ", len(valid_labels))

    train_data = DiseasePredDataset(train_features, train_rel_index, train_feat_index, train_labels)
    test_data = DiseasePredDataset(test_features, test_rel_index, test_feat_index, test_labels)
    valid_data = DiseasePredDataset(valid_features, valid_rel_index, valid_feat_index, valid_labels)

    with open('../data/mimic-iiii/adjacent_matrix.pkl', 'rb') as f:
        adj = pickle.load(f)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    model = DiseasePredModel(
        model_type=args.model,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=512,
        num_path=args.num_path,
        threshold=args.threshold,
        dropout=args.dropout_ratio,
        gamma_GraphCare=args.gamma_GraphCare,
        lambda_HAR=args.lambda_HAR,
        bi_direction=args.bi_direction,
        device=device,
    )

    epoch = 35
    optimzer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    model = model.cuda(1)

    best_eval_macro_auc = 0
    best_eval_epoch = 0

    best_test_macro_auc = 0
    best_test_epoch = 0
    for i in range(epoch):
        print("\nepoch {} --------------------------".format(i))
        total_loss = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            feature_index, rel_index, feat_index, y = batch
            feature_index, rel_index, feat_index, y = (
                feature_index.cuda(1),
                rel_index.cuda(1),
                feat_index.cuda(1),
                y.cuda(1)
            )
            optimzer.zero_grad()
            if not args.only_dipole:
                output_f, output_c, output_t, _, _ = model(feature_index, rel_index, feat_index, args.only_dipole, args.p)
                loss = regularization_loss(output_f, output_c, output_t, y, args.Lambda, adj, model, args.beta, args.batch_size)
            else:
                output = model(feature_index, rel_index, feat_index, args.only_dipole, args.p)
                loss = regularization_loss_dipole(output, y)

            loss.backward()

            optimzer.step()
            llprint("\rtraining step: {} / {}".format(idx, len(train_loader)))
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {i}, Average Loss: {avg_loss}")

        # eval:
        if not args.only_dipole:
            macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = evaluate(
                model,
                valid_loader,
                device,
                args.only_dipole,
                args.p,
                adj
            )
        else:
            macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = evaluate_dipole(
                model,
                valid_loader,
                device,
                args.only_dipole,
                args.p
            )
        print(
            f"\nValid Result:\n"
            f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}"
            f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}"
        )
        if macro_auc > best_eval_macro_auc:
            best_eval_macro_auc = macro_auc
            best_eval_epoch = i

        # test:
        if not args.only_dipole:
            macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = evaluate(
                model,
                test_loader,
                device,
                args.only_dipole,
                args.p,
                adj
            )
        else:
            macro_auc, micro_auc, precision_mean, recall_mean, f1_mean = evaluate_dipole(
                model,
                test_loader,
                device,
                args.only_dipole,
                args.p
            )
        print(
            f"\nTest Result:\n"
            f"\nmacro_auc:{macro_auc}, micro_auc:{micro_auc}"
            f"\nprecision_mean:{precision_mean}\nrecall_mean:{recall_mean}\nf1_mean:{f1_mean}"
        )

        if macro_auc > best_test_macro_auc:
            best_test_macro_auc = macro_auc
            best_test_epoch = i
        # save_model:
        epoch_name = f"Epoch_{i}.model"
        # with open(os.path.join(saved_path, epoch_name), "wb") as model_file:
        #     print()
        #     # torch.save(model.state_dict(), model_file)


        if i > 10:
            print(
                f"Nowbest Eval Epoch:{best_eval_epoch}, Nowbest_Macro_auc:{best_eval_macro_auc}"
            )
            print(
                f"Nowbest Test Epoch:{best_test_epoch}, Nowbest_Macro_auc:{best_test_macro_auc}"
            )
    print(f"Best Eval Epoch:{best_eval_epoch}, best_Macro_auc:{best_eval_macro_auc}")
    print(f"Best Test Epoch:{best_test_epoch}, best_Macro_auc:{best_test_macro_auc}")


if __name__ == "__main__":
    # torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Dip_g",
        choices=["Dip_l", "Dip_g", "Dip_c", "Retain", "LSTM", "GraphCare", "MedPath", "StageAware"],
        help="model",
    )
    parser.add_argument("--lambda_HAR", type=float, default=0.1, help="the lambda of HAR")
    parser.add_argument("--gamma_GraphCare", type=float, default=0.1, help="the gamma of GraphCare")
    parser.add_argument("--dropout_ratio", type=float, default=0.3, help="the dropout_ratio")
    parser.add_argument("--K", type=int, default=3, help="the path length")
    parser.add_argument("--data_type", type=str, default='mimic-iii', help="the type of data")
    parser.add_argument("--num_path", type=int, default=4, help="the maximum number of the paths of each pair")
    parser.add_argument("--threshold", type=int, default=0.005, help="the threshold of the path")
    parser.add_argument("--input_dim", type=int, default=1992, help="input_dim (feature_size)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden_dim")
    parser.add_argument("--output_dim", type=int, default=80, help="output_dim (disease_size)")
    parser.add_argument("--bi_direction", action="store_true", default=True, help="bi_direction")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight_decay")
    parser.add_argument("--beta", type=float, default=0.0001, help="KG factor in loss")
    parser.add_argument("--p", type=float, default=0.8, help="Proportion of the Left part")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--only_dipole", action="store_true", default=False, help="use only diploe moudle")
    parser.add_argument("--Lambda", type=float, default=0.5, help="Lambda")
    parser.add_argument("--show_interpretation", action="store_true", default=False, help="show significant paths for interpretation")

    args = parser.parse_args()

    # 录入数据
    print("loading dataset...")
    # loading the baseline data
    base_path = os.path.join('../data/', args.data_type)
    if args.model in ["GraphCare", "MedPath", "StageAware"]:
        with open(os.path.join(base_path, 'graphcare_data/feat_index.pkl'), 'rb') as f:
            feat_index = pickle.load(f)
        with open(os.path.join(base_path, 'graphcare_data/rel_index.pkl'), 'rb') as f:
            rel_index = pickle.load(f)
        with open(os.path.join(base_path, 'graphcare_data/neighbor_index.pkl'), 'rb') as f:
            neighbor_index = pickle.load(f)
    # loading the CAPF data
    else:
        rel_index = torch.load(os.path.join(base_path, f'rel_index_{args.K}.pt'))
        feat_index = torch.load(os.path.join(base_path, f'feat_index_{args.K}.pt'))

    features = torch.load(os.path.join(base_path, 'features_one_hot.pt'))
    labels = torch.load(os.path.join(base_path, 'new_label_one_hot.pt')) # every element in list : [100]

    print(
        f"features_len:{len(features)}, rel_index_len:{len(rel_index)}, feat_index_len:{len(feat_index)}, labels_len:{len(labels)}"
    )

    main(args, features, rel_index, feat_index, labels)