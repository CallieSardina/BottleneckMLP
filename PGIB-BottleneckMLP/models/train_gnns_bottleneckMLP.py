import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures_bottleneckMLP import data_args, train_args, model_args, mcts_args
from my_mcts import mcts
from tqdm import tqdm
from proto_join import join_prototypes_by_activations
from utils import PlotUtils
from torch_geometric.utils import to_networkx
from itertools import accumulate
from torch_geometric.datasets import MoleculeNet
import pdb
import random
from utils_edge_and_plots.edge_estimator import EDGE
import torch_scatter
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import subgraph
from copy import deepcopy
from torch_geometric.nn import global_mean_pool
from similarity_metrics import *
from similarity_metrics import LNSA_loss


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def append_record(info, args):
    task = args.task

    f = open(f'./log/hyper_search_{task}.txt', 'a')
    f.write(info)
    f.write('\n')
    f.close()


# train for graph classification
def train_GC(model_type, args):

    task = args.task
    
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    dataloader = get_dataloader(dataset, data_args.dataset_name, train_args.batch_size, data_split_ratio=data_args.data_split_ratio) # train, val, test dataloader 나눔

    print('start training model==================')

    gnnNets = GnnNets(input_dim, output_dim, model_args) 

    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]

    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print("Dataset : ", data_args.dataset_name)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_acc = 0.0
    data_size = len(dataset)

    # HERE 
    best_auroc = 0.0
    best_epoch = -1

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))

    early_stop_count = 0
    data_indices = dataloader['train'].dataset.indices 

    best_acc = 0.0

    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        if epoch >= train_args.proj_epochs and epoch % 50 == 0:
            gnnNets.eval()

            # prototype projection
            for i in range( gnnNets.model.prototype_vectors.shape[0] ): 
                count = 0
                best_similarity = 0
                label = gnnNets.model.prototype_class_identity[0].max(0)[1]
                for j in range(i*10, len(data_indices)): 
                    data = dataset[data_indices[j]] 
                    if data.y == label: 
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i]) 
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= train_args.count:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print('Projection of prototype completed')
                        break


            # prototype merge
            share = True
            if train_args.share: 
                if gnnNets.model.prototype_vectors.shape[0] > round(output_dim * model_args.num_prototypes_per_class * (1-train_args.merge_p)) :  
                    join_info = join_prototypes_by_activations(gnnNets, train_args.proto_percnetile,  dataloader['train'], optimizer)

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for i, batch in enumerate(dataloader['train']):
            if model_args.cont:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, min_distance, topk_node_index, bottomk_node_index, mlp_embeddings = gnnNets(batch)
            else:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, min_distance, topk_node_index, bottomk_node_index, mlp_embeddings = gnnNets(batch) 

            # # HERE 
            # print("active node index: ", active_node_index)
            # # print(tmp)


            # TODO: compute MI between embeddings # HERE
            # for key in embeddings:
            #     print(f"{key} embedding shape: {embeddings[key].shape}")

            if batch.num_graphs < 10:
                continue

            # print("graph emb shape: ", graph_emb.shape)
            # print("y shape: ", batch.y.shape)

            # for name, tensor in embeddings.items():
            #     print(f"{name}: {tensor.shape}")

            # # print(f"batch type: {type(batch)}, batch contents: {batch}")
            # for key in embeddings:
            #     if embeddings[key].shape[0] != batch.y.shape[0]:
            #         embeddings[key] = torch_scatter.scatter(embeddings[key] , batch.batch, dim=0, reduce="mean")  # Shape: [batch_size, emb_dim]
            
            for key in mlp_embeddings:
                print(f"{key} embedding shape: {mlp_embeddings[key].shape}")

            # mi_XZ = [EDGE(embeddings['gnn_layer_0'].cpu().detach().numpy(), embeddings[key].clone().detach().numpy()) for key in embeddings]
            mi_XZ = [EDGE(mlp_embeddings[key].clone().detach().numpy(), mlp_embeddings[key].clone().detach().numpy()) for key in mlp_embeddings]

            mi_ZY = [EDGE(batch.y.cpu().detach().numpy(), mlp_embeddings[key].clone().detach().numpy()) for key in mlp_embeddings]

            with open(f'./MI_logs/{task}.txt', 'a') as f:
                print(f"Epoch {epoch}, MI_XZ: {mi_XZ}, MI_ZY: {mi_ZY}", file=f)

            # HERE add Danish's similarity metrics
            # HERE QUESTION: what about batching?
            # Compute similarity metrics
            # nsa = NSALoss()
            # lnsa = LNSA_loss(k=40)
            # # pick k to be what fraction of the space you reallly care about 1/50 or 1/100 of the data size is usually a good number
            # #unless you data is really large in which case try not to go above k=1000 or 2000


            # space1 = mlp_embeddings['node_embs']
            # space2 = mlp_embeddings['layer_2']

            # print("space1 shape: ", space1.shape)
            # print("space2 shape: ", space2.shape)

            # space1_np = space1.cpu().detach().numpy()
            # space2_np = space2.cpu().detach().numpy()

            # cka_value = 5 #cka(space1_np, space2_np)
            # # space1 = torch.tensor(space1)
            # # space2 = torch.tensor(space2)
            # rtd_value = rtd(space1, space2)
            # nsa_value = nsa(space1, space2) + lnsa(space1,space2)
                
            # values = {"CKA": cka_value,
            #         "RTD": rtd_value,
            #         "NSA+LNSA (k={})".format(k): nsa_value}

            # # Run the pipeline

            # # Print results
            # for metric, value in values.items():
            #     print(f"{metric}: {value}")

            # with open(f'./similarity_logs/{task}.txt', 'a') as f:
            #     print(f"Epoch {epoch}, CKA: {values['CKA']}, RTD: {values['RTD']}, NSA+LNSA: {values['NSA+LNSA (k={})'.format(k)]}", file=f)

            # # HERE debugging -- changed this because I used the processed data from GSAT, so batch.y was formatted differently.
            #loss = criterion(logits, batch.y.squeeze().long())
            batch.y = batch.y.squeeze().long()
            loss = criterion(logits, batch.y)

            if model_args.cont:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(model_args.device) 
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                positive_sim_matrix = sim_matrix * prototypes_of_correct_class
                negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

                contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
                contrastive_loss = - torch.log(contrastive_loss).mean()

            #diversity loss
            prototype_numbers = []
            for i in range(gnnNets.model.prototype_class_identity.shape[1]):
                prototype_numbers.append(int(torch.count_nonzero(gnnNets.model.prototype_class_identity[: ,i])))
            prototype_numbers = accumulate(prototype_numbers)
            n = 0
            ld = 0

            for k in prototype_numbers:    
                p = gnnNets.model.prototype_vectors[n : k]
                n = k
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3 
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device) 
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2)) 

            if model_args.cont:
                loss = loss #+ train_args.alpha2 * contrastive_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss #+ model_args.con_weight*connectivity_loss # HERE + train_args.alpha2 * contrastive_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 
            else:
                loss = loss #+ train_args.alpha2 * prototype_pred_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss #+ model_args.con_weight*connectivity_loss # HERE + train_args.alpha2 * prototype_pred_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 

            with open(f'./for_KL_plot/with_MLP_{task}.txt', 'a') as f:
                print(f"Epoch {epoch}, KL Loss: {KL_Loss}", file=f)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

            # HERE -- aucroc
            # auroc_list = []
            # for i in range(batch.num_graphs):
            #     data = batch[i]
            #     graph = to_networkx(data, to_undirected=True)
            #     true_edge_labels = data.edge_label.cpu().numpy()
            #     num_edges = true_edge_labels.shape[0]
            #     edge_mask = get_edge_mask(graph, active_node_index[i], num_edges)
            #     print("edge mask: ", edge_mask) # edge mask:  [(2, 5), (3, 7), (3, 8), (5, 15), (11, 18)]
            #     pred_edge_scores = edge_mask.int().cpu().numpy()
                
            #     #print("pred edge scores: ", pred_edge_scores)
            #     print("true edge labels: ", true_edge_labels)
            #     print("true edge labels shape: ", true_edge_labels.shape)
            #     try:
            #         auroc = roc_auc_score(true_edge_labels, pred_edge_scores)
            #         auroc_list.append(auroc)
            #     except ValueError:
            #         print("Skipping graph with only one class in ground truth edges.")
            iou_list = []
            for i in range(batch.num_graphs):
                data = batch[i]
                graph = to_networkx(data, to_undirected=True)
                
                # Get true edge labels and number of edges
                true_edge_labels = data.edge_label.cpu().numpy()
                num_edges = true_edge_labels.shape[0]
                
                # Get the edge mask for the active nodes in the subgraph
                nodelist = active_node_index[i] # HERE active_node_index
                if not isinstance(nodelist, list):
                    continue
                edge_mask = get_edge_mask(graph, nodelist, num_edges)
                
                # Convert edge mask to numpy array for AUROC calculation
                pred_edge_scores = edge_mask.int().cpu().numpy()

                intersection = len(set(pred_edge_scores) & set(true_edge_labels))
                union = len(set(pred_edge_scores) | set(true_edge_labels))
                iou = intersection / union
                iou_list.append(iou)
                # print("edge mask: ", edge_mask)  # Example: edge mask:  [1, 1, 1, 0, 0, 0, ...]
                # print("true edge labels: ", true_edge_labels)
                # print("true edge labels shape: ", true_edge_labels.shape)
                # matching_count = (edge_mask.int().cpu().numpy() == true_edge_labels).sum()
                # print(f"Number of times edge mask pred = true: {matching_count}")

                # # Count the number of 0s and 1s in true_edge_labels
                # count_zeros = (true_edge_labels == 0).sum()
                # count_ones = (true_edge_labels == 1).sum()
                # print(f"Count of 0s true_edge_labels: {count_zeros}")
                # print(f"Count of 1s true_edge_labels: {count_ones}")

                # # Count the number of 0s and 1s in pred_edge_scores
                # count_zeros = (pred_edge_scores == 0).sum()
                # count_ones = (pred_edge_scores == 1).sum()
                # print(f"Count of 0s pred_edge_scores: {count_zeros}")
                # print(f"Count of 1s pred_edge_scores: {count_ones}")

                # if len(set(true_edge_labels)) > 1:  # Ensure both 0 and 1 are present
                #     try:
                #         auroc = roc_auc_score(true_edge_labels, edge_mask.int().cpu().numpy())
                #         auroc_list.append(auroc)
                #         # print("Calculating AUROC.")
                #     except ValueError:
                #         print("Error calculating AUROC.")
                # else:
                #     print("Skipping AUROC calculation: only one class present in true labels.")


        # report train msg
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | "
              f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")
        
        # HERE --aucroc
        # Track best AUROC
        if iou_list:
            mean_iou = np.mean(iou_list)
            print(f"Epoch {epoch} | Mean AUROC: {mean_iou:.4f}")
            append_record(f"Epoch {epoch}, mean AUROC: {mean_iou:.4f}", args)

            # if mean_auroc > best_auroc:
            #     best_auroc = mean_auroc
            #     best_epoch = epoch
            #     torch.save(gnnNets.state_dict(), os.path.join(ckpt_dir, 'best_model.pt'))
            #     print(f"Best AUROC updated: {best_auroc:.4f} at epoch {epoch}")
        else:
            print(f"Epoch {epoch} | No valid AUROC scores")

        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), np.concatenate(acc, axis=0).mean()), args)


        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record("Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']), args)

        test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
        print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f} | IoU: {test_state['iou']:.3f} | Fid+: {test_state['fid+']:.3f} | Fid-: {test_state['fid-']:.3f}")           

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        # HERE -- removed early stopping so we can run more epochs for IB
        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best, args)

        # mean_auroc = np.mean(auroc_list) if auroc_list else 0.0
        # print(f"Epoch {epoch}: AUROC = {mean_auroc:.4f}")

        # if mean_auroc > best_auroc:
        #     best_auroc = mean_auroc
        #     best_auroc_epoch = epoch
        #     print(f"New best AUROC: {best_auroc:.4f} at epoch {best_auroc_epoch}")


    print(f"The best validation accuracy is {best_acc}.")

    if iou_list:
        print("Final Mean AUROC across last epoch:", np.mean(iou_list))
    
    # # === After training ends ===
    # print(f"Loading best model from epoch {best_epoch} with AUROC {best_auroc:.4f}")
    # gnnNets.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pt')))
    # gnnNets.eval()

    # # Evaluate on test set
    # test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    # print(f"Final Test (Best AUROC Epoch {best_epoch}): Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    # append_record("Test on best AUROC epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(best_epoch, test_state['loss'], test_state['acc']), args)



    
    # report test msg
    gnnNets = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_{model_type}_{model_args.readout}_best_{task}.pth')) # .to_device()
    gnnNets.to_device()
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    print(f"Test | Dataset: {data_args.dataset_name:s} | model: {model_args.model_name:s}_{model_type:s} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f} | IoU: {test_state['iou']:.3f} | Fid+: {test_state['fid+']:.3f} | Fid-: {test_state['fid-']:.3f}")
    append_record("loss: {:.3f}, acc: {:.3f}, auroc: {:.3f}".format(test_state['loss'], test_state['acc'], test_state['iou']), args)

    return test_state['acc']


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            # HERE 
            batch.y = batch.y.squeeze().long()
            logits, probs, _, _, _, _, _, _, _, _, _ = gnnNets(batch) # HERE , _
            if data_args.dataset_name == 'clintox':
                batch.y = torch.tensor([torch.argmax(i).item() for i in batch.y]).to(model_args.device)
            loss = criterion(logits, batch.y)


            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state


# HERE -- aucroc
def get_edge_mask(graph, nodelist, num_edges):
    # print("nodelist: ", nodelist)
    active_edges = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if n_frm in nodelist and n_to in nodelist]
    
    edge_mask = torch.zeros(num_edges, dtype=torch.int)
    
    for i, (n_frm, n_to) in enumerate(active_edges):
        edge_mask[i] = 1 
    
    return edge_mask


def calc_fidelity(y_true, y_pred, y_pred_removed, y_pred_retained):
    """
    Calculates Fid+ and Fid- for explanation evaluation.

    Args:
        y_true (Tensor): Ground truth labels, shape [n]
        y_pred (Tensor): Predictions on full graphs, shape [n]
        y_pred_removed (Tensor): Predictions after removing explanation, shape [n]
        y_pred_retained (Tensor): Predictions using only the explanation, shape [n]
    
    Returns:
        fid_plus (float), fid_minus (float)
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    y_pred_removed = y_pred_removed.cpu()
    y_pred_retained = y_pred_retained.cpu()

    print("IN FIDELITY FUNC:")
    print("y_true: ", y_true.shape)
    print("y_pred: ", y_pred.shape)
    print("y_pred_removed: ", y_pred_removed.shape)
    print("y_pred_retained: ", y_pred_retained.shape)

    correct_full = (y_true == y_pred).int()
    correct_removed = (y_true == y_pred_removed).int()
    correct_retained = (y_true == y_pred_retained).int()

    fid_plus = torch.mean((correct_full - correct_removed).float()).item()
    fid_minus = torch.mean((correct_full - correct_retained).float()).item()

    return fid_plus, fid_minus


def subgraph_wrapper(data, node_idx):
    node_idx = torch.tensor(node_idx, dtype=torch.long)
    node_idx_set = set(node_idx.tolist())

    # Mapping original node indices to new ones
    new_node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(node_idx_set))}

    # Filter edges where both ends are in node_idx
    src, dst = data.edge_index
    mask = [(u.item() in node_idx_set and v.item() in node_idx_set) for u, v in zip(src, dst)]
    mask_tensor = torch.tensor(mask, dtype=torch.bool)

    filtered_edges = data.edge_index[:, mask_tensor]

    if filtered_edges.size(1) > 0:
        reindexed_src = torch.tensor([new_node_map[u.item()] for u in filtered_edges[0]], dtype=torch.long)
        reindexed_dst = torch.tensor([new_node_map[v.item()] for v in filtered_edges[1]], dtype=torch.long)
        new_edge_index = torch.stack([reindexed_src, reindexed_dst], dim=0)
        new_edge_attr = data.edge_attr[mask_tensor] if data.edge_attr is not None else None
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.empty((0, data.edge_attr.size(1)), dtype=data.edge_attr.dtype) if data.edge_attr is not None else None

    # Slice node features if present

    new_x = data.x[node_idx] if data.x is not None else None

    # Slice node-level labels if present
    new_y = data.y[node_idx] if data.y is not None and data.y.shape[0] == data.num_nodes else data.y

    # Construct new data object
    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        y=new_y
    )

    return new_data

# def subgraph_wrapper(data, node_idx):

#     node_idx = torch.tensor(node_idx, dtype=torch.long)
#     node_idx_set = set(node_idx.tolist())

#     # Mapping original node indices to new ones
#     new_node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(node_idx_set))}

#     # Filter edges where both ends are in node_idx
#     src, dst = data.edge_index
#     mask = [(u.item() in node_idx_set and v.item() in node_idx_set) for u, v in zip(src, dst)]
#     filtered_edges = data.edge_index[:, torch.tensor(mask, dtype=torch.bool)]

#     # Reindex edges
#     reindexed_src = torch.tensor([new_node_map[u.item()] for u in filtered_edges[0]], dtype=torch.long)
#     reindexed_dst = torch.tensor([new_node_map[v.item()] for v in filtered_edges[1]], dtype=torch.long)
#     new_edge_index = torch.stack([reindexed_src, reindexed_dst], dim=0)

#     # Slice node features if present
#     new_x = data.x[node_idx] if data.x is not None else None

#     # Slice edge_attr if present
#     new_edge_attr = data.edge_attr[torch.tensor(mask, dtype=torch.bool)] if data.edge_attr is not None else None

#     # Slice node-level labels if present
#     new_y = data.y[node_idx] if data.y is not None and data.y.shape[0] == data.num_nodes else data.y

#     # Construct new data object
#     new_data = Data(
#         x=new_x,
#         edge_index=new_edge_index,
#         edge_attr=new_edge_attr,
#         y=new_y
#     )

#     return new_data


def build_explanation_subgraphs(batch, node_index):
    retained_graphs = []
    removed_graphs = []

    for i, data in enumerate(batch.to_data_list()):
        data_1 = deepcopy(data)
        data_2 = deepcopy(data)

        important_nodes = node_index[i]
        if not isinstance(important_nodes, list):
            important_nodes = [important_nodes]

        all_nodes = torch.arange(data.num_nodes)
        unimportant_nodes = list(set(all_nodes.tolist()) - set(important_nodes))

        # Handle edge case where there are no unimportant nodes
        if not unimportant_nodes:
            unimportant_nodes = [0]

        # Create subgraphs
        retained_data = subgraph_wrapper(data_1, important_nodes)
        removed_data = subgraph_wrapper(data_2, unimportant_nodes)

        # Create batch indices for each subgraph (0 for retained, 1 for removed)
        retained_batch = torch.full((retained_data.num_nodes,), i, dtype=torch.long)  # Batch ID = i for retained subgraph
        removed_batch = torch.full((removed_data.num_nodes,), i, dtype=torch.long)  # Batch ID = i for removed subgraph

        # Create Batch objects
        retained_data_batch = Data(
            x=retained_data.x,  # Node features for retained nodes
            edge_index=retained_data.edge_index,  # Edge index for retained nodes
            y=retained_data.y,  # Labels for retained nodes
            num_nodes=retained_data.num_nodes,  # Number of nodes in retained subgraph
            batch=retained_batch  # Batch indices for retained subgraph
        )

        removed_data_batch = Data(
            x=removed_data.x,  # Node features for removed nodes
            edge_index=removed_data.edge_index,  # Edge index for removed nodes
            y=removed_data.y,  # Labels for removed nodes
            num_nodes=removed_data.num_nodes,  # Number of nodes in removed subgraph
            batch=removed_batch  # Batch indices for removed subgraph
        )

        # Append to the respective lists
        retained_graphs.append(retained_data_batch)
        removed_graphs.append(removed_data_batch)

    # Check the retained_graphs and removed_graphs to ensure they are populated
    print(f"retained_graphs: {retained_graphs}")
    print(f"removed_graphs: {removed_graphs}")

    # Now, if the lists are empty, return None
    if not retained_graphs or not removed_graphs:
        print("Error: retained_graphs or removed_graphs is empty!")
        return None

    # Convert lists to a single batch of graphs
    retained_batch = Batch.from_data_list(retained_graphs)
    removed_batch = Batch.from_data_list(removed_graphs)


    # Debug prints
    print(f"Final retained batch: {retained_batch}")
    print(f"Final removed batch: {removed_batch}")

    print("Original batch size:", batch.y.shape[0])
    print("Retained batch size:", retained_batch.num_graphs)
    print("Removed batch size:", removed_batch.num_graphs)

    return retained_batch, removed_batch



def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            logits, probs, active_node_index, _, _, _, _, _, topk_node_index, bottomk_node_index, mlp_embeddings = gnnNets(batch) # HERE , _
            # HERE 
            batch.y = batch.y.squeeze().long()
            loss = criterion(logits, batch.y)
            
            # test_subgraph extraction          
            save_dir = os.path.join('./masking_interpretation_results',
                                    f"{mcts_args.dataset_name}_"
                                    f"{model_args.readout}_"
                                    f"{model_args.model_name}_")
            # if not os.path.isdir(save_dir):
            #     os.mkdir(save_dir)
            # plotutils = PlotUtils(dataset_name=data_args.dataset_name)

            # for i, index in enumerate(test_dataloader.dataset.indices[batch_index * train_args.batch_size: (batch_index+1) * train_args.batch_size]):
            #     data = test_dataloader.dataset.dataset[index]
            #     graph = to_networkx(data, to_undirected=True)
            #     if type(active_node_index[i]) == int:
            #         active_node_index[i] = [active_node_index[i]]
            #     # print(active_node_index[i])
            #     plotutils.plot(graph, active_node_index[i], x=data.x,
            #                 figname=os.path.join(save_dir, f"example_{i}.png"))
    
            iou_list = []
            fid_plus_list = []
            fid_minus_list = []
            for i in range(batch.num_graphs):
                data = batch[i]
                graph = to_networkx(data, to_undirected=True)
                
                # Get true edge labels and number of edges
                true_edge_labels = data.edge_label.cpu().numpy()
                num_edges = true_edge_labels.shape[0]
                
                # Get the edge mask for the active nodes in the subgraph
                nodelist = active_node_index[i]
                if not isinstance(nodelist, list):
                    continue
                edge_mask = get_edge_mask(graph, nodelist, num_edges)
                
                # Convert edge mask to numpy array for AUROC calculation
                pred_edge_scores = edge_mask.int().cpu().numpy()
                
                intersection = len(set(pred_edge_scores) & set(true_edge_labels))
                union = len(set(pred_edge_scores) | set(true_edge_labels))
                iou = intersection / union
                iou_list.append(iou)

                # if node is active and only has edges to other active nodes, category = 1
                # if node is active and has edges to both active and inactive nodes, category = 2
                # if node is not active, and does not have edges to any active nodes, category = 3

                from collections import defaultdict
                adj_dict = defaultdict(set)

                for src, dst in data.edge_index.t().tolist():
                    adj_dict[src].add(dst)
                    adj_dict[dst].add(src)  

                active_nodes_set = set(active_node_index[i])
                all_nodes = set(adj_dict.keys())

                category_1 = []  # active, only connects to active
                category_2 = []  # active, connects to both active and inactive
                category_3 = []  # inactive, no connection to active

                for node in all_nodes:
                    neighbors = adj_dict[node]
                    is_active = node in active_nodes_set
                    has_active_neighbors = any(n in active_nodes_set for n in neighbors)
                    has_inactive_neighbors = any(n not in active_nodes_set for n in neighbors)

                    if is_active:
                        if has_inactive_neighbors and has_active_neighbors:
                            category_2.append(node)
                        elif has_active_neighbors and not has_inactive_neighbors:
                            category_1.append(node)
                    else:
                        if not has_active_neighbors:
                            category_3.append(node)

                # print("cat 1: ", category_1)
                # print("cat 2: ", category_2)
                # print("cat 3: ", category_3)


            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

            # HERE fidelity
            print("active_node_index: ", active_node_index)
            # topk_node_index = [tensor.tolist() for tensor in topk_node_index]
            # bottomk_node_index = [tensor.tolist() for tensor in bottomk_node_index]
            print("topk: ", topk_node_index)
            print("bottomk: ", bottomk_node_index)
            retained_batch_pos, removed_batch_pos = build_explanation_subgraphs(batch, topk_node_index)
            retained_batch_neg, removed_batch_neg = build_explanation_subgraphs(batch, bottomk_node_index)

            logits_full, probs, *_ = gnnNets(batch)
            logits_retained, _, *_ = gnnNets(retained_batch_pos)
            logits_removed, _, *_ = gnnNets(retained_batch_neg)

            # print("logits_retained shape:", logits_retained.shape)
            # print("retained_batch.batch shape:", retained_batch.batch.shape)

            _, y_pred = torch.max(logits_full, -1)
            _, y_pred_retained = torch.max(logits_retained, -1)
            _, y_pred_removed = torch.max(logits_removed, -1)

            print("Calling calc_fidelity with:")
            print("batch.y:", batch.y.shape)
            print("y_pred:", y_pred.shape)
            print("y_pred_removed:", y_pred_removed.shape)
            print("y_pred_retained:", y_pred_retained.shape)

            fid_plus, fid_minus = calc_fidelity(batch.y, y_pred, y_pred_removed, y_pred_retained)
            print(f"Fid+: {fid_plus:.4f}, Fid-: {fid_minus:.4f}")
            fid_plus_list.append(fid_plus)
            fid_minus_list.append(fid_minus)



    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0).mean()),
                  'iou':np.average(iou_list),
                  'fid+': np.average(fid_plus_list),
                  'fid-': np.average(fid_minus_list)}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best, args):
    # print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }

    task = args.task

    pth_name = f"{model_name}_{model_type}_{model_args.readout}_latest_{task}.pth"
    best_pth_name = f'{model_name}_{model_type}_{model_args.readout}_best_{task}.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(gnnNets, os.path.join(ckpt_dir, best_pth_name) )
    gnnNets.to(model_args.device)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PGIB')
    parser.add_argument('--task', type=str, help='description for filenames')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--fc_dims', nargs='+', type=int, help='Dimensions for FC layers after GNN layers')
    args = parser.parse_args()
    task = args.task
    model_args.fc_dims = args.fc_dims
    print("fc dims: ", args.fc_dims)

    if os.path.isfile(f"./log/hyper_search_{task}.txt"):
        os.remove(f"./log/hyper_search_{task}.txt")

    if model_args.cont:
        model_type = 'cont'
    else:
        model_type = 'var'

    accuracy = train_GC(model_type, args)