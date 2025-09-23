import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import pickle as pkl
import os

from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict

#SBATCH --gres=gpu:h200-1g.35gb:1
#SBATCH --nodes=1 --ntasks-per-node 8
class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config, gaussianize=False, max_gauss_var=0.05, max_gauss_schedule='fixed', nograd_on='on'):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']
        self.bottleneck_dim = method_config.get('bottleneck_dim', 'normal')
        self.gaussianize = gaussianize
        self.max_gauss_var = max_gauss_var
        self.max_gauss_schedule = max_gauss_schedule
        self.nograd_on = nograd_on

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef

        #change -eren
        if self.bottleneck_dim == 'normal':
            loss = pred_loss + info_loss
        else:
            loss = pred_loss
        
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        
        
        if self.gaussianize:
            # Conditionally use torch.no_grad() based on nograd_on flag
            if self.nograd_on == 'on':
                with torch.no_grad():
                    # Get attention scores from extractor
                    att_eps = torch.sigmoid(self.extractor(emb, data.edge_index, data.batch)) + 1e-6

                    if self.learn_edge_att:
                        # Convert edge attention to node attention by averaging
                        node_att = torch.zeros(emb.size(0), device=emb.device)

                        # Squeeze to (num_edges,) then expand to (2, num_edges)
                        edge_att_expanded = att_eps.squeeze(-1).unsqueeze(0).expand(2, -1)

                        node_att.scatter_add_(0, data.edge_index.view(-1), edge_att_expanded.reshape(-1))

                        # Count edges per node for averaging
                        edge_count = torch.zeros(emb.size(0), device=emb.device)
                        edge_count.scatter_add_(0, data.edge_index.view(-1),
                                                torch.ones_like(data.edge_index.view(-1), dtype=torch.float))
                        edge_count = torch.clamp(edge_count, min=1.0)  # Avoid division by zero

                        # Average attention per node
                        node_att = node_att / edge_count
                    else:
                        # Node attention: use directly
                        node_att = att_eps

                    # Inverse scaling (more noise for lower attention)
                    scale = (1.0 / node_att)
                    
                    # Apply scheduling to max_gauss_var
                    if self.max_gauss_schedule == 'linear':
                        progress = epoch / self.epochs
                        current_max_var = 0.001 + (self.max_gauss_var - 0.001) * progress

                    elif self.max_gauss_schedule == 'cosine':
                        progress = epoch / self.epochs
                        current_max_var = self.max_gauss_var * (1 - torch.cos(torch.tensor(progress * 3.14159))) / 2

                    elif self.max_gauss_schedule == 'exp':
                        progress = epoch / self.epochs
                        current_max_var = 0.001 * ((self.max_gauss_var / 0.001) ** progress)

                    elif self.max_gauss_schedule == 'step':
                        if epoch < int(0.3 * self.epochs):
                            current_max_var = 0.0
                        else:
                            progress = (epoch - 0.3 * self.epochs) / (0.7 * self.epochs)
                            current_max_var = 0.0 + (self.max_gauss_var - 0.0) * progress

                    elif self.max_gauss_schedule == 'sigmoid':
                        progress = epoch / self.epochs
                        k = 10.0  # steeper = faster ramp around midpoint
                        current_max_var = self.max_gauss_var / (1 + torch.exp(-k * (progress - 0.5)))

                    else:  # fixed
                        current_max_var = self.max_gauss_var

                    
                    scale = (scale / scale.max()) * current_max_var  # max stddev (tunable)

                    # Reshape to (N, 1) for broadcasting over embedding dim
                    scale = scale.view(-1, 1)

                    # Add noise
                    noise = torch.randn_like(emb) * scale
                    emb = emb + noise
            else:
                # Get attention scores from extractor (with gradients)
                att_eps = torch.sigmoid(self.extractor(emb, data.edge_index, data.batch)) + 1e-6

                if self.learn_edge_att:
                    # Convert edge attention to node attention by averaging
                    node_att = torch.zeros(emb.size(0), device=emb.device)

                    # Squeeze to (num_edges,) then expand to (2, num_edges)
                    edge_att_expanded = att_eps.squeeze(-1).unsqueeze(0).expand(2, -1)

                    node_att.scatter_add_(0, data.edge_index.view(-1), edge_att_expanded.reshape(-1))

                    # Count edges per node for averaging
                    edge_count = torch.zeros(emb.size(0), device=emb.device)
                    edge_count.scatter_add_(0, data.edge_index.view(-1),
                                            torch.ones_like(data.edge_index.view(-1), dtype=torch.float))
                    edge_count = torch.clamp(edge_count, min=1.0)  # Avoid division by zero

                    # Average attention per node
                    node_att = node_att / edge_count
                else:
                    # Node attention: use directly
                    node_att = att_eps

                # Inverse scaling (more noise for lower attention)
                scale = (1.0 / node_att)
                
                # Apply scheduling to max_gauss_var
                if self.max_gauss_schedule == 'linear':
                    progress = epoch / self.epochs
                    current_max_var = 0.001 + (self.max_gauss_var - 0.001) * progress

                elif self.max_gauss_schedule == 'cosine':
                    progress = epoch / self.epochs
                    current_max_var = self.max_gauss_var * (1 - torch.cos(torch.tensor(progress * 3.14159))) / 2

                elif self.max_gauss_schedule == 'exp':
                    progress = epoch / self.epochs
                    current_max_var = 0.001 * ((self.max_gauss_var / 0.001) ** progress)

                elif self.max_gauss_schedule == 'step':
                    if epoch < int(0.8 * self.epochs):
                        current_max_var = 0.001
                    else:
                        progress = (epoch - 0.8 * self.epochs) / (0.2 * self.epochs)
                        current_max_var = 0.001 + (self.max_gauss_var - 0.001) * progress

                elif self.max_gauss_schedule == 'sigmoid':
                    progress = epoch / self.epochs
                    k = 10.0  # steeper = faster ramp around midpoint
                    current_max_var = self.max_gauss_var / (1 + torch.exp(-k * (progress - 0.5)))

                else:  # fixed
                    current_max_var = self.max_gauss_var

                
                scale = (scale / scale.max()) * current_max_var  # max stddev (tunable)

                # Reshape to (N, 1) for broadcasting over embedding dim
                scale = scale.view(-1, 1)

                # Add noise
                noise = torch.randn_like(emb) * scale
                emb = emb + noise

        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, epoch, training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits, _ = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch)
            

        return edge_att, loss, loss_dict, clf_logits, att


    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()
        print()
        att, loss, loss_dict, clf_logits, node_att = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()

        att, loss, loss_dict, clf_logits, node_att = self.forward_pass(data, epoch, training=True)
        embeddings = self.clf.get_all_embeddings()

        global all_embs
        for key in embeddings:
            all_embs[key].append((embeddings[key], data, node_att))


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            if self.dataset_name in ['NCI1', 'PROTEINS', 'AIDS']:
                data.y = data.y.unsqueeze(-1)

            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()

            
            precision_at_k = self.get_precision_at_k(att.cpu(), exp_labels.cpu(), self.k, data.batch.cpu(), data.edge_index.cpu())

            desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                 data.y.data.cpu(), clf_logits, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v


            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)
            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att,
                
                                                                                      all_precision_at_k, all_clf_labels, all_clf_logits, batch=False)
            pbar.set_description(desc)
        return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)
        
        # Convergence tracking - no improvement in val accuracy for patience epochs
        best_val_accuracy = None
        epochs_without_improvement = 0
        convergence_epoch = None
        patience = 50
        
        for epoch in range(300):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('gsat_train/lr', get_lr(self.optimizer), epoch)
            
            # Check for convergence - no improvement in val accuracy for patience epochs
            current_val_accuracy = valid_res[2]  # clf_acc is at index 2
            
            # Initialize best_val_accuracy on first epoch
            if best_val_accuracy is None:
                best_val_accuracy = current_val_accuracy
            elif current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            # Check if we've reached convergence (no improvement for patience epochs)
            if epochs_without_improvement >= patience and convergence_epoch is None:
                convergence_epoch = epoch
                print(f"[CONVERGENCE] Model converged at epoch {epoch} (no val accuracy improvement for {patience} epochs)")
                print(f"[CONVERGENCE] Best val accuracy: {best_val_accuracy:.3f}, Current val accuracy: {current_val_accuracy:.3f}")

            assert len(train_res) == 5
            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2  # clf_roc or clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if (r == self.final_r or self.fix_r) and epoch > 10 and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                                     or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                         and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'gsat_best/{metric}', value, epoch)

            if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
                if self.multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')
        
        # Print convergence summary and ensure convergence_epoch is always in metric_dict
        if convergence_epoch is not None:
            print(f"[CONVERGENCE SUMMARY] Model converged at epoch {convergence_epoch} (no val accuracy improvement for {patience} epochs)")
            print(f"[CONVERGENCE SUMMARY] Best validation accuracy achieved: {best_val_accuracy:.3f}")
            # Ensure convergence_epoch is in the final metric_dict (it might have been lost if a better model was found after convergence)
            metric_dict['metric/convergence_epoch'] = convergence_epoch
        else:
            print(f"[CONVERGENCE SUMMARY] Model did not converge within {self.epochs} epochs (no {patience}-epoch plateau in val accuracy)")
            if best_val_accuracy is not None:
                print(f"[CONVERGENCE SUMMARY] Best validation accuracy achieved: {best_val_accuracy:.3f}")
            else:
                print(f"[CONVERGENCE SUMMARY] No validation accuracy recorded")
            metric_dict['metric/convergence_epoch'] = -1  # -1 indicates no convergence
        
        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'gsat_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
        desc += eval_desc
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        self.writer.add_histogram(f'gsat_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        self.writer.add_histogram(f'gsat_{phase}/signal_att_weights', signal_att_weights, epoch)
        self.writer.add_scalar(f'gsat_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/precision@{self.k}/', precision_at_k, epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.writer.add_pr_curve(f'PR_Curve/gsat_{phase}/', exp_labels, att, epoch)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        if self.dataset_name in ['NCI1', 'PROTEINS', 'AIDS']:
            data.y = data.y.unsqueeze(-1)

        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset.cpu(), data.edge_index.cpu(), edge_attr=batch_att.cpu())

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name, norm=self.viz_norm_att, mol_type=mol_type, coor=coor)
            imgs.append(img)
        imgs = np.stack(imgs)
        self.writer.add_images(tag, imgs, epoch, dataformats='NHWC')

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)
            #self.feature_extractor = MLP([16, 32, 16, 1], dropout=dropout_p)
            #-eren
            #self.feature_extractor = MLP([64, 128, 512, 128, 64, 1], dropout=dropout_p)
            #self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size * 4, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, save_embs=False, actual_model_name=None, gaussianize=False, max_gauss_var=0.05, max_gauss_schedule='fixed', nograd_on='on'):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == (actual_model_name or model_name)
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    # Save original model name and override to be the base name for get_model function
    original_model_name = model_config['model_name']
    model_config['model_name'] = model_name  
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device, save_embs=save_embs)
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        print('[INFO] Pretraining the model...')
        train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                           model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        load_checkpoint(model, model_dir=log_dir, model_name=f'epoch_{pretrain_epochs}')
    else:
        print('[INFO] Training both the model and the attention from scratch...')

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    activations = defaultdict(list)

    if save_weights:
        def hook_fn(module, input, output):
            global activations
            activations[module].append(output.clone())

        for layer in extractor.children():
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(hook_fn)

    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training GSAT...')
    gsat = GSAT(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config, gaussianize=gaussianize, max_gauss_var=max_gauss_var, max_gauss_schedule=max_gauss_schedule, nograd_on=nograd_on)
    metric_dict = gsat.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    
    # Restore original model name for next seed
    model_config['model_name'] = original_model_name
    
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')

    parser.add_argument('--save_weights', type=bool, help='save activations')
    parser.add_argument('--bottleneck_dim', type=str, required=True, help='dimensions for bottleneck FC layers (e.g., "16" or "48-32")')
    parser.add_argument('--save_embs', action='store_true', help='save embeddings from model layers (default: False)')
    parser.add_argument('--seeds', type=int, help='number of random seeds to run (overrides global config)')
    parser.add_argument('--gaussianize', action='store_true', help='Add Gaussian noise to embeddings based on attention scores')
    parser.add_argument('--max_gauss_var', type=float, default=0.05, help='Maximum Gaussian noise variance (default: 0.05)')
    parser.add_argument('--max_gauss_schedule', type=str, default='fixed', choices=['fixed', 'linear', 'cosine', 'exp', 'step', 'sigmoid'], help='Gaussian noise scheduling strategy (default: fixed)')
    parser.add_argument('--nograd_on', type=str, default='on', choices=['on', 'off'], help='Whether to use torch.no_grad() for gaussianize operations (default: on)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train (default: 300)')

    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda
    
    print('bottleneck_dim', args.bottleneck_dim)

    #print('TORCH GRAD IS ON FOR THIS GAUSSIANIZE')

    global save_weights
    save_weights = args.save_weights

    from collections import defaultdict

    global all_embs
    all_embs = defaultdict(list)

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    global tempgotcha
    tempgotcha = True

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print(f"CUDA Available: {cuda_available}")
    print(f"Device: {device}")
    
    print(f'save_embs: {args.save_embs}')
    print(f'gaussianize: {args.gaussianize}')
    print(f'max_gauss_var: {args.max_gauss_var}')
    print(f'max_gauss_schedule: {args.max_gauss_schedule}')
    
    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    
    # Determine the actual model that will be used based on bottleneck_dim
    if args.bottleneck_dim in ['normal', 'noinfo']:
        actual_model_name = model_name  # Use regular model
    else:
        actual_model_name = f'{model_name}_with_fc_extractor'  # Use fc_extractor variant
    
    print(f'[INFO] Loading config for: {actual_model_name}')
    local_config_name = get_local_config_name(actual_model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))
    
    # Parse bottleneck dimensions from string and add to model_config
    if isinstance(args.bottleneck_dim, str):
        # Skip parsing for special values
        if args.bottleneck_dim in ['normal', 'noinfo']:
            bottleneck_dims = None  # Won't be used for these special cases
        else:
            bottleneck_dims = [int(x) for x in args.bottleneck_dim.split('-')]
    else:
        bottleneck_dims = [args.bottleneck_dim]  # backwards compatibility
    
    local_config['model_config']['bottleneck_dims'] = bottleneck_dims
    local_config['model_config']['bottleneck_dim'] = args.bottleneck_dim  # Keep original string for reference
    local_config['GSAT_config']['bottleneck_dim'] = args.bottleneck_dim  # Also add to method config for GSAT class

    data_dir = Path(global_config['data_dir'])
    num_seeds = args.seeds if args.seeds is not None else global_config['num_seeds']
    print(f'[INFO] Running with {num_seeds} seeds')

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    # Create embeddings folder if saving embeddings
    if args.save_embs:
        gaussianize_suffix = f'_gaussianize_var_{args.max_gauss_var}_sched_{args.max_gauss_schedule}_nograd_{args.nograd_on}' if args.gaussianize else ''
        embeddings_folder = f'new_embeddings_{dataset_name}_{model_name}_{args.bottleneck_dim.replace("-", "_")}_{gaussianize_suffix}'
        os.makedirs(embeddings_folder, exist_ok=True)
        print(f'[INFO] Created embeddings folder: {embeddings_folder}')
 
    print(f'[INFO] Gaussianize: {args.gaussianize}')
    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, save_embs=args.save_embs, actual_model_name=actual_model_name, gaussianize=args.gaussianize, max_gauss_var=args.max_gauss_var, max_gauss_schedule=args.max_gauss_schedule, nograd_on=args.nograd_on)
        metric_dicts.append(metric_dict)

        if args.save_embs:
            # Save all embeddings that were collected during training  
            print('[INFO] Saving embeddings...')
            import pickle
            gaussianize_suffix = f'_gaussianize_var_{args.max_gauss_var}_sched_{args.max_gauss_schedule}_nograd_{args.nograd_on}' if args.gaussianize else ''
            embeddings_folder = f'new_embeddings_{dataset_name}_{model_name}_{args.bottleneck_dim.replace("-", "_")}_{gaussianize_suffix}'
            embeddings_file = os.path.join(embeddings_folder, f'embeddings_seed_{random_state}.pkl')
            with open(embeddings_file, 'wb') as f:
                pickle.dump(all_embs, f)
            print(f'[INFO] Embeddings saved to: {embeddings_file}')
            print(f'[INFO] Number of embedding keys: {len(all_embs)}')
            for key, embs in all_embs.items():
                print(f'[INFO] Key {key}: {len(embs)} embeddings saved')
            # Clear embeddings for next seed
            all_embs.clear()
    print(metric_dicts)
    log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)
    
    best_clf_valid = []
    best_clf_test = []
    best_x_roc_test = []
    best_epochs = []
    convergence_epochs = []
    for i, metric_dict in enumerate(metric_dicts):
        best_clf_valid.append(metric_dict["metric/best_clf_valid"])
        best_clf_test.append(metric_dict[ "metric/best_clf_test"])
        best_x_roc_test.append(metric_dict["metric/best_x_roc_test"])
        best_epochs.append(metric_dict["metric/best_clf_epoch"])
        convergence_epochs.append(metric_dict.get("metric/convergence_epoch", -1))

    
    agg_clf_valid = sum(best_clf_valid) / len(best_clf_valid)
    agg_clf_test = sum(best_clf_test) / len(best_clf_test)
    agg_x_roc_test = sum(best_x_roc_test) / len(best_x_roc_test)
    agg_best_epoch = sum(best_epochs) / len(best_epochs)
    
    # Calculate average convergence epoch (only for seeds that converged)
    converged_epochs = [epoch for epoch in convergence_epochs if epoch != -1]
    if converged_epochs:
        agg_convergence_epoch = sum(converged_epochs) / len(converged_epochs)
        convergence_rate = len(converged_epochs) / len(convergence_epochs)
    else:
        agg_convergence_epoch = -1
        convergence_rate = 0.0 

    std_clf_test = np.std(best_clf_test, ddof=1)
    std_roc_test = np.std(best_x_roc_test, ddof=1)

    print("="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Bottleneck Dim: {args.bottleneck_dim}")
    print(f"Gaussianize: {args.gaussianize}")
    if args.gaussianize:
        print(f"Max Gauss Var: {args.max_gauss_var}")
        print(f"Gauss Schedule: {args.max_gauss_schedule}")
        print(f"Nograd On: {args.nograd_on}")
    print(f"Seeds: {num_seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"CUDA Device: {cuda_id}")
    print("-"*60)
    print(f"Best Test Pred ACC/ROC:   {agg_clf_test:.3f} ± {std_clf_test:.3f}")
    print(f"Best Test X AUROC:        {agg_x_roc_test:.3f} ± {std_roc_test:.3f}")
    print(f"Average Best Epoch:       {agg_best_epoch:.1f}")
    if agg_convergence_epoch != -1:
        std_convergence_epoch = np.std(converged_epochs, ddof=1) if len(converged_epochs) > 1 else 0.0
        print(f"Average Convergence Epoch: {agg_convergence_epoch:.1f} ± {std_convergence_epoch:.1f}")
        print(f"Convergence Rate:         {convergence_rate:.1%}")
    else:
        print(f"Average Convergence Epoch: No convergence")
        print(f"Convergence Rate:         0.0%")
    print("-"*60)
    print("="*60)

    
    # Also print in a single line format for easy parsing
    gauss_params = f"{args.max_gauss_var}|{args.max_gauss_schedule}|{args.nograd_on}" if args.gaussianize else "None|None|None"

    if agg_convergence_epoch != -1:
        std_convergence_epoch = np.std(converged_epochs, ddof=1) if len(converged_epochs) > 1 else 0.0
        conv_epoch_str = f"{agg_convergence_epoch:.1f}±{std_convergence_epoch:.1f}"
    else:
        std_convergence_epoch = "NA"
        conv_epoch_str = "NoConv"

    print(
        f"RESULT|{dataset_name}|{model_name}|{args.bottleneck_dim}|{args.gaussianize}|"
        f"{gauss_params}|{num_seeds}|{agg_clf_test:.3f}|{agg_x_roc_test:.3f}|"
        f"{std_clf_test:.3f}|{std_roc_test:.3f}|{agg_best_epoch:.1f}|"
        f"{conv_epoch_str}|{convergence_rate:.1%}"
    )


    if save_weights:
        print('activations', activations)

if __name__ == '__main__':
    main()
