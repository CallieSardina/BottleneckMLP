import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem

from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict
from utils.mi_utils.edge_estimator import EDGE
from utils.mi_utils.si_estimator import optimal_SI

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
import torch_scatter

import matplotlib.pyplot as plt
import functools


class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config):
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

        # HERE -- for plots drift/ diffusion
        # self.gradient_means_clf = {}
        # self.gradient_stds_clf = {}
        # self.gradient_means_extractor = {}
        # self.gradient_stds_extractor = {}
        # self.hooks = []
        # self._register_gradient_hooks()

        # self.gradient_means_extractor_epoch = {}
        # self.gradient_stds_extractor_epoch = {}
        # self.gradient_means_clf_epoch = {}
        # self.gradient_stds_clf_epoch = {}

    # HERE -- for plots drift/ diffusion
    # def _register_gradient_hooks(self):
    #     def hook_fn(module, grad_input, grad_output, name):
    #         grad = grad_output[0].detach()
    #         mean_grad = grad.mean().item()
    #         std_grad = grad.std().item()

    #         # Check whether it's a classifier or extractor layer
    #         if "feature_extractor" in name or "extractor" in name:
    #             if name not in self.gradient_means_extractor:
    #                 self.gradient_means_extractor[name] = []
    #                 self.gradient_stds_extractor[name] = []
    #             self.gradient_means_extractor[name].append(mean_grad)
    #             self.gradient_stds_extractor[name].append(std_grad)
    #         else:
    #             if name not in self.gradient_means_clf:
    #                 self.gradient_means_clf[name] = []
    #                 self.gradient_stds_clf[name] = []
    #             self.gradient_means_clf[name].append(mean_grad)
    #             self.gradient_stds_clf[name].append(std_grad)

    #     # Register hooks for classifier (self.clf)
    #     for name, module in self.clf.named_modules():
    #         # print("clf: ", name, module)
    #         if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
    #             hook = module.register_full_backward_hook(functools.partial(hook_fn, name=name))
    #             self.hooks.append(hook)

    #     # Register hooks for extractor (self.extractor)
    #     for name, module in self.extractor.named_modules():
    #         # print("extractor: ", name, module)
    #         if isinstance(module, torch.nn.Linear):
    #             hook = module.register_full_backward_hook(functools.partial(hook_fn, name=name))
    #             self.hooks.append(hook)

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = 0.0 #(att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef

        if exp_type == 'normal':
            loss = pred_loss + info_loss
        else:
            loss = pred_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        # embeddings_per_layer = [] 

        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)

        # print("Emb shape: ", emb.shape)
        # embeddings_per_layer.append(emb)

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

        clf_logits, g_sub = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        embeddings = self.clf.get_all_embeddings()
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch)
        return edge_att, loss, loss_dict, clf_logits, emb, embeddings, g_sub # HERE -- add g_sub in calls

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()

        att, loss, loss_dict, clf_logits, _, _, _ = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch, variances_per_epoch = None):
        self.extractor.train()
        self.clf.train()

        mi_XZ = []
        mi_ZY = []

        att, loss, loss_dict, clf_logits, emb, embeddings, emb_sub = self.forward_pass(data, epoch, training=True)

        graph_emb = torch_scatter.scatter(emb, data.batch, dim=0, reduce="mean")  # Shape: [batch_size, emb_dim]
        # graph_emb_sub = torch_scatter.scatter(emb_sub, data.batch, dim=0, reduce="mean")  # Shape: [batch_size, emb_dim]

        # compute MI
        g = graph_emb.clone().detach().cpu().numpy()
        g_s = emb_sub.clone().detach().cpu().numpy()
        y = data.y.clone().detach().cpu().numpy()

        if calculate_variance:
            for key in embeddings:
                embeds_in_layer = embeddings[key].clone().detach().cpu()
                var = embeds_in_layer.var(dim=0, unbiased=False)
                var_mean = var.mean()
                if key not in variances_per_epoch:
                    variances_per_epoch[key] = []
                variances_per_epoch[key].append(var_mean)
        
        mi_g_gs = EDGE(g, g_s)
        mi_gs_y = EDGE(g_s, y)
        
        global ahh
        if not ahh:
            print('embeddings', embeddings)
            ahh = True
        
        # print('g.shape', g.shape)
        # print('g_s.shape', g_s.shape)

        #mi_g_gs = optimal_SI.optimal_SI(g, g_s, num_slices=100)
        #mi_gs_y = optimal_SI.optimal_SI(g_s, y, num_slices=100)
            
        #mi_XZ = [EDGE(embeddings['fc_0'].clone().detach().cpu().numpy(), embeddings[key].clone().detach().cpu().numpy()) for key in embeddings]
        #mi_XZ = [optimal_SI.optimal_SI(embeddings['fc_0'].clone().detach().cpu().numpy(), embeddings[key].clone().detach().cpu().numpy(), num_slices=100) for key in embeddings]

        print("Embeddings keys:", embeddings.keys())
        # mi_XZ = [EDGE(embeddings['x_before_ext'].clone().detach().cpu().numpy(), embeddings['pre_fc_out'].clone().detach().cpu().numpy()),
        #          EDGE(embeddings['x_after_ext'].clone().detach().cpu().numpy(), embeddings['pre_fc_out'].clone().detach().cpu().numpy())]

        mi_XZ = [EDGE(embeddings['x_before_ext'].clone().detach().cpu().numpy(), embeddings[key].clone().detach().cpu().numpy()) for key in embeddings]
        
        

        mi_ZY = [EDGE(y, embeddings[key].clone().detach().cpu().numpy()) for key in embeddings]
        #mi_ZY = [EDGE(y, embeddings['pre_fc_out'].clone().detach().cpu().numpy()), EDGE(y, embeddings['pre_fc_out'].clone().detach().cpu().numpy())]


        for i in range(len(mi_XZ)):
            mi_XZ[i] = mi_XZ[i].item()
        
        for i in range(len(mi_ZY)):
            mi_ZY[i] = mi_ZY[i].item()
            
        # mi_ZY = [optimal_SI.optimal_SI(y, embeddings[key].clone().detach().cpu().numpy(), num_slices=100) for key in embeddings]
        # with open(f'MI_logs/gsat_ggs_{exp_type}-{dataset_name}-{timestamp}.txt', 'a') as f:
        #     print(f"Epoch {epoch}, MI_XZ: {mi_g_gs}, MI_ZY: {mi_gs_y}", file=f)

        with open(f'MI_logs/gsat_embs_{exp_type}-{dataset_name}-{timestamp}.txt', 'a') as f:
            print(f"Epoch {epoch}, MI_XZ: {mi_XZ}, MI_ZY: {mi_ZY}", file=f)

        
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
        if calculate_variance:
            variances_per_epoch = {}
        else:
            variances_per_epoch = None

        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            if phase == 'train':
                att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch, variances_per_epoch)
            else:
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

        if calculate_variance:
            for key in variances_per_epoch:
                if key not in all_variances:
                    all_variances[key] = []
                all_variances[key].append(sum(variances_per_epoch[key]) / len(variances_per_epoch[key]))
        return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('gsat_train/lr', get_lr(self.optimizer), epoch)

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
                # for idx, tag in viz_set:
                #     self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')
            # HERE drift/ diff
            # for name in self.gradient_means_extractor:
            #     if name not in self.gradient_means_extractor_epoch:
            #         self.gradient_means_extractor_epoch[name] = []
            #         self.gradient_stds_extractor_epoch[name] = []
                
            #     self.gradient_means_extractor_epoch[name].append(np.mean(self.gradient_means_extractor[name]))
            #     self.gradient_stds_extractor_epoch[name].append(np.mean(self.gradient_stds_extractor[name]))

            # for name in self.gradient_means_clf:
            #     if name not in self.gradient_means_clf_epoch:
            #         self.gradient_means_clf_epoch[name] = []
            #         self.gradient_stds_clf_epoch[name] = []

            #     self.gradient_means_clf_epoch[name].append(np.mean(self.gradient_means_clf[name]))
            #     self.gradient_stds_clf_epoch[name].append(np.mean(self.gradient_stds_clf[name]))

        # HERE drift/ diffusion
        # Remove hooks after training
        # for hook in self.hooks:
        #     hook.remove()
        # # Plot gradient statistics
        # self._plot_gradient_statistics()


        if calculate_variance:
            print('all_variances', all_variances)
            plt.figure(figsize=(10, 6))

            for layer, vars in all_variances.items():
                plt.plot(vars, label=layer)

            plt.xlabel("Epoch")
            plt.ylabel("Variance")
            plt.title("Embedding Variance per Layer")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            plt.savefig(f"MI_logs/vars_{exp_type}-{dataset_name}.png", dpi=300)  # or .pdf, .svg, etc.

            plt.close()  # Close the figure to free up memory if plotting repeatedly
                

        return metric_dict

    # HERE drift/ diffusion
    # def _plot_gradient_statistics(self):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    #     # Function to plot mean and std with solid and dotted lines
    #     def plot_gradient_stats(ax, gradient_means, gradient_stds, title):
    #         for layer, means in gradient_means.items():
    #             stds = gradient_stds[layer]
    #             epochs = np.arange(1, len(means) + 1)

    #             # Normalize
    #             norm_means = np.array(means) / abs(means[0])
    #             norm_stds = np.array(stds) / abs(means[0])

    #             # Plot mean (solid) and std (dotted)
    #             ax.plot(epochs, norm_means, label=f"Mean {layer}", linestyle="-")
    #             ax.plot(epochs, norm_stds, linestyle=":", color=ax.get_lines()[-1].get_color(), label=f"STD {layer}")

    #         ax.set_xscale("log")
    #         ax.set_yscale("log")
    #         ax.set_xlabel("# Epochs", fontsize=12)
    #         ax.set_ylabel("Normalized Mean and STD", fontsize=12)
    #         ax.set_title(title, fontsize=14)
    #         ax.legend(fontsize=10)
    #         ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    #     # Classifier Gradients
    #     plot_gradient_stats(ax1, self.gradient_means_clf_epoch, self.gradient_stds_clf_epoch, "Gradient Drift & Diffusion (Classifier)")

    #     # Extractor Gradients
    #     plot_gradient_stats(ax2, self.gradient_means_extractor_epoch, self.gradient_stds_extractor_epoch, "Gradient Drift & Diffusion (Extractor)")

    #     plt.tight_layout()
    #     plt.savefig("gradient_drift_diffusion_fc_inside_GIN_64_16_64_400_epochs.png", dpi=300)
    #     plt.show()

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
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

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
            # print([hidden_size * 2, hidden_size * 4, hidden_size, 1])
            layers_list = [hidden_size * 2, hidden_size * 4, hidden_size, 1]
            # print(layers_list)
            self.feature_extractor = MLP(layers_list, dropout=dropout_p)
        else:
            # print([hidden_size * 1, hidden_size * 2, hidden_size, 1])
            layers_list = [hidden_size, hidden_size * 2, hidden_size, 1]
            # print(layers_list)
            self.feature_extractor = MLP(layers_list, dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)

        # Debug: Print model structure before wrapping
        # print("\n=== Feature Extractor Model ===")
        # print(self.feature_extractor)

        # Use the feature extractor wrapper
        # wrapper = FeatureExtractorWrapper(self.feature_extractor)
        # embeddings = wrapper.get_activations(emb, batch)  

        return att_log_logits

# class FeatureExtractorWrapper:
#     def __init__(self, model):
#         self.model = model
#         self.activations = {}
#         self.hooks = []
#         self._register_hooks()

#     def _register_hooks(self):
#         # print("\n=== Registering Hooks for Feature Extraction ===")
#         for name, layer in self.model.named_modules():  
#             if isinstance(layer, nn.Linear):
#                 # print(f"Registering hook for: {name} ({layer})")
#                 hook = layer.register_forward_hook(self._get_hook(name))
#                 self.hooks.append(hook)  

#     def _get_hook(self, name):
#         def hook(module, input, output):
#             # print(f"Capturing activation from layer: {name}")
#             self.activations[name] = output.clone().detach()  
#         return hook

#     def get_activations(self, x, batch):
#         self.activations = {}  
#         # print("\n=== Running Forward Pass to Capture Activations ===")
#         _ = self.model(x, batch) 
#         # print("Captured activations keys:", self.activations.keys())
#         return self.activations

#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks = []


# class ExtractorMLP(nn.Module):

#     def __init__(self, hidden_size, shared_config):
#         super().__init__()
#         self.learn_edge_att = shared_config['learn_edge_att']
#         dropout_p = shared_config['extractor_dropout_p']
#         print([hidden_size * 2, hidden_size * 4] + list(range(hidden_size, 1, -2)) + [1])
#         if self.learn_edge_att:
#             #self.feature_extractor = MLP([64, hidden_size * 2, hidden_size * 4] + list(range(hidden_size, 1, -2)) + [1], dropout=dropout_p)
#             self.feature_extractor = MLP([128, 256, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1], dropout=dropout_p)            
#         else:
#             self.feature_extractor = MLP([128, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1], dropout=dropout_p)


#     def forward(self, emb, edge_index, batch):
#         if self.learn_edge_att:
#             col, row = edge_index
#             f1, f2 = emb[col], emb[row]
#             f12 = torch.cat([f1, f2], dim=-1)
#             att_log_logits = self.feature_extractor(f12, batch[col])
#         else:
#             att_log_logits = self.feature_extractor(emb, batch)
#         embs = self.feature_extractor.embeddings
#         print(embs)
#         print(len(embs))
#         return att_log_logits, embs

# class MLP(nn.Module): # HERE has to make MLP class to track embeddings
#     def __init__(self, layer_sizes, dropout=0.5):
#         super(MLP, self).__init__()
#         layers = []
#         for i in range(len(layer_sizes) - 1):
#             layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
#             if i < len(layer_sizes) - 2:  
#                 layers.append(nn.ReLU())
#                 layers.append(nn.Dropout(dropout))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         embeddings = []  
#         for layer in self.model:
#             x = layer(x)
#             if isinstance(layer, nn.Linear): 
#                 embeddings.append(x.clone().detach())
#         return x, embeddings
    
# class ExtractorMLP(nn.Module):
#     def __init__(self, hidden_size, shared_config):
#         super().__init__()
#         self.learn_edge_att = shared_config['learn_edge_att']
#         dropout_p = shared_config['extractor_dropout_p']
#         print([hidden_size * 2, hidden_size * 4] + list(range(hidden_size, 1, -2)) + [1])
#         if self.learn_edge_att:
#             self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4] + list(range(hidden_size, 1, -2)) + [1], dropout=dropout_p)
#         else:
#             self.feature_extractor = MLP([hidden_size * 2]  + list(range(hidden_size, 1, -2)) + [1], dropout=dropout_p)

#     def forward(self, emb, edge_index, batch):
#         embeddings = []  # Store activations

#         # Get embeddings from feature_extractor
#         if self.learn_edge_att:
#             col, row = edge_index
#             f1, f2 = emb[col], emb[row]
#             f12 = torch.cat([f1, f2], dim=-1)
#             att_log_logits, feature_extractor_embs = self.feature_extractor(f12)
#         else:
#             att_log_logits, feature_extractor_embs = self.feature_extractor(emb)

#         embeddings.extend(feature_extractor_embs)  # Store embeddings from feature extractor

#         return att_log_logits, embeddings
    


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state):
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
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device, save_embs=True)
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
    gsat = GSAT(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config)
    metric_dict = gsat.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    parser.add_argument('--exp_type', type=str, help='cuda device id, -1 for cpu')
    parser.add_argument('--calculate_variance', type=bool, default=False)

    args = parser.parse_args()
    global dataset_name
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda
    global exp_type
    exp_type = args.exp_type

    global calculate_variance
    calculate_variance = args.calculate_variance

    global all_variances
    all_variances = {}

    global timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    global ahh
    ahh = False

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    print('exp_type', exp_type)

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    num_seeds = 1 # global_config['num_seeds'] -- HERE

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state)
        metric_dicts.append(metric_dict)

    log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()
