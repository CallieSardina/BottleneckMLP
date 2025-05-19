import torch.nn as nn
from models.GCN import GCNNet
from models.GAT import GATNet
from models.GIN import GINNet
from models.GIN_bottleneckMLP import GINNet_bottleneckMLP

__all__ = ['GnnNets']

class EmbeddingsExtractor:
    def __init__(self, model):
        self.model = model
        self.embeddings = {}
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.embeddings[name] = output.detach()
        return hook

    def _register_hooks(self):
        # Only hook the Linear layers inside model.node_mlp
        for name, layer in self.model.node_mlp.named_modules():
            if isinstance(layer, nn.Linear):
                hook = layer.register_forward_hook(self._hook_fn(f"mlp_linear_{name}"))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_embeddings(self):
        return self.embeddings

# # HERE
# class EmbeddingsExtractor:
#     def __init__(self, model):
#         self.model = model
#         self.embeddings = {}
#         self.hooks = []
#         self._register_hooks()

#     def _hook_fn(self, name):
#         def hook(module, input, output):
#             self.embeddings[name] = output.detach()
#         return hook

#     def _register_hooks(self):
#         # # Hook GIN layers
#         # for i, layer in enumerate(self.model.gnn_layers):
#         #     self.hooks.append(layer.register_forward_hook(self._hook_fn(f"gnn_layer_{i}")))

#         # Hook fully connected layers
#         for name, layer in self.model.named_modules():
#             if isinstance(layer, nn.Linear):
#                 hook = layer.register_forward_hook(self._hook_fn(f"linear_{name}"))
#                 self.hooks.append(hook)

#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()

#     def get_embeddings(self):
#         return self.embeddings
    
def get_model(input_dim, output_dim, model_args):
    if model_args.model_name.lower() == 'gcn':
        return GCNNet(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gat':
        return GATNet(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gin':
        return GINNet(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gin_new':
        print("Using model gnnNet_bottleneckMLP")
        return GINNet_bottleneckMLP(input_dim, output_dim, model_args)
    else:
        raise NotImplementedError(f"Model '{model_args.model_name}' is not supported.")



class GnnBase(nn.Module):
    def __init__(self):
        super(GnnBase, self).__init__()

    def forward(self, data):
        data = data.to(self.device)
        logits, prob, emb1, emb2, min_distances = self.model(data)
        return logits, prob, emb1, emb2, min_distances

    def update_state_dict(self, state_dict):
        original_state_dict = self.state_dict()
        loaded_state_dict = dict()
        for k, v in state_dict.items():
            if k in original_state_dict.keys():
                loaded_state_dict[k] = v
        self.load_state_dict(loaded_state_dict)

    def to_device(self):
        self.to(self.device)

    def save_state_dict(self):
        pass


class GnnNets(GnnBase):
    def __init__(self, input_dim, output_dim, model_args):
        super(GnnNets, self).__init__()
        self.model = get_model(input_dim, output_dim, model_args)
        self.device = 'cpu' # model_args.device
        if model_args.model_name.lower() == 'gin':
            self.model_name = 'gin'
        else:
            self.model_name = 'other'

    def forward(self, data, merge=True, test_without_mlp=False):
        data = data.to(self.device)
        if self.model_name == 'gin-old':
            logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, sim_matrix, min_distance, topk_node_index, bottomk_node_index = self.model(data, merge=True)
        else:
            logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, sim_matrix, min_distance, topk_node_index, bottomk_node_index, mlp_embeddings_list  = self.model(data, merge=True)

        if test_without_mlp:
            logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, sim_matrix, min_distance, topk_node_index, bottomk_node_index, mlp_embeddings_list  = self.model(data, merge=True, test_without_mlp=True)
        if self.model_name == 'gin-old':
            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, sim_matrix, min_distance, topk_node_index, bottomk_node_index
        else:
            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, sim_matrix, min_distance, topk_node_index, bottomk_node_index, mlp_embeddings_list 
    
