# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from normality_normalization import *

from .conv_layers import GINConv, GINEConv


class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, save_embs=False):
        super().__init__()
        self.save_embs = save_embs
        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        self.embeddings_dict = {}

        if model_config.get('atom_encoder', False):
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for i in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                conv = GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size)
            else:
                conv = GINConv(GIN.MLP(hidden_size, hidden_size))
            self.convs.append(conv)

        
        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        if self.save_embs:
            self.embeddings_dict['pre_fc_out_unpooled'] = x.clone().detach()
            self.embeddings_dict['pre_fc_out_pooled'] = self.pool(x.clone().detach(), batch)

        return self.fc_out(self.pool(x, batch)), self.pool(x, batch)

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        if self.save_embs:
            self.embeddings_dict['layer_0'] = x.clone().detach()
            self.embeddings_dict['layer_0_pooled'] = self.pool(x.clone().detach(), batch)




        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
    
    def save_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = output.detach()
        return hook

    def get_all_embeddings(self):
        """ Returns all stored embeddings for Mutual Information computation. """
        return self.embeddings_dict
    
    def save_pre_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = input[0].detach()
        return hook

class GIN_norm(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, save_embs=False):
        super().__init__()
        self.save_embs = save_embs
        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        self.embeddings_dict = {}

        if model_config.get('atom_encoder', False):
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool
        self.norms = nn.ModuleList([BatchNormalNorm1d(hidden_size) for _ in range(self.n_layers)])

        
        for i in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                conv = GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size)
            else:
                conv = GINConv(GIN.MLP(hidden_size, hidden_size))
            self.convs.append(conv)

        
        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
        if self.save_embs:
            self.fc_out.register_forward_hook(self.save_pre_activation('pre_fc_out'))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.norms[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            
        if self.save_embs:
            self.embeddings_dict['pre_fc_out_unpooled'] = x.clone().detach()
            
        return self.fc_out(self.pool(x, batch)), self.pool(x, batch)

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.norms[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        if self.save_embs:
            self.embeddings_dict['layer_0'] = self.pool(x,batch).clone().detach()

        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
    
    def save_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = output.detach()
        return hook

    def get_all_embeddings(self):
        """ Returns all stored embeddings for Mutual Information computation. """
        return self.embeddings_dict
    
    def save_pre_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = input[0].detach()
        return hook
    

# class GIN_with_fc(nn.Module):
#     def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
#         super().__init__()

#         self.n_layers = model_config['n_layers']
#         hidden_size = model_config['hidden_size']
#         self.edge_attr_dim = edge_attr_dim
#         self.dropout_p = model_config['dropout_p']
#         self.use_edge_attr = model_config.get('use_edge_attr', True)

#         self.embeddings_dict = {}

#         # Node and Edge Encoders
#         self.node_encoder = Linear(x_dim, hidden_size)
#         if edge_attr_dim != 0 and self.use_edge_attr:
#             self.edge_encoder = Linear(edge_attr_dim, hidden_size)

#         # GNN Convolutional Layers
#         self.convs = nn.ModuleList()
#         self.relu = nn.ReLU()
#         self.pool = global_add_pool  # Summing over nodes per graph

#         for i in range(self.n_layers):
#             if edge_attr_dim != 0 and self.use_edge_attr:
#                 self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
#             else:
#                 self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

#             # Register hook to store embeddings
#             #self.convs[i].register_forward_hook(self.save_activation(f'conv_{i}'))

#         # Fully Connected Layers (Hardcoded)
#         self.fcs = nn.Sequential(
#             nn.Linear(hidden_size, 48),  
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_p),
#             nn.Linear(48, 32), 
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_p),
#             nn.Linear(32, 16), 
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_p),
#         )

#         self.fc_out = nn.Linear(64, 1 if num_class == 2 and not multi_label else num_class)  

#         self.fc_out.register_forward_hook(self.save_pre_activation('pre_fc_out'))

#     def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
#         x = self.node_encoder(x)
#         if edge_attr is not None and self.use_edge_attr:
#             edge_attr = self.edge_encoder(edge_attr)

#         for i in range(self.n_layers):
#             x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
#             x = self.relu(x)
#             x = F.dropout(x, p=self.dropout_p, training=self.training)
        


#         x = self.pool(x, batch)  # Graph-level representation

#         self.embeddings_dict['x_after_ext'] = self.pool(x,batch).clone().detach()
#         x = self.fcs(x)  # Pass through fully connected layers
#         x_out = self.fc_out(x)
        
#         return x_out, x

#     @staticmethod
#     def MLP(in_channels: int, out_channels: int):
#         return nn.Sequential(
#             Linear(in_channels, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             Linear(out_channels, out_channels),
#         )

#     def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
#         """ Extracts node embeddings before pooling. """
#         x = self.node_encoder(x)
#         if edge_attr is not None and self.use_edge_attr:
#             edge_attr = self.edge_encoder(edge_attr)

#         for i in range(self.n_layers):
#             x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
#             x = self.relu(x)
#             x = F.dropout(x, p=self.dropout_p, training=self.training)
        
#         self.embeddings_dict['x_before_ext'] = self.pool(x,batch).clone().detach()

#         return x  # Node-level embeddings

#     def get_pred_from_emb(self, emb, batch):
#         """ Gets predictions from graph-level embeddings. """
#         x = self.pool(emb, batch)  # Aggregate node embeddings to graph level
#         x = self.fcs(x)  # Pass through fully connected layers
#         return x
    
#     def get_pred_from_emb_linear(self, emb):
#         """ Gets predictions from linear layer embeddings. """
#         # print("Shape of emb before passing to self.fcs:", emb.shape)
#         x = self.fcs(emb)  # Pass through fully connected layers
#         return x

#     def save_activation(self, name):
#         """ Hook function to store activations for MI computation. """
#         def hook(module, input, output):
#             self.embeddings_dict[name] = output.detach()
#         return hook

#     def get_all_embeddings(self):
#         """ Returns all stored embeddings for Mutual Information computation. """
#         return self.embeddings_dict
    

class GIN_with_fc_extractor(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, save_embs=False):
        super().__init__()
        self.save_embs = save_embs
        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        self.embeddings_dict = {}

        # Node and Edge Encoders
        self.node_encoder = Linear(x_dim, hidden_size)
        if edge_attr_dim != 0 and self.use_edge_attr:
            self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        # GNN Convolutional Layers
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool  # Summing over nodes per graph

        for i in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

            # Register hook to store embeddings
            #self.convs[i].register_forward_hook(self.save_activation(f'conv_{i}'))

        # Fully Connected Layers (Hardcoded)
        
        self.bottleneck_fc = nn.Sequential(
            nn.Linear(hidden_size, 48),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        )
        
        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
        
        if self.save_embs:
            cnt = 0
            for i, layer in enumerate(self.bottleneck_fc):
                if isinstance(layer, nn.Linear):
                    layer.register_forward_hook(self.save_pre_activation(f'layer_{cnt}'))
                    cnt += 1
            self.bottleneck_fc[-1].register_forward_hook(self.save_activation(f'layer_{cnt}'))
            self.fc_out.register_forward_hook(self.save_pre_activation('pre_fc_out'))
        

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        if self.save_embs:
            self.embeddings_dict['x_after_ext'] = self.pool(x,batch).clone().detach()
            self.embeddings_dict['pre_fc_out_pooled'] = self.pool(x.clone().detach(), batch)
            self.embeddings_dict['pre_fc_out_unpooled'] = x.clone().detach()

        return self.fc_out(self.pool(x, batch)), self.pool(x, batch)

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        """ Extracts node embeddings before pooling. """
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        if self.save_embs:
            self.embeddings_dict['x_before_ext'] = self.pool(x,batch).clone().detach()
            self.embeddings_dict['layer_0_pooled'] = self.pool(x.clone().detach(), batch)
            
    
        return self.bottleneck_fc(x)  # Node-level embeddings

    def get_pred_from_emb(self, emb, batch):
        """ Gets predictions from graph-level embeddings. """
        x = self.pool(emb, batch)  # Aggregate node embeddings to graph level
        x = self.fcs(x)  # Pass through fully connected layers
        return x
    
    def get_pred_from_emb_linear(self, emb):
        """ Gets predictions from linear layer embeddings. """
        # print("Shape of emb before passing to self.fcs:", emb.shape)
        x = self.fcs(emb)  # Pass through fully connected layers
        return x

    def save_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = output.detach()
        return hook

    def save_pre_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = input[0].detach()
        return hook
    
    def get_all_embeddings(self):
        """ Returns all stored embeddings for Mutual Information computation. """
        return self.embeddings_dict

    def freeze_encoder(self):
        # Freeze node encoder
        for param in self.node_encoder.parameters():
            param.requires_grad = False

        # Freeze edge encoder if it exists and is used
        if self.edge_attr_dim != 0 and self.use_edge_attr:
            for param in self.edge_encoder.parameters():
                param.requires_grad = False

        # Freeze all GNN convolution layers
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

        # Freeze bottleneck FC layers
        for param in self.bottleneck_fc.parameters():
            param.requires_grad = False

class GIN_with_fc_extractor_norm(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, save_embs=False):
        super().__init__()
        self.save_embs = save_embs
        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        self.embeddings_dict = {}

        self.norms = nn.ModuleList([BatchNormalNorm1d(hidden_size) for _ in range(self.n_layers)])

        # Node and Edge Encoders
        self.node_encoder = Linear(x_dim, hidden_size)
        if edge_attr_dim != 0 and self.use_edge_attr:
            self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        # GNN Convolutional Layers
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool  # Summing over nodes per graph

        for i in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

            # Register hook to store embeddings
            #self.convs[i].register_forward_hook(self.save_activation(f'conv_{i}'))

        # Fully Connected Layers (Hardcoded)
        
        self.norms_bottl = nn.ModuleList([BatchNormalNorm1d(hidden_size) for _ in range(self.n_layers)])
        self.bottleneck_fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            BatchNormalNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(16, 64),
            BatchNormalNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        )
        if save_embs:
            cnt = 0
            for i, layer in enumerate(self.bottleneck_fc):
                if isinstance(layer, nn.Linear):
                    layer.register_forward_hook(self.save_pre_activation(f'layer_{cnt}'))
                    cnt += 1
                self.bottleneck_fc[-1].register_forward_hook(self.save_activation(f'layer_{cnt}'))

                self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
                self.fc_out.register_forward_hook(self.save_pre_activation('pre_fc_out'))
        

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            #x = self.norms[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        if self.save_embs:
            self.embeddings_dict['x_after_ext'] = self.pool(x,batch).clone().detach()

            self.embeddings_dict['pre_fc_out_unpooled'] = x.clone().detach()

        return self.fc_out(self.pool(x, batch)), self.pool(x, batch)

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        """ Extracts node embeddings before pooling. """
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            #x = self.norms[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        if self.save_embs:
            self.embeddings_dict['x_before_ext'] = self.pool(x,batch).clone().detach()
    
        return self.bottleneck_fc(x)  # Node-level embeddings

    def get_pred_from_emb(self, emb, batch):
        """ Gets predictions from graph-level embeddings. """
        x = self.pool(emb, batch)  # Aggregate node embeddings to graph level
        x = self.fcs(x)  # Pass through fully connected layers
        return x
    
    def get_pred_from_emb_linear(self, emb):
        """ Gets predictions from linear layer embeddings. """
        # print("Shape of emb before passing to self.fcs:", emb.shape)
        x = self.fcs(emb)  # Pass through fully connected layers
        return x

    def save_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = output.detach()
        return hook

    def save_pre_activation(self, name):
        """ Hook function to store activations for MI computation. """
        def hook(module, input, output):
            self.embeddings_dict[name] = input[0].detach()
        return hook
    
    def get_all_embeddings(self):
        """ Returns all stored embeddings for Mutual Information computation. """
        return self.embeddings_dict

    def freeze_encoder(self):
        # Freeze node encoder
        for param in self.node_encoder.parameters():
            param.requires_grad = False

        # Freeze edge encoder if it exists and is used
        if self.edge_attr_dim != 0 and self.use_edge_attr:
            for param in self.edge_encoder.parameters():
                param.requires_grad = False

        # Freeze all GNN convolution layers
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

        # Freeze bottleneck FC layers
        for param in self.bottleneck_fc.parameters():
            param.requires_grad = False

# class GIN_with_fc_both(nn.Module):
#     def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
#         super().__init__()

#         self.n_layers = model_config['n_layers']
#         hidden_size = model_config['hidden_size']
#         self.edge_attr_dim = edge_attr_dim
#         self.dropout_p = model_config['dropout_p']
#         self.use_edge_attr = model_config.get('use_edge_attr', True)

#         self.embeddings_dict = {}

#         # Node and Edge Encoders
#         self.node_encoder = Linear(x_dim, hidden_size)
#         if edge_attr_dim != 0 and self.use_edge_attr:
#             self.edge_encoder = Linear(edge_attr_dim, hidden_size)

#         # GNN Convolutional Layers
#         self.convs = nn.ModuleList()
#         self.relu = nn.ReLU()
#         self.pool = global_add_pool  # Summing over nodes per graph

#         for i in range(self.n_layers):
#             if edge_attr_dim != 0 and self.use_edge_attr:
#                 self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
#             else:
#                 self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

#             # Register hook to store embeddings
#             #self.convs[i].register_forward_hook(self.save_activation(f'conv_{i}'))

#         # Fully Connected Layers (Hardcoded)
#         self.bottleneck_fc = nn.Sequential(
#             nn.Linear(hidden_size, 48),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_p),
#             nn.Linear(48, 32),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_p),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_p),
#         )

#         self.fc_out = nn.Sequential(nn.Linear(16, 1 if num_class == 2 and not multi_label else num_class))

#         self.bottleneck_fc[1].register_forward_hook(self.save_activation('fc_0'))
#         self.bottleneck_fc[-3].register_forward_hook(self.save_activation('fc_last'))
#         self.fc_out.register_forward_hook(self.save_activation('pre_fc_out'))

#     def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
#         x = self.node_encoder(x)
#         if edge_attr is not None and self.use_edge_attr:
#             edge_attr = self.edge_encoder(edge_attr)

#         for i in range(self.n_layers):
#             x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
#             x = self.relu(x)
#             x = F.dropout(x, p=self.dropout_p, training=self.training)

#         x = self.bottleneck_fc(x)
#         return self.fc_out(self.pool(x, batch)), self.pool(x, batch)

#     @staticmethod
#     def MLP(in_channels: int, out_channels: int):
#         return nn.Sequential(
#             Linear(in_channels, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             Linear(out_channels, out_channels),
#         )

#     def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
#         """ Extracts node embeddings before pooling. """
#         x = self.node_encoder(x)
#         if edge_attr is not None and self.use_edge_attr:
#             edge_attr = self.edge_encoder(edge_attr)

#         for i in range(self.n_layers):
#             x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
#             x = self.relu(x)
#             x = F.dropout(x, p=self.dropout_p, training=self.training)
    
#         return x  # Node-level embeddings

#     def get_pred_from_emb(self, emb, batch):
#         """ Gets predictions from graph-level embeddings. """
#         x = self.pool(emb, batch)  # Aggregate node embeddings to graph level
#         x = self.fcs(x)  # Pass through fully connected layers
#         return x
    
#     def get_pred_from_emb_linear(self, emb):
#         """ Gets predictions from linear layer embeddings. """
#         # print("Shape of emb before passing to self.fcs:", emb.shape)
#         x = self.fcs(emb)  # Pass through fully connected layers
#         return x

#     def save_activation(self, name):
#         """ Hook function to store activations for MI computation. """
#         def hook(module, input, output):
#             self.embeddings_dict[name] = output.detach()
#         return hook

#     def get_all_embeddings(self):
#         """ Returns all stored embeddings for Mutual Information computation. """
#         return self.embeddings_dict
    
#     def freeze_encoder(self):
#         # Freeze node encoder
#         for param in self.node_encoder.parameters():
#             param.requires_grad = False

#         # Freeze edge encoder if it exists and is used
#         if self.edge_attr_dim != 0 and self.use_edge_attr:
#             for param in self.edge_encoder.parameters():
#                 param.requires_grad = False

#         # Freeze all GNN convolution layers
#         for conv in self.convs:
#             for param in conv.parameters():
#                 param.requires_grad = False

#         # Freeze bottleneck FC layers
#         for param in self.bottleneck_fc.parameters():
#             param.requires_grad = False
