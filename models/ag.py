# Файл: models/ag.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
import joblib
from config.registries import register_model
from core.interfaces import ClusterModel
from typing import Dict, Any


class DMoN_DPR(nn.Module):
    """Deep Modularity Network with Diversity-Preserving Regularization"""
    def __init__(self, input_dim, num_clusters, hidden_dim, lambda_, dropout):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv(input_dim, hidden_dim, add_self_loops=None, normalize=True), 
            GCNConv(hidden_dim, hidden_dim//2, add_self_loops=None, normalize=True),
            GCNConv(hidden_dim//2, num_clusters, add_self_loops=None, normalize=True)
        ])
        self.skip_proj = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim//2) if hidden_dim != (hidden_dim//2) else nn.Identity(),
            nn.Identity()
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Identity()
        ])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for i in self.gcn_layers[:-1]])
        self.lambda_ = lambda_
        self.num_clusters = num_clusters
        
    def forward(self, x, edge_index, edge_weight):
        x_res = x
        for i, (conv, proj, bn) in enumerate(zip(self.gcn_layers, self.skip_proj, self.bn_layers)):
            x_conv = conv(x, edge_index, edge_weight=edge_weight)
            #x_conv = bn(x_conv)
            if i < len(self.gcn_layers)-1:
                x = F.selu(x_conv + proj(x_res))
                x = self.dropout_layers[i](x)
                x_res = x
            else:
                x = F.selu(x_conv)
        return F.softmax(x, dim=-1)
    
    def compute_loss(self, c, edge_index, edge_weight, x):
        losses = {
            'modularity': self.modularity_loss(c, edge_index, edge_weight),
            'collapse': self.collapse_regularization(c),
            'distance': self.distance_regularization(c, x),
            'variance': self.variance_regularization(c),
            'entropy': self.entropy_regularization(c)
        }
        return sum(self.lambda_.get(k, 0) * losses[k] for k in losses)
        

    def modularity_loss(self, c, edge_index, edge_weight):
        adj = torch.sparse_coo_tensor(
            edge_index, 
            edge_weight,
            size=(c.size(0), c.size(0))
        ).coalesce().to_dense()
        m = adj.sum()
        if m < 1e-16:
            return torch.tensor(0.0, device=c.device)
        deg = adj.sum(axis=1).unsqueeze(1)
        B = adj - (deg @ deg.t()) / (2 * m + 1e-8)
        return -torch.trace(c.t() @ B @ c) / (2 * m + 1e-8)

    def collapse_regularization(self, c):
        self.norm = torch.sqrt(torch.tensor(self.num_clusters, device=c.device)) / c.size(0)
        return self.norm * torch.norm(c.sum(dim=0), p = 'fro') - 1

    def distance_regularization(self, c, x):
        centroids = c.t() @ x
        dists = F.tanh(torch.cdist(centroids, centroids))
        return -dists[~torch.eye(c.size(1), dtype=bool, device=c.device)].mean()

    def variance_regularization(self, c):
        return -c.var(dim=0).mean()

    def entropy_regularization(self, c):
        entropy = -torch.sum(c * torch.log(c + 1e-10))
        return entropy / c.size(0)


@register_model(
    name='dmon',
    params_help={
        'num_clusters': 'Number of target clusters (positive integer)',
        'hidden_dim': 'GCN hidden dimension size (positive integer)',
        'lambda_': 'Loss coefficients dict {modularity: float, collapse: float, distance: float, variance: float, entropy: float}',
        'epochs': 'Training iterations (positive integer)',
        'lr': 'Learning rate (positive float)',
        'dropout': 'Dropout probability (0.0-1.0)'
    }
)
class DMoN(ClusterModel):
    """Deep Modularity Network (DMoN)"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.labels_ = None

    def fit(self, data_loader):
        features, adj_matrix = data_loader.full_data()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_tensor = torch.FloatTensor(features.values).to(self.device)
        adj_tensor = torch.FloatTensor(adj_matrix.values).to(self.device)
        edge_index, edge_attr = dense_to_sparse(adj_tensor)
        self.params['input_dim'] = features.shape[1]
        
        self.model = DMoN_DPR(
            input_dim=self.params.get('input_dim'),
            num_clusters=self.params.get('num_clusters'),
            hidden_dim=self.params.get('hidden_dim', 256),
            lambda_=self.params.get('lambda_', {'modularity': 1.0, 'collapse': 1.0, 
                                                'distance':0.0, 'variance': 0.0, 'entropy':0.0}),
            dropout=self.params.get('dropout', 0.5)
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.get('lr', 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                      base_lr=self.params.get('lr', 1e-4), max_lr=0.01, 
                                                      step_size_up=150, mode='triangular2')
        
        for epoch in range(self.params.get('epochs', 200)):
            self.model.train()
            optimizer.zero_grad()
            
            c = self.model(features_tensor, edge_index, edge_attr)
            loss = self.model.compute_loss(c, edge_index, edge_attr, features_tensor)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.4f}")

        self.model.eval()
        with torch.no_grad():
            c = self.model(features_tensor, edge_index, edge_attr)
            self.labels_ = c.argmax(dim=1).cpu().numpy()
            
        del features_tensor, adj_tensor, edge_index, edge_attr, features, adj_matrix
        return None

    def predict(self, data_loader):
        features, adj_matrix = data_loader.full_data()
        features_tensor = torch.FloatTensor(features.values).to(self.device)
        adj_tensor = torch.FloatTensor(adj_matrix.values).to(self.device)
        edge_index, edge_attr = dense_to_sparse(adj_tensor)
        
        self.model.eval()
        with torch.no_grad():
            c = self.model(features_tensor, edge_index, edge_attr)
            preds = c.argmax(dim=1).cpu().numpy()
        return preds

    def save(self, path: str) -> None:
        torch.save({
            'model_state': self.model.state_dict(),
            'params': self.params,
            'labels': self.labels_
        }, path, _use_new_zipfile_serialization=True, pickle_protocol=4)

    @classmethod
    def load(cls, path: str) -> 'DMoN':
        data = torch.load(path, map_location=torch.device('cpu'))
        
        model = cls(data['params'])
        model.model = DMoN_DPR(**data['params'])
        
        model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.model.load_state_dict(data['model_state'])
        model.model.to(model.device)
        
        model.labels_ = data['labels']
        return model

    @property
    def model_data(self) -> dict:
        return {}