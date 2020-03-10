import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel

class GNN(nn.Module):
    def __init__(self, n_iters=8, n_link_features=32, n_path_features=32, n_edge_features=32, n_path_outputs=1, dropout_rate=0.5):
        """
        Args:
          n_iters: Number of graph iterations.
          n_node_features: Number of features in the states of each node.
          n_node_inputs: Number of inputs to each graph node (on each graph iteration).
          n_edge_features: Number of features in the messages sent along the edges of the graph (produced
              by the message network).
          n_node_outputs: Number of outputs produced by at each node of the graph.
        """
        super(GNN, self).__init__()
        self.n_iters = n_iters
        self.n_link_features = n_link_features
        self.n_path_features = n_path_features
        self.n_edge_features = n_edge_features
        self.n_path_outputs = n_path_outputs 
        self.dropout_rate = dropout_rate
        self.link_state_update = nn.GRU(input_size=self.n_edge_features, hidden_size=self.n_link_features, batch_first=True)
        self.path_state_update = nn.GRU(input_size=self.n_edge_features, hidden_size=self.n_path_features, batch_first=True)
        self.readout_net = nn.Sequential(nn.Linear(self.n_path_features, 256), 
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, n_path_outputs))
        self.msg_net = nn.Sequential(nn.Linear(self.n_link_features + self.n_path_features, 256), 
                                    nn.ReLU(),
                                    # nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    # nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, self.n_edge_features))

    def forward(self, x):
        """
        Args:
          node_inputs of shape (n_nodes, n_node_inputs): Tensor of inputs to every node of the graph.
          src_ids of shape (n_edges): Indices of source nodes of every edge.
          dst_ids of shape (n_edges): Indices of destination nodes of every edge.
          
        Returns:
          outputs of shape (n_iters, n_nodes, n_node_outputs): Outputs of all the nodes at every iteration of the
              graph neural network.
        """
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        TM, link_capacity, link_indices, path_indices, sequ_indices, n_paths, n_links, n_total, paths = x
        n_links = torch.sum(n_links)
        n_paths = torch.sum(n_paths)
        link_states = torch.zeros(size=(n_links, self.n_link_features), dtype=torch.float).to(device)
        link_states[:, 0] = link_capacity
        path_states = torch.zeros(size=(n_paths, self.n_path_features), dtype=torch.float).to(device)
        path_states[:, 0] = TM.squeeze()
        
        for _ in range(self.n_iters):
            feature_concat1 = torch.cat([link_states[link_indices, :], path_states[path_indices, :]], dim=1)
            feature_concat2 = torch.cat([path_states[path_indices, :], link_states[link_indices, :]], dim=1)
            message = self.msg_net(torch.cat([feature_concat1, feature_concat2], dim=0))
            message_aggregation_for_path = torch.zeros(size=(n_paths, self.n_edge_features), dtype=torch.float).to(device)
            message_aggregation_for_link = torch.zeros(size=(n_links, self.n_edge_features), dtype=torch.float).to(device)
            message_aggregation_for_path = message_aggregation_for_path.index_add(0, path_indices, message[: link_indices.shape[0]])
            message_aggregation_for_link = message_aggregation_for_link.index_add(0, link_indices, message[link_indices.shape[0]:])
            _, path_hidden_states = self.path_state_update(message_aggregation_for_path.unsqueeze(1), path_states.unsqueeze(0))
            _, link_hidden_states = self.link_state_update(message_aggregation_for_link.unsqueeze(1), link_states.unsqueeze(0))
            path_states = path_hidden_states.squeeze(0)
            link_states = link_hidden_states.squeeze(0)
            
        outputs = self.readout_net(path_states)
        return outputs