import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class GNN(nn.Module):
    def __init__(self, n_iters=8, n_link_features=32, n_path_features=32, n_edge_features=32, n_path_outputs=1, dropout_rate=0.5):
        """
        Args:
          n_iters: Number of graph iterations.
          n_link_features: Number of features in the states of each link node.
          n_path_features: Number of features in the states of each path node.
          n_edge_features: Number of features in the messages sent along the edges of the graph (produced
              by the message network).
          n_path_outputs: Number of outputs produced by at each path node of the graph.
          dropout_rate: dropout rate used for dropout layers after layer1 and layer2 in the readout net
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
        self.path_msg_net = nn.Sequential(nn.Linear(self.n_link_features + self.n_path_features, 256), 
                                    nn.ReLU(),
                                    # nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    # nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, self.n_edge_features))
        self.link_msg_net = nn.Sequential(nn.Linear(self.n_link_features + self.n_path_features, 256), 
                                    nn.ReLU(),
                                    # nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    # nn.Dropout(self.dropout_rate),
                                    nn.Linear(256, self.n_edge_features))

    def forward(self, x):
        """
        Args:
          x: see the details in the dataloder
          
        Returns:
          outputs of shape (n_paths, n_path_outputs): Outputs of all the path nodes after n_iters iterations of the
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
            message_for_path = self.path_msg_net(feature_concat1)
            message_for_link = self.link_msg_net(feature_concat2)
            message_aggregation_for_path = torch.zeros(size=(n_paths, self.n_edge_features), dtype=torch.float).to(device)
            message_aggregation_for_link = torch.zeros(size=(n_links, self.n_edge_features), dtype=torch.float).to(device)
            message_aggregation_for_path = message_aggregation_for_path.index_add(0, path_indices, message_for_path)
            message_aggregation_for_link = message_aggregation_for_link.index_add(0, link_indices, message_for_link)
            _, path_hidden_states = self.path_state_update(message_aggregation_for_path.unsqueeze(1), path_states.unsqueeze(0))
            _, link_hidden_states = self.link_state_update(message_aggregation_for_link.unsqueeze(1), link_states.unsqueeze(0))
            path_states = path_hidden_states.squeeze(0)
            link_states = link_hidden_states.squeeze(0)
            
        outputs = self.readout_net(path_states)
        return outputs