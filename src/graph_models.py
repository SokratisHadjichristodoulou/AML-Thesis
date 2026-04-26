import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

class GCN(torch.nn.Module):
    """
    Simple 2-layer GCN for node classification.

    x -> GCNConv -> ReLU -> Dropout -> GCNConv -> logits
    """
    def __init__(self, in_channels, hidden_channels=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits; loss applies softmax internally

def get_gcn(in_channels, hidden_channels=64, num_classes=2, dropout=0.5):
    return GCN(in_channels, hidden_channels, num_classes, dropout)

class GraphSAGE(torch.nn.Module):
    """
    2-layer GraphSAGE for node classification.

    Uses mean aggregation over neighbors, with separate transforms for
    self vs neighbor representations like GCN.
    """
    def __init__(self, in_channels, hidden_channels=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, num_classes, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def get_graphsage(in_channels, hidden_channels=64, num_classes=2, dropout=0.5):
    return GraphSAGE(in_channels, hidden_channels, num_classes, dropout)

class EvolveGCNO(torch.nn.Module):
    """
    EvolveGCN-O: GCN with weights that evolve through time via LSTM.

    Paper logic:
        W_t = LSTM(W_{t-1})
        H_t = GCN(X_t, A_t, W_t)
    """
    def __init__(self, in_channels, hidden_channels=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout

        # Initial trainable weight matrices
        self.W1 = torch.nn.Parameter(torch.empty(in_channels, hidden_channels))
        self.W2 = torch.nn.Parameter(torch.empty(hidden_channels, num_classes))
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)

        # LSTM cells for weight evolution
        self.lstm1 = torch.nn.LSTMCell(
            in_channels * hidden_channels,
            in_channels * hidden_channels,
        )
        self.lstm2 = torch.nn.LSTMCell(
            hidden_channels * num_classes,
            hidden_channels * num_classes,
        )

    def reset_weights(self, device):
        """
        Reset evolved weights and LSTM states at the start of an epoch
        or evaluation pass.
        Keep everything as 2D: (1, hidden_size)
        """
        self.W1_t = self.W1.flatten().to(device).unsqueeze(0)  # shape (1, D1)
        self.W2_t = self.W2.flatten().to(device).unsqueeze(0)  # shape (1, D2)

        self.h1 = self.W1_t.clone()
        self.c1 = torch.zeros_like(self.h1)

        self.h2 = self.W2_t.clone()
        self.c2 = torch.zeros_like(self.h2)

    def forward_step(self, x, edge_index):
        """
        One time step:
            evolve W_t from W_{t-1}
            then apply 2-layer GCN with evolved weights
        """
        # LSTMCell expects input/h/c as 1D or 2D tensors
        self.h1, self.c1 = self.lstm1(self.W1_t, (self.h1, self.c1))
        self.h2, self.c2 = self.lstm2(self.W2_t, (self.h2, self.c2))

        # Updated weights become next-step inputs
        self.W1_t = self.h1
        self.W2_t = self.h2

        W1 = self.W1_t.squeeze(0).view(self.in_channels, self.hidden_channels)
        W2 = self.W2_t.squeeze(0).view(self.hidden_channels, self.num_classes)

        # Layer 1
        h = x @ W1
        h = self._gcn_propagate(h, edge_index, x.shape[0])
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = h @ W2
        h = self._gcn_propagate(h, edge_index, x.shape[0])

        return h

    @staticmethod
    def _gcn_propagate(h, edge_index, num_nodes):
        """
        Symmetric normalized GCN propagation.
        """
        if edge_index.shape[1] == 0:
            return h

        from torch_geometric.utils import add_self_loops, degree

        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_sl

        deg = degree(row, num_nodes, dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = torch.zeros_like(h)
        out.index_add_(0, row, h[col] * norm.unsqueeze(-1))
        return out

def get_evolve_gcn(in_channels, hidden_channels=64, num_classes=2, dropout=0.5):
    return EvolveGCNO(in_channels, hidden_channels, num_classes, dropout)