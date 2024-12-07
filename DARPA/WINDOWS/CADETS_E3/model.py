from kairos_utils import *  # Import utility functions for Kairos
from config import *  # Import configuration settings

# Set the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss function for classification tasks
criterion = nn.CrossEntropyLoss()

# Maximum number of nodes in the node2id table (+1 to account for indexing from 0)
max_node_num = 268243
min_dst_idx, max_dst_idx = 0, max_node_num

# Helper vector to map global node indices to local indices in the graph
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


class GraphAttentionEmbedding(torch.nn.Module):
    """
    A graph embedding module that uses Transformer-based graph attention layers
    to compute node embeddings while considering temporal and message information.
    This is the architecture for the encoder of KAIROS that uses a graph neural
    network based model to generate the edge embeddings.

        in_channels (int): The size of input node features.
        out_channels (int): The size of output node features.
        msg_dim (int): The size of message feature dimensions.
        time_enc (torch.nn.Module): A module for encoding relative time differences.

    Methods:
        forward: Perform the forward pass to compute embeddings for graph nodes.
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc  # Time encoding module

        # Dimension of edge features (message + time encoding)
        edge_dim = msg_dim + time_enc.out_channels

        # First TransformerConv layer for graph attention
        self.conv = TransformerConv(
            in_channels, out_channels, heads=8, dropout=0.0, edge_dim=edge_dim
        )

        # Second TransformerConv layer for further processing
        self.conv2 = TransformerConv(
            out_channels * 8,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.0,
            edge_dim=edge_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        """
        Forward pass for computing node embeddings.

        Args:
            x (torch.Tensor): Input node features.
            last_update (torch.Tensor): Last update timestamps for nodes.
            edge_index (torch.Tensor): Edge connections in the graph.
            t (torch.Tensor): Event timestamps.
            msg (torch.Tensor): Message features for edges.

        Returns:
            torch.Tensor: Updated node embeddings.
        """
        last_update.to(device)
        x = x.to(device)
        t = t.to(device)

        # Compute relative time differences and encode them
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))

        # Combine relative time encoding and message features as edge attributes
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        # First graph attention layer
        x = F.relu(self.conv(x, edge_index, edge_attr))

        # Second graph attention layer
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        return x


class LinkPredictor(torch.nn.Module):
    """
    A module for predicting links between pairs of nodes based on their embeddings.
    This is the decoder architecture for KAIROS that uses a multilayer perceptron to
    predict the type of edge in the form of probabilities.

        in_channels (int): The size of input node embeddings.
        out_channels (int): The number of output classes (e.g., edge types).

    Methods:
        forward: Perform the forward pass to predict links.
    """

    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()

        # Linear layers for transforming source and destination node embeddings
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        # A sequential network for link prediction
        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),  # Expand dimension
            torch.nn.Dropout(0.5),  # Add dropout for regularization
            nn.Tanh(),  # Activation function
            Linear(in_channels * 8, in_channels * 2),  # Reduce dimension
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),  # Further reduction
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels),  # Final prediction layer
        )

    def forward(self, z_src, z_dst):
        """
        Forward pass to predict the link between two nodes.

            z_src (torch.Tensor): Embedding of the source node.
            z_dst (torch.Tensor): Embedding of the destination node.

        Returns:
            torch.Tensor: Predicted probabilities for each class (edge type).
        """
        # Concatenate transformed source and destination embeddings
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)

        # Pass through the sequential network for prediction
        h = self.lin_seq(h)

        return h


def cal_pos_edges_loss_multiclass(link_pred_ratio, labels):
    """
    Calculate the loss for positive edges in a multiclass classification task.

        link_pred_ratio (torch.Tensor): Predicted probabilities for edge types.
        labels (torch.Tensor): True labels for the edges.

    Returns:
        torch.Tensor: Loss for each edge.
    """
    loss = []

    # Compute loss for each edge individually
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1, -1), labels[i].reshape(-1)))

    # Convert list of losses to a tensor
    return torch.tensor(loss)
