##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging
import os
from tqdm import tqdm
import torch
from kairos_utils import *  # Custom utility functions
from config import *  # Configuration variables
from model import *  # Model definitions

# Setting up logging
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)

# Define path for training log file
artifact_path = os.path.join(artifact_dir, "training.log")
file_handler = logging.FileHandler(artifact_path)
file_handler.setLevel(logging.INFO)

# Define logging format
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train(train_data, memory, gnn, link_pred, optimizer, neighbor_loader):
    """
    Perform one epoch of training on the provided temporal graph data.

    train_data: TemporalData, the dataset for training.
    memory: TGNMemory, memory module for temporal graph networks. (Gated Recurrent Unit [GRU] for State Update)
    gnn: GraphAttentionEmbedding, graph neural network model. (Encoder)
    link_pred: LinkPredictor, link prediction model. (Decoder)
    optimizer: torch.optim.Optimizer, optimizer for training.
    neighbor_loader: LastNeighborLoader, loader for neighbor nodes. (Required for obtaining the state of neighbor nodes)
    return: float, average loss for the training epoch.
    """
    memory.train()  # Set memory module to training mode
    gnn.train()  # Set GNN module to training mode
    link_pred.train()  # Set link predictor to training mode

    # Reset states for memory and neighbor loader
    memory.reset_state()
    neighbor_loader.reset_state()

    total_loss = 0  # Accumulate total loss over all batches

    # Iterate through batches of temporal data
    for batch in train_data.seq_batches(batch_size=BATCH):
        optimizer.zero_grad()  # Clear gradients from previous steps

        # Extract batch data: source nodes, destination nodes, timestamps, and messages
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Get unique nodes involved in this batch
        n_id = torch.cat([src, pos_dst]).unique()

        # Retrieve edges and associated indices for these nodes
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Update memory and compute new embeddings
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])

        # Compute positive link predictions
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # Prepare labels for loss computation
        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        this_true = y_true.reshape(-1).to(torch.long).to(device=device)

        # Compute loss
        loss = criterion(y_pred, y_true)

        # Update memory and neighbor loader with ground-truth states
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Detach memory to avoid retaining gradients
        memory.detach()

        # Accumulate total loss
        total_loss += float(loss) * batch.num_events

    # Return average loss over the entire dataset
    return total_loss / train_data.num_events


def load_train_data():

    # Load temporal graph data from disk and move to device (e.g., GPU)
    graph_4_2 = torch.load(
        os.path.join(graphs_dir, "graph_4_2.TemporalData.simple")
    ).to(device=device)
    graph_4_3 = torch.load(
        os.path.join(graphs_dir, "graph_4_3.TemporalData.simple")
    ).to(device=device)
    graph_4_4 = torch.load(
        os.path.join(graphs_dir, "graph_4_4.TemporalData.simple")
    ).to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4]


def init_models(node_feat_size):

    # Initialize TGN memory module
    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        node_state_dim,
        time_dim,
        message_module=IdentityMessage(node_feat_size, node_state_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    # Initialize graph neural network
    gnn = GraphAttentionEmbedding(
        in_channels=node_state_dim,
        out_channels=edge_dim,
        msg_dim=node_feat_size,
        time_enc=memory.time_enc,
    ).to(device)

    # Initialize link predictor
    out_channels = len(include_edge_type)
    link_pred = LinkPredictor(in_channels=edge_dim, out_channels=out_channels).to(
        device
    )

    # Setup optimizer for all model parameters
    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
    )

    # Initialize neighbor loader
    neighbor_loader = LastNeighborLoader(
        max_node_num, size=neighbor_size, device=device
    )

    return memory, gnn, link_pred, optimizer, neighbor_loader


if __name__ == "__main__":
    logger.info("Start logging.")

    # Load data for training
    train_data = load_train_data()

    # Initialize the models and the optimizer
    node_feat_size = train_data[0].msg.size(-1)
    memory, gnn, link_pred, optimizer, neighbor_loader = init_models(
        node_feat_size=node_feat_size
    )

    # Train the model
    for epoch in tqdm(range(1, epoch_num + 1)):
        for g in train_data:
            loss = train(
                train_data=g,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader,
            )
            logger.info(f"  Epoch: {epoch:02d}, Loss: {loss:.4f}")

    # Save the trained model
    model = [memory, gnn, link_pred, neighbor_loader]

    models_path = os.path.join(models_dir)
    os.makedirs(models_path, exist_ok=True)
    torch.save(model, os.path.join(models_path, "models.pt"))
