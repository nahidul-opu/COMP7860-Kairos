##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import os.path as osp

import torch
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import ICEWS18
# from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage,
                                           LastAggregator)
from torch_geometric import *
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import copy
import re
import time
# Choosing CUDA to run the code if available in the host machine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
# msg structure:      [src_node_feature,edge_attr,dst_node_feature]
# Load training datasets
train_data=torch.load("/home/shahidul2k9/data/streamspot/graph/graph_0.TemporalData")
train_data2=torch.load("/home/shahidul2k9/data/streamspot/graph/graph_100.TemporalData")
train_data3=torch.load("/home/shahidul2k9/data/streamspot/graph/graph_200.TemporalData")
train_data4=torch.load("/home/shahidul2k9/data/streamspot/graph/graph_400.TemporalData")
train_data5=torch.load("/home/shahidul2k9/data/streamspot/graph/graph_500.TemporalData")
# test_data=torch.load("/home/shahidul2k9/data/streamspot/graph/graph_305.TemporalData")
val_data=torch.load("/home/shahidul2k9/data/streamspot/graph/graph_505.TemporalData")

# Utility method to merge two temporal provenance graphs a and b
def graph_merge(a,b):
    merged=a
    merged.msg=torch.cat([merged.msg,b.msg],dim=0)
    merged.src=torch.cat([merged.src,b.src],dim=0)
    merged.dst=torch.cat([merged.dst,b.dst],dim=0)
    merged.t=torch.cat([merged.t,b.t],dim=0)
    return merged
# Concat temporal provenacne graphs
train_data=graph_merge((train_data),copy.deepcopy(train_data2))
train_data=graph_merge((train_data),copy.deepcopy(train_data3))
train_data=graph_merge((train_data),copy.deepcopy(train_data4))
train_data=graph_merge((train_data),copy.deepcopy(train_data5))

# Maximum number of node across all loaded provenance graphs
max_node_num = 5045000
min_dst_idx, max_dst_idx = 0, max_node_num
# Loader of the neighbour
neighbor_loader = LastNeighborLoader(max_node_num, size=5, device=device)

# Graph attention layer configuration
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        # Transformer convolution filter
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    # Forward method is call before passing input to next layer
    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        # Convolution filter is applied
        return self.conv(x, edge_index, edge_attr)

# Configuration for predicting link or system call
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

# Dimension of node memory and time
memory_dim = time_dim = embedding_dim = 200

# Initialization of memory layer
memory = TGNMemory(
    max_node_num,
    train_data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(train_data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)
# Initialization of attention embedding layer
gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=train_data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)
# Initializing link predictor
link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
# Adam optimizer configuration for training dataset with learning rate
optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
# Batch size indicating number of elements in each batch
BATCH=1024

# Train method trains the model over the input dataset
def train():
    # Signal each layer to prepares for training modes
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    # train_loader = TemporalDataLoader(train_data, batch_size=BATCH)

    # Iterate over all the batches and train the model
    for batch in train_data.seq_batches(batch_size=BATCH):
        # Load the batch into target device(cpu/cuda)
        batch = batch.to(device)
        # Reset the gradiant of all parameters to zeroes
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),),
                                dtype=torch.long, device=device)
        #         edge_i = torch.vstack([src, pos_dst])
        #         neg_src, neg_dst = negative_sampling(edge_i)
        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        #         n_id = torch.cat([src, pos_dst, neg_src, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])

        # Predict likelihood of positive edge
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        # Predict likelihood of negative edge
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        # Calculate positive edge losses
        loss = criterion(pos_out, torch.ones_like(pos_out))
        # Add negative edge losses
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        # Computes the gradients of the loss with respect to the model's parameters
        loss.backward()
        # Updates the model parameters using the computed gradients
        optimizer.step()
        # Detaches the memory tensor from the computation graph
        memory.detach()
        #         print(z.shape)
        total_loss += float(loss) * batch.num_events
    #     print("trained_stage_data:",train_data)
    return total_loss / train_data.num_events


@torch.no_grad()
def test_new(inference_data):
    # Switch each layer from training to test modes
    memory.eval()
    gnn.eval()
    link_pred.eval()

    total_loss = 0
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    torch.manual_seed(12345)  # Ensure determi|nistic sampling across epochs.

    aps, aucs = [], []
    pos_o = []

    # test_loader = TemporalDataLoader(inference_data, batch_size=BATCH)
    # Iterate over batches of validation dataset
    for batch in inference_data.seq_batches(batch_size=BATCH):
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        # Generate negative destination
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)

        # Pass the node representations and graph information through the Graph Neural Network (GNN)
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        # Predict the likelihood of an edge existing between the source and positive destination nodes
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # Predict the likelihood of an edge existing between the source and negative destination nodes
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])
        pos_o.append(pos_out)

        # Compute the loss for positive link predictions
        loss = criterion(pos_out, torch.ones_like(pos_out))

        # Compute the loss for negative link predictions and add it to the total loss
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Accumulate the loss, weighted by the number of events in the batch
        total_loss += float(loss) * batch.num_events

        # Update memory state
        memory.update_state(src, pos_dst, t, msg)
        # Insert the edge into the neighbor loader for future neighborhood sampling
        neighbor_loader.insert(src, pos_dst)
    # Compute the average loss
    loss = total_loss / inference_data.num_events
    # Return some metrics from this evaluation
    return float(torch.tensor(aps).mean()), float(
        torch.tensor(aucs).mean()), pos_out.sigmoid().cpu(), neg_out.sigmoid().cpu(), loss


for epoch in tqdm(range(1, 11)):
    # Train model
    loss = train()
    print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # Validate current state of the training model with validation dataset
    test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(val_data)
    print(f'val_data: Loss: {loss_test:.4f}')

model = [memory, gnn, link_pred, neighbor_loader]
torch.save(model, "/home/shahidul2k9/data/streamspot/model_saved.pt")







