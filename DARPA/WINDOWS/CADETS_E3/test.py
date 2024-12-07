##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging
import os
import time
import torch

from kairos_utils import *  # Importing custom utility functions
from config import *  # Importing configuration settings
from model import *  # Importing the model definition

# Set up logging to track the progress of the reconstruction process
logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)

# Log output will be written to a file named 'reconstruction.log' in the artifact directory
file_handler = logging.FileHandler(os.path.join(artifact_dir, "reconstruction.log"))
file_handler.setLevel(logging.INFO)

# Define the format for log messages
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Function to sanitize filenames by replacing invalid characters with '_'
def sanitize_filename(filename):
    """
    Sanitize the filename by replacing invalid characters with '_'.
    Args:
        filename (str): Original filename.
    Returns:
        str: Sanitized filename.
    """
    return filename.translate(str.maketrans({":": "_", "\\": "_", "/": "_"}))


@torch.no_grad()
def test(inference_data, memory, gnn, link_pred, neighbor_loader, nodeid2msg, path):
    """
    Test the model by reconstructing edges from the graph data and saving results.
        inference_data: Temporal graph data for inference.
        memory: TGNMemory, memory module for temporal graph networks. (Gated Recurrent Unit [GRU] for State Update)
        gnn: GraphAttentionEmbedding, graph neural network model. (Encoder)
        link_pred: LinkPredictor, link prediction model. (Decoder)
        neighbor_loader: LastNeighborLoader, loader for neighbor nodes. (Required for obtaining the state of neighbor nodes)
        nodeid2msg: Mapping from node IDs to messages.
        path: Path to save the reconstruction results.
    Returns:
        dict: Loss statistics for each time interval.
    """
    if not os.path.exists(path):
        os.makedirs(path)  # Create the output directory if it does not exist

    # Set the model to evaluation mode
    memory.eval()
    gnn.eval()
    link_pred.eval()

    # Initialize memory and neighbor loader states
    memory.reset_state()
    neighbor_loader.reset_state()

    # Data structures to hold results and statistics
    time_with_loss = {}
    total_loss = 0
    edge_list = []
    unique_nodes = torch.tensor([]).to(device=device)  # Track unique nodes
    total_edges = 0

    start_time = inference_data.t[0]  # Initial timestamp in the graph
    event_count = 0  # Count the number of events processed
    pos_o = []  # Stores positive link prediction results

    # Start timing the evaluation process
    start = time.perf_counter()

    # Process the graph data in batches
    for batch in inference_data.seq_batches(batch_size=BATCH):
        # Extract data from the batch
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat(
            [unique_nodes, src, pos_dst]
        ).unique()  # Update unique nodes
        total_edges += BATCH

        # Load neighbors and update association mapping
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Update memory and compute node embeddings
        z, last_update = memory(n_id)
        z = gnn(
            z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id]
        )

        # Perform link prediction for the current batch
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        pos_o.append(pos_out)

        # Calculate true labels for the edges in the batch
        y_pred = torch.cat([pos_out], dim=0)
        y_true = [
            tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1 for m in msg
        ]
        y_true = torch.tensor(y_true).to(device=device).to(torch.long)

        # Compute the loss for this batch
        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        # Update memory and neighbor loader states
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # Calculate per-edge losses for ranking
        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

        # Store edge-specific data for reconstruction
        for i in range(len(pos_out)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])
            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][node_embedding_dim:-node_embedding_dim], 1)
            edge_type = rel2id[edgeindex]
            loss = each_edge_loss[i]

            edge_list.append(
                {
                    "loss": float(loss),
                    "srcnode": srcnode,
                    "dstnode": dstnode,
                    "srcmsg": srcmsg,
                    "dstmsg": dstmsg,
                    "edge_type": edge_type,
                    "time": t_var,
                }
            )

        event_count += len(batch.src)

        # Save results periodically after processing each time window
        if t[-1] > start_time + time_window_size:
            time_interval = (
                ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])
            )
            sanitized_time_interval = sanitize_filename(time_interval)

            # Save edge data to a log file
            log_path = os.path.join(path, f"{sanitized_time_interval}.txt")
            with open(log_path, "w") as log:
                for e in edge_list:
                    total_loss += e["loss"]
                loss = total_loss / event_count
                logger.info(
                    f"Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, "
                    f"Edges_count: {event_count}, Cost Time: {(time.perf_counter() - start):.2f}s"
                )
                edge_list.sort(
                    key=lambda x: x["loss"], reverse=True
                )  # Rank edges by loss
                for e in edge_list:
                    log.write(str(e) + "\n")

            # Reset counters and lists for the next time window
            event_count = 0
            total_loss = 0
            start_time = t[-1]
            edge_list.clear()

    return time_with_loss


# Function to load graph data for evaluation
def load_data():
    """
    Load temporal graph data for different time windows.
    Returns:
        List of graphs for training, validation, and testing.
    """
    graph_4_3 = torch.load(
        os.path.join(graphs_dir, "graph_4_3.TemporalData.simple")
    ).to(device=device)
    graph_4_4 = torch.load(
        os.path.join(graphs_dir, "graph_4_4.TemporalData.simple")
    ).to(device=device)
    graph_4_5 = torch.load(
        os.path.join(graphs_dir, "graph_4_5.TemporalData.simple")
    ).to(device=device)
    graph_4_6 = torch.load(
        os.path.join(graphs_dir, "graph_4_6.TemporalData.simple")
    ).to(device=device)
    graph_4_7 = torch.load(
        os.path.join(graphs_dir, "graph_4_7.TemporalData.simple")
    ).to(device=device)

    return [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]


# Main program
if __name__ == "__main__":
    logger.info("Start logging.")

    # Load the mapping from node IDs to messages
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Load graph data
    graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7 = load_data()

    # Load the trained model components
    model_path = os.path.join(models_dir, "models.pt")
    memory, gnn, link_pred, neighbor_loader = torch.load(
        model_path, map_location=device
    )

    # Reconstruct edges for each graph and save the results
    for graph, name in zip(
        [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7],
        ["graph_4_3", "graph_4_4", "graph_4_5", "graph_4_6", "graph_4_7"],
    ):
        test(
            inference_data=graph,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg,
            path=os.path.join(artifact_dir, name),
        )
