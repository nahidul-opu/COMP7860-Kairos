##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging
from kairos_utils import *
from config import *
from model import *

# Setting up logging to track progress and results
logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + "reconstruction.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Function to test the model by reconstructing edges in the inference dataset
@torch.no_grad()  # Disables gradient computation for inference
def test(inference_data, memory, gnn, link_pred, neighbor_loader, nodeid2msg, path):
    """
    Perform edge reconstruction and loss computation for inference data.
    Saves the ranked edges and their respective losses to disk.

    Parameters:
        inference_data: TemporalData object containing inference data.
        memory: TGNMemory, memory module for temporal graph networks. (Gated Recurrent Unit [GRU] for State Update)
        gnn: GraphAttentionEmbedding, graph neural network model. (Encoder)
        link_pred: LinkPredictor, link prediction model. (Decoder)
        neighbor_loader: LastNeighborLoader, loader for neighbor nodes. (Required for obtaining the state of neighbor nodes)
        nodeid2msg: Mapping from node IDs to human-readable messages.
        path: Directory to save the reconstruction results.

    Returns:
        Dictionary with time intervals and associated losses, node counts, and edge counts.
    """

    # Create output directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    # Switch models to evaluation mode
    memory.eval()
    gnn.eval()
    link_pred.eval()

    # Reset memory and neighbor loader states for inference
    memory.reset_state()
    neighbor_loader.reset_state()

    time_with_loss = {}  # Dictionary to store results for each time window
    total_loss = 0  # Cumulative loss across events
    edge_list = []  # List to store details of edges processed

    unique_nodes = torch.tensor([]).to(device=device)  # Track unique nodes
    total_edges = 0  # Total edges processed

    start_time = inference_data.t[0]  # Initial timestamp for time window
    event_count = 0  # Counter for number of events in the current window
    pos_o = []  # Store positive predictions

    # Record the start time for performance measurement
    start = time.perf_counter()

    # Iterate through batches of inference data
    for batch in inference_data.seq_batches(batch_size=BATCH):

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes = torch.cat(
            [unique_nodes, src, pos_dst]
        ).unique()  # Update unique nodes
        total_edges += BATCH

        # Prepare node and edge information for neighbor loader
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(
            n_id.size(0), device=device
        )  # Associate node IDs with their indices

        # Retrieve and update memory embeddings for nodes
        z, last_update = memory(n_id)
        z = gnn(
            z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id]
        )

        # Compute positive edge scores
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        pos_o.append(pos_out)

        # Compute true labels from messages
        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        # Compute loss for the current batch
        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        # Update memory and neighbor loader with current edges
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # Compute per-edge losses
        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

        # Log details for each edge
        for i in range(len(pos_out)):
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])

            srcmsg = str(nodeid2msg[srcnode])
            dstmsg = str(nodeid2msg[dstnode])
            t_var = int(t[i])
            edgeindex = tensor_find(msg[i][node_embedding_dim:-node_embedding_dim], 1)
            edge_type = rel2id[edgeindex]
            loss = each_edge_loss[i]

            temp_dic = {
                "loss": float(loss),
                "srcnode": srcnode,
                "dstnode": dstnode,
                "srcmsg": srcmsg,
                "dstmsg": dstmsg,
                "edge_type": edge_type,
                "time": t_var,
            }

            edge_list.append(temp_dic)

        # Checkpoint results at the end of each time window
        event_count += len(batch.src)
        if t[-1] > start_time + time_window_size:
            time_interval = (
                ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])
            )

            end = time.perf_counter()
            time_with_loss[time_interval] = {
                "loss": loss,
                "nodes_count": len(unique_nodes),
                "total_edges": total_edges,
                "costed_time": (end - start),
            }

            log = open(path + "/" + time_interval + ".txt", "w")

            # Average loss over events
            for e in edge_list:
                loss += e["loss"]
            loss = loss / event_count

            logger.info(
                f"Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s"
            )

            # Rank edges by loss and save to file
            edge_list = sorted(edge_list, key=lambda x: x["loss"], reverse=True)
            for e in edge_list:
                log.write(str(e))
                log.write("\n")
            event_count = 0
            total_loss = 0
            start_time = t[-1]
            log.close()
            edge_list.clear()

    return time_with_loss


# Function to load graph data for inference
def load_data():
    """Load the datasets for initialization and testing."""
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(
        device=device
    )
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(
        device=device
    )
    graph_4_5 = torch.load(graphs_dir + "/graph_4_5.TemporalData.simple").to(
        device=device
    )
    graph_4_6 = torch.load(graphs_dir + "/graph_4_6.TemporalData.simple").to(
        device=device
    )
    graph_4_7 = torch.load(graphs_dir + "/graph_4_7.TemporalData.simple").to(
        device=device
    )

    return [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7]


# Main execution starts here
if __name__ == "__main__":
    logger.info("Start logging.")

    # Load node ID to message mapping
    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Load data for inference
    graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7 = load_data()

    # Load pre-trained model components
    memory, gnn, link_pred, neighbor_loader = torch.load(
        f"{models_dir}/models.pt", map_location=device
    )

    # Perform edge reconstruction for each graph
    for i, graph in enumerate(
        [graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7], start=3
    ):
        test(
            inference_data=graph,
            memory=memory,
            gnn=gnn,
            link_pred=link_pred,
            neighbor_loader=neighbor_loader,
            nodeid2msg=nodeid2msg,
            path=artifact_dir + f"graph_4_{i}",
        )
