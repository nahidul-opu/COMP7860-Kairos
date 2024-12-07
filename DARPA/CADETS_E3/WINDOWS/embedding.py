from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm

import numpy as np
import logging
import torch
import os

from config import *
from kairos_utils import *

# Setting for logging
logger = logging.getLogger("embedding_logger")
logger.setLevel(logging.INFO)

# Create a file handler to log the information to a file
file_handler = logging.FileHandler(artifact_dir + "embedding.log")
file_handler.setLevel(logging.INFO)

# Format the log messages with timestamps
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


# Converts a hierarchical file path into a list of hierarchical strings
def path2higlist(p):
    l = []
    spl = p.strip().split("/")
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + "/" + i)
        else:
            l.append(i)
    return l


# Converts an IP address into a hierarchical representation
def ip2higlist(p):
    l = []
    spl = p.strip().split(".")
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + "." + i)
        else:
            l.append(i)
    return l


# Concatenates a list of strings into a single string
def list2str(l):
    s = ""
    for i in l:
        s += i
    return s


# Generates node embeddings based on hierarchical features
def gen_feature(cur):
    # Obtain all node labels from the database
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # List to store hierarchical representations of node labels
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys()):
        if type(i) == int:  # Ensure the key is an integer
            if "netflow" in nodeid2msg[i].keys():  # Handle netflow nodes
                higlist = ["netflow"]
                higlist += ip2higlist(nodeid2msg[i]["netflow"])

            if "file" in nodeid2msg[i].keys():  # Handle file nodes
                higlist = ["file"]
                higlist += path2higlist(nodeid2msg[i]["file"])

            if "subject" in nodeid2msg[i].keys():  # Handle subject nodes
                higlist = ["subject"]
                higlist += path2higlist(nodeid2msg[i]["subject"])

            # Convert the hierarchical list to a single string
            node_msg_dic_list.append(list2str(higlist))

    # Use FeatureHasher to generate fixed-size feature vectors
    FH_string = FeatureHasher(n_features=node_embedding_dim, input_type="string")
    node2higvec = []
    for i in tqdm(node_msg_dic_list):
        vec = FH_string.transform([i]).toarray()  # Generate hashed feature vector
        node2higvec.append(vec)

    # Convert the list of vectors to a numpy array
    node2higvec = np.array(node2higvec).reshape([-1, node_embedding_dim])

    # Save the node embeddings
    torch.save(node2higvec, artifact_dir + "node2higvec")
    return node2higvec


# Generates one-hot encoded relation vectors
def gen_relation_onehot():
    # Create one-hot encoded vectors for each relation type
    relvec = torch.nn.functional.one_hot(
        torch.arange(0, len(rel2id.keys()) // 2), num_classes=len(rel2id.keys()) // 2
    )
    rel2vec = {}
    for i in rel2id.keys():
        if type(i) is not int:  # Ensure the key is not an integer
            rel2vec[i] = relvec[rel2id[i] - 1]  # Map relation name to its vector
            rel2vec[relvec[rel2id[i] - 1]] = i  # Map vector to relation name

    # Save the relation vector mapping
    torch.save(rel2vec, artifact_dir + "rel2vec")
    return rel2vec


# Generates vectorized temporal graphs for daily events
def gen_vectorized_graphs(cur, node2higvec, rel2vec, logger):
    for day in tqdm(range(2, 14)):  # Loop over the range of days (April 2 to April 14)
        # Define the time range for the day
        start_timestamp = datetime_to_ns_time_US("2018-04-" + str(day) + " 00:00:00")
        end_timestamp = datetime_to_ns_time_US("2018-04-" + str(day + 1) + " 00:00:00")

        # Query the event table for events within the time range
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (
            start_timestamp,
            end_timestamp,
        )
        cur.execute(sql)
        events = cur.fetchall()

        # Log the number of events processed for the day
        logger.info(f"2018-04-{day}, events count: {len(events)}")

        # Process the events into an edge list
        edge_list = []
        for e in events:
            edge_temp = [
                int(e[1]),
                int(e[4]),
                e[2],
                e[5],
            ]  # Extract source, destination, relation, and timestamp
            if e[2] in include_edge_type:  # Filter by edge type
                edge_list.append(edge_temp)
        logger.info(f"2018-04-{day}, edge list len: {len(edge_list)}")

        # Initialize the PyTorch Geometric TemporalData object
        dataset = TemporalData()
        src = []  # Source nodes
        dst = []  # Destination nodes
        msg = []  # Edge message features
        t = []  # Timestamps

        # Populate the TemporalData object
        for i in edge_list:
            src.append(int(i[0]))
            dst.append(int(i[1]))
            # Concatenate source node features, relation vector, and destination node features
            msg.append(
                torch.cat(
                    [
                        torch.from_numpy(node2higvec[i[0]]),
                        rel2vec[i[2]],
                        torch.from_numpy(node2higvec[i[1]]),
                    ]
                )
            )
            t.append(int(i[3]))

        # Assign processed data to the TemporalData object
        dataset.src = torch.tensor(src).to(torch.long)
        dataset.dst = torch.tensor(dst).to(torch.long)
        dataset.t = torch.tensor(t).to(torch.long)
        dataset.msg = torch.vstack(msg).to(torch.float)

        # Save the TemporalData object for the day
        torch.save(
            dataset, graphs_dir + "/graph_4_" + str(day) + ".TemporalData.simple"
        )


# Main function to orchestrate the graph generation process
if __name__ == "__main__":
    # Log the start of the process
    logger.info("Start logging.")

    # Create the directory for saving graphs if it does not exist
    os.system(f"mkdir -p {graphs_dir}")

    # Initialize the database connection
    cur, _ = init_database_connection()

    # Generate node features and save them
    node2higvec = gen_feature(cur=cur)

    # Generate relation vectors and save them
    rel2vec = gen_relation_onehot()

    # Generate vectorized graphs for daily events
    gen_vectorized_graphs(
        cur=cur, node2higvec=node2higvec, rel2vec=rel2vec, logger=logger
    )
