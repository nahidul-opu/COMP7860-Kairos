import logging
import os
import math
import copy
import torch
from tqdm import tqdm
from kairos_utils import *
from config import *

# Setting for logging
logger = logging.getLogger("anomalous_queue_logger")
logger.setLevel(logging.INFO)

# Ensure cross-platform paths for the log file
artifact_path = os.path.join(artifact_dir, "anomalous_queue.log")
file_handler = logging.FileHandler(artifact_path)  # Log output to a file
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def cal_anomaly_loss(loss_list, edge_list):
    """
    Calculate anomaly-related metrics based on edge loss values.

        loss_list (list): List of loss values for edges.
        edge_list (list): List of edges (source and destination nodes).

    Returns:
        tuple: Count of anomalous edges, average anomaly loss, set of anomalous nodes, set of anomalous edges.
    """
    if len(loss_list) != len(edge_list):
        print("error!")  # Sanity check for input lengths
        return 0

    count = 0  # Count of anomalous edges
    loss_sum = 0  # Sum of anomalous losses
    loss_std = std(loss_list)  # Standard deviation of loss values
    loss_mean = mean(loss_list)  # Mean of loss values
    edge_set = set()  # Set of anomalous edges
    node_set = set()  # Set of anomalous nodes

    # Define anomaly threshold as 1.5 standard deviations above the mean
    thr = loss_mean + 1.5 * loss_std
    logger.info(f"thr:{thr}")

    for i in range(len(loss_list)):
        if loss_list[i] > thr:  # Check if the edge is anomalous
            count += 1
            src_node = edge_list[i][0]  # Source node
            dst_node = edge_list[i][1]  # Destination node
            loss_sum += loss_list[i]

            # Add source and destination nodes to the anomalous set
            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(
                edge_list[i][0] + edge_list[i][1]
            )  # Combine source and destination for edge ID
    return count, loss_sum / count, node_set, edge_set  # Return results


def compute_IDF():
    """
    Compute Inverse Document Frequency (IDF) for nodes in the dataset. Higher the IDF, rarer the node.

    Returns:
        dict: Mapping of node to its IDF value.
        list: List of file paths processed.
    """
    node_IDF = {}  # Dictionary to store IDF values for nodes
    file_list = []  # List of files to process

    # Gather all file paths for the graphs
    for subdir in ["graph_4_3", "graph_4_4", "graph_4_5"]:
        file_path = os.path.join(artifact_dir, subdir)
        file_l = os.listdir(file_path)
        for file_name in file_l:
            file_list.append(os.path.join(file_path, file_name))

    node_set = {}  # Dictionary to store node occurrences across files
    for f_path in tqdm(file_list):  # Iterate over all files
        with open(f_path, "r") as f:
            for line in f:
                l = line.strip()
                jdata = eval(l)  # Parse the JSON line into a dictionary
                if jdata["loss"] > 0:  # Consider only nodes with positive loss
                    if "netflow" not in str(jdata["srcmsg"]):  # Ignore netflow nodes
                        node_set.setdefault(str(jdata["srcmsg"]), set()).add(f_path)
                    if "netflow" not in str(jdata["dstmsg"]):  # Ignore netflow nodes
                        node_set.setdefault(str(jdata["dstmsg"]), set()).add(f_path)

    # Calculate IDF for each node
    for n in node_set:
        include_count = len(node_set[n])  # Number of files containing the node
        IDF = math.log(len(file_list) / (include_count + 1))  # IDF formula
        node_IDF[n] = IDF

    # Save the IDF values for later use
    torch.save(node_IDF, os.path.join(artifact_dir, "node_IDF"))
    logger.info("IDF weight calculation complete!")
    return node_IDF, file_list


def cal_set_rel(s1, s2, node_IDF, tw_list):
    """
    Calculate the relationship between two time windows based on overlapping nodes.

        s1 (set): Set of nodes in the first time window.
        s2 (set): Set of nodes in the second time window.
        node_IDF (dict): IDF values for nodes.
        tw_list (list): List of time windows.

    Returns:
        int: Count of rare nodes shared between the two time windows.
    """

    def is_include_key_word(s):
        """
        Check if a node contains any common or noisy keywords.

            s (str): Node identifier.
        Returns:
            bool: True if the node contains a keyword, False otherwise.
        """
        keywords = [
            "netflow",
            "/home/george/Drafts",
            "usr",
            "proc",
            "var",
            "cadet",
            "/var/log/debug.log",
            "/var/log/cron",
            "/home/charles/Drafts",
            "/etc/ssl/cert.pem",
            "/tmp/.31.3022e",
        ]
        return any(k in s for k in keywords)

    new_s = s1 & s2  # Nodes common to both time windows
    count = 0  # Count of rare nodes
    for i in new_s:
        if is_include_key_word(i):  # Adjust IDF for noisy nodes
            node_IDF[i] = math.log(len(tw_list) / (1 + len(tw_list)))

        IDF = node_IDF.get(i, math.log(len(tw_list) / 1))  # Default IDF for rare nodes

        # Threshold for rareness
        if IDF > math.log(len(tw_list) * 0.9):
            logger.info(f"node:{i}, IDF:{IDF}")
            count += 1
    return count


def anomalous_queue_construction(node_IDF, tw_list, graph_dir_path):
    """
    Construct queues of anomalous nodes based on correlationship of nodes across time windows.

        node_IDF (dict): IDF values for nodes.
        tw_list (list): List of time windows.
        graph_dir_path (str): Directory containing graph files.

    Returns:
        list: List of history queues for anomalous nodes.
    """
    history_list = []  # List of historical anomalous queues
    file_l = os.listdir(graph_dir_path)  # List of files in the directory
    index_count = 0  # Index of the current time window

    for f_path in sorted(file_l):  # Process files in sorted order
        logger.info("**************************************************")
        logger.info(f"Time window: {f_path}")

        full_path = os.path.join(graph_dir_path, f_path)
        with open(full_path, "r") as f:
            edge_loss_list = []  # Losses for edges
            edge_list = []  # List of edges
            logger.info(f"Time window index: {index_count}")

            for line in f:
                l = line.strip()
                jdata = eval(l)
                edge_loss_list.append(jdata["loss"])  # Add edge loss
                edge_list.append(
                    [str(jdata["srcmsg"]), str(jdata["dstmsg"])]
                )  # Add edge

        # Compute anomaly metrics for this time window
        count, loss_avg, node_set, edge_set = cal_anomaly_loss(
            edge_loss_list, edge_list
        )
        current_tw = {
            "name": f_path,
            "loss": loss_avg,
            "index": index_count,
            "nodeset": node_set,
        }

        # Check relationships with historical queues
        added_que_flag = False
        for hq in history_list:
            for his_tw in hq:
                if (
                    cal_set_rel(
                        current_tw["nodeset"], his_tw["nodeset"], node_IDF, tw_list
                    )
                    != 0
                    and current_tw["name"] != his_tw["name"]
                ):
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
            if added_que_flag:
                break

        # If no match, create a new queue
        if not added_que_flag:
            temp_hq = [copy.deepcopy(current_tw)]
            history_list.append(temp_hq)

        index_count += 1

        # Log metrics for the current time window
        logger.info(f"Average loss: {loss_avg}")
        logger.info(f"Num of anomalous edges within the time window: {count}")
        logger.info(f"Percentage of anomalous edges: {count / len(edge_list)}")
        logger.info(f"Anomalous node count: {len(node_set)}")
        logger.info(f"Anomalous edge count: {len(edge_set)}")
        logger.info("**************************************************")

    return history_list


if __name__ == "__main__":
    logger.info("Start logging.")

    # Compute IDF values for nodes
    node_IDF, tw_list = compute_IDF()

    # Process each graph directory and construct anomalous queues
    for subdir in ["graph_4_5", "graph_4_6", "graph_4_7"]:
        graph_dir = os.path.join(artifact_dir, subdir)
        history_list = anomalous_queue_construction(
            node_IDF=node_IDF, tw_list=tw_list, graph_dir_path=graph_dir
        )
        # Save the anomalous queue history to a file
        torch.save(history_list, os.path.join(artifact_dir, f"{subdir}_history_list"))
