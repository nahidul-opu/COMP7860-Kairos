import os
from graphviz import Digraph
import networkx as nx
import community.community_louvain as community_louvain
from tqdm import tqdm

from config import *  # Configuration settings
from kairos_utils import *  # Utility functions

# Common path abstractions to simplify file paths for visualization
replace_dic = {
    "/run/shm/": "/run/shm/*",
    "/home/admin/.cache/mozilla/firefox/": "/home/admin/.cache/mozilla/firefox/*",
    "/home/admin/.mozilla/firefox": "/home/admin/.mozilla/firefox*",
    "/data/replay_logdb/": "/data/replay_logdb/*",
    "/home/admin/.local/share/applications/": "/home/admin/.local/share/applications/*",
    "/usr/share/applications/": "/usr/share/applications/*",
    "/lib/x86_64-linux-gnu/": "/lib/x86_64-linux-gnu/*",
    "/proc/": "/proc/*",
    "/stat": "*/stat",
    "/etc/bash_completion.d/": "/etc/bash_completion.d/*",
    "/usr/bin/python2.7": "/usr/bin/python2.7/*",
    "/usr/lib/python2.7": "/usr/lib/python2.7/*",
}


# Replace specific file paths with generalized names for better readability
def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    return path_name


# Replace invalid characters in filenames with underscores for cross-platform compatibility
def sanitize_filename(filename):
    return filename.translate(str.maketrans({":": "_", "\\": "_", "/": "_"}))


# Manually detected anomalous time windows (files containing edge data)
attack_list = [
    os.path.join(
        artifact_dir,
        "graph_4_6",
        sanitize_filename(
            "2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt"
        ),
    ),
    os.path.join(
        artifact_dir,
        "graph_4_6",
        sanitize_filename(
            "2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt"
        ),
    ),
    os.path.join(
        artifact_dir,
        "graph_4_6",
        sanitize_filename(
            "2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt"
        ),
    ),
    os.path.join(
        artifact_dir,
        "graph_4_6",
        sanitize_filename(
            "2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt"
        ),
    ),
]

# Initialize variables for storing graphs and edges
original_edges_count = 0  # Count of total edges in the original data
graphs = []  # List of individual graphs
gg = nx.DiGraph()  # Combined directed graph
count = 0  # Overall edge count

# Process each file in the anomalous time windows
for path in tqdm(attack_list):
    if ".txt" in path:  # Ensure the file is a text file
        line_count = 0
        node_set = set()
        tempg = nx.DiGraph()  # Temporary graph for the current file
        with open(path, "r") as f:
            edge_list = []
            for line in f:
                count += 1
                l = line.strip()
                jdata = eval(l)  # Evaluate the line to get edge data
                edge_list.append(jdata)

        # Sort edges by their loss values (descending)
        edge_list = sorted(edge_list, key=lambda x: x["loss"], reverse=True)
        original_edges_count += len(edge_list)

        # Calculate threshold for identifying high-loss edges
        loss_list = [e["loss"] for e in edge_list]
        loss_mean = mean(loss_list)
        loss_std = std(loss_list)
        thr = loss_mean + 1.5 * loss_std  # Threshold: Mean + 1.5 * Std
        print(loss_mean)
        print(loss_std)
        print("thr:", thr)

        # Add high-loss edges to the graph
        for e in edge_list:
            if e["loss"] > thr:
                tempg.add_edge(
                    str(hashgen(replace_path_name(e["srcmsg"]))),
                    str(hashgen(replace_path_name(e["dstmsg"]))),
                )
                gg.add_edge(
                    str(hashgen(replace_path_name(e["srcmsg"]))),
                    str(hashgen(replace_path_name(e["dstmsg"]))),
                    loss=e["loss"],
                    srcmsg=e["srcmsg"],
                    dstmsg=e["dstmsg"],
                    edge_type=e["edge_type"],
                    time=e["time"],
                )

# Apply community detection to group nodes into communities
partition = community_louvain.best_partition(gg.to_undirected())

# Generate subgraphs based on community detection results
communities = {}
max_partition = max(partition.values())
for i in range(max_partition + 1):
    communities[i] = nx.DiGraph()
for e in gg.edges:
    communities[partition[e[0]]].add_edge(e[0], e[1])
    communities[partition[e[1]]].add_edge(e[0], e[1])


# Define attack nodes for visualization purposes
def attack_edge_flag(msg):
    attack_nodes = [
        "/tmp/vUgefal",
        "vUgefal",
        "/var/log/devc",
        "/etc/passwd",
        "81.49.200.166",
        "61.167.39.128",
        "78.205.235.65",
        "139.123.0.113",
        "'nginx'",
    ]
    return any(i in msg for i in attack_nodes)


# Render and save subgraphs as PDF files
graph_visual_path = os.path.join(artifact_dir, "graph_visual")
os.makedirs(graph_visual_path, exist_ok=True)

graph_index = 0
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr["rankdir"] = "LR"

    for e in communities[c].edges:
        try:
            temp_edge = gg.edges[e]
        except KeyError:
            continue

        # Source node visualization
        if "'subject': '" in temp_edge["srcmsg"]:
            src_shape = "box"
        elif "'file': '" in temp_edge["srcmsg"]:
            src_shape = "oval"
        elif "'netflow': '" in temp_edge["srcmsg"]:
            src_shape = "diamond"
        src_node_color = "red" if attack_edge_flag(temp_edge["srcmsg"]) else "blue"
        dot.node(
            name=str(hashgen(replace_path_name(temp_edge["srcmsg"]))),
            label=str(
                replace_path_name(temp_edge["srcmsg"])
                + str(partition[str(hashgen(replace_path_name(temp_edge["srcmsg"])))]),
            ),
            color=src_node_color,
            shape=src_shape,
        )

        # Destination node visualization
        if "'subject': '" in temp_edge["dstmsg"]:
            dst_shape = "box"
        elif "'file': '" in temp_edge["dstmsg"]:
            dst_shape = "oval"
        elif "'netflow': '" in temp_edge["dstmsg"]:
            dst_shape = "diamond"
        dst_node_color = "red" if attack_edge_flag(temp_edge["dstmsg"]) else "blue"
        dot.node(
            name=str(hashgen(replace_path_name(temp_edge["dstmsg"]))),
            label=str(
                replace_path_name(temp_edge["dstmsg"])
                + str(partition[str(hashgen(replace_path_name(temp_edge["dstmsg"])))]),
            ),
            color=dst_node_color,
            shape=dst_shape,
        )

        # Edge visualization
        edge_color = (
            "red"
            if attack_edge_flag(temp_edge["srcmsg"])
            and attack_edge_flag(temp_edge["dstmsg"])
            else "blue"
        )
        dot.edge(
            str(hashgen(replace_path_name(temp_edge["srcmsg"]))),
            str(hashgen(replace_path_name(temp_edge["dstmsg"]))),
            label=temp_edge["edge_type"],
            color=edge_color,
        )

    # Save the subgraph to a file
    dot.render(os.path.join(graph_visual_path, f"subgraph_{graph_index}"), view=False)
    graph_index += 1
