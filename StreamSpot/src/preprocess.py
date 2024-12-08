# Input: StreamSpot dataset
# Output: Vectorized graphs

import functools
import os
import json
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *

# If the datum have already been loaded in the database before, set this as False.
# Set it as True if it is the first time running this code.
process_raw_data = True


import psycopg2
from psycopg2 import extras as ex
# Create a postgreSQL DB connection object for storing provenance graph edges into DB
# Original '/var/run/postgresql/' has been replaced with 'localhost' since we are using docker and accessing as a service in port 5432
connect = psycopg2.connect(database = 'streamspot',
                           host = 'localhost',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                           )

# Create a cursor to operate the database
cur = connect.cursor()
# Rollback when there exists any problem
connect.rollback()

if process_raw_data:
    # Uncompressed(2.1 GB) TSV file path
    path = "/home/shahidul2k9/data/streamspot/all.tsv"  # The paths to the dataset.
    # Temporary buffer for storing ~10K edges after reading from file and then clearing it once flush into DB
    datalist = []
    # Open the file object
    with open(path) as f:
        # Read file content line by line with tqdm library
        for line in tqdm(f):
            spl = line.strip().split('\t')
            datalist.append(spl)
            # If buffer reaches 10K threshold then insert records/eges intot DB
            if len(datalist) >= 10000:
                sql = '''insert into raw_data
                 values %s
                '''
                ex.execute_values(cur, sql, datalist, page_size=10000)
                # Commit the insertion command
                connect.commit()
                # Clear buffer
                datalist = []

# StreamSpot Dataset has 6 types of nodes that are mapped into a-h
node_type = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}
# StreamSpot Dataset has 26 types of edges that are mapped into A-H and i-z
edge_type = {'A',
             'B',
             'C',
             'D',
             'E',
             'F',
             'G',
             'H',
             'i',
             'j',
             'k',
             'l',
             'm',
             'n',
             'o',
             'p',
             'q',
             'r',
             's',
             't',
             'u',
             'v',
             'w',
             'x',
             'y',
             'z'}
# Mapping StreamSpot input nodes and edges to compact formed a-z and A-H
maps = {'process': 'a',
        'thread': 'b',
        'file': 'c',
        'MAP_ANONYMOUS': 'd',
        'NA': 'e',
        'stdin': 'f',
        'stdout': 'g',
        'stderr': 'h',
        'accept': 'i',
        'access': 'j',
        'bind': 'k',
        'chmod': 'l',
        'clone': 'm',
        'close': 'n',
        'connect': 'o',
        'execve': 'p',
        'fstat': 'q',
        'ftruncate': 'r',
        'listen': 's',
        'mmap2': 't',
        'open': 'u',
        'read': 'v',
        'recv': 'w',
        'recvfrom': 'x',
        'recvmsg': 'y',
        'send': 'z',
        'sendmsg': 'A',
        'sendto': 'B',
        'stat': 'C',
        'truncate': 'D',
        'unlink': 'E',
        'waitpid': 'F',
        'write': 'G',
        'writev': 'H',
        }
# One-Hot vectors generation for mapping each node labels
nodevec = torch.nn.functional.one_hot(torch.arange(0, len(node_type)), num_classes=len(node_type))
# One-Hot vectors generation for mapping each node edges
edgevec = torch.nn.functional.one_hot(torch.arange(0, len(edge_type)), num_classes=len(edge_type))


edge2onehot = {}
node2onehot = {}
# Allocating One-Hot vector for each node
c = 0
for i in node_type:
    node2onehot[i] = nodevec[c]
    c += 1
# Allocating One-Hot vector for each edge
c = 0
for i in edge_type:
    edge2onehot[i] = edgevec[c]
    c += 1
# Creating directory for storing temporal graph of Stream Spot Dataset
os.system("mkdir -p /home/shahidul2k9/data/streamspot/graph/")
# Read 600 graphs one by one, generate temporal graph and store it each one in separate file
for graph_id in tqdm(range(600)):
    # Query string to fetch a specific provenance graph
    sql = "select * from raw_data where graph_id='{graph_id}' ORDER BY _id;".format(graph_id=graph_id)
    cur.execute(sql)
    rows = cur.fetchall()
    from torch_geometric.data import TemporalData
    # Create a temporal graph
    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    # Loop through all edges of a specific provenance graph and add source node, destination node, edge message labels
    for i in rows:
        src.append(int(i[0]))
        dst.append(int(i[2]))
        msg_t = torch.cat([node2onehot[i[1]], edge2onehot[i[4]], node2onehot[i[3]]], dim=0)
        msg.append(msg_t)
        t.append(int(i[-1]))    # Use logical order of the event to represent the time
    # Assigns source nodes
    dataset.src = torch.tensor(src)
    # Assigns destination nodes
    dataset.dst = torch.tensor(dst)
    # Assigns times associated in each node
    dataset.t = torch.tensor(t)
    # Assigns messages all edge
    dataset.msg = torch.vstack(msg)
    # Set associated node data types
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    # Store the provenance graph in persistent storage
    torch.save(dataset, "/home/shahidul2k9/data/streamspot/graph/graph_" + str(graph_id) + ".TemporalData")

print("end")
