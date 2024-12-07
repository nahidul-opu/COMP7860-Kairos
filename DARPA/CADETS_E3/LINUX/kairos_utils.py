import pytz
from time import mktime
from datetime import datetime
import time
import psycopg2
from psycopg2 import extras as ex
import os.path as osp
import os
import copy
import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import TemporalData
from torch_geometric.nn import TGNMemory, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import (
    LastNeighborLoader,
    IdentityMessage,
    MeanAggregator,
    LastAggregator,
)
from torch_geometric import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import copy
import time
import xxhash
import gc

from config import *


# Function to convert nanosecond timestamps to datetime objects
def ns_time_to_datetime(ns):
    """
    Convert nanosecond timestamp to a datetime object.

    :param ns: int, nanosecond timestamp
    :return: datetime object in the format 'YYYY-MM-DD HH:MM:SS.000000000'
    """
    dt = datetime.fromtimestamp(int(ns) // 1_000_000_000)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    s += "." + str(int(ns % 1_000_000_000)).zfill(9)
    return s


# Function to convert nanosecond timestamps to datetime in US/Eastern timezone
def ns_time_to_datetime_US(ns):
    """
    Convert nanosecond timestamp to a datetime object in US/Eastern timezone.

    :param ns: int, nanosecond timestamp
    :return: datetime object in the format 'YYYY-MM-DD HH:MM:SS.000000000'
    """
    tz = pytz.timezone("US/Eastern")
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1_000_000_000, tz)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    s += "." + str(int(ns % 1_000_000_000)).zfill(9)
    return s


# Function to convert Unix time to datetime in US/Eastern timezone
def time_to_datetime_US(s):
    """
    Convert Unix timestamp to a datetime object in US/Eastern timezone.

    :param s: int, Unix timestamp
    :return: datetime object in the format 'YYYY-MM-DD HH:MM:SS'
    """
    tz = pytz.timezone("US/Eastern")
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# Function to convert datetime to nanosecond timestamp
def datetime_to_ns_time(date):
    """
    Convert a datetime string to a nanosecond timestamp.

    :param date: str, datetime in format '%Y-%m-%d %H:%M:%S'
    :return: int, nanosecond timestamp
    """
    time_array = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timestamp = int(mktime(time_array))
    return timestamp * 1_000_000_000


# Function to convert datetime to nanosecond timestamp in US/Eastern timezone
def datetime_to_ns_time_US(date):
    """
    Convert a datetime string to a nanosecond timestamp in US/Eastern timezone.

    :param date: str, datetime in format '%Y-%m-%d %H:%M:%S'
    :return: int, nanosecond timestamp
    """
    tz = pytz.timezone("US/Eastern")
    time_array = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(time_array))
    timestamp = tz.localize(dt).timestamp()
    return int(timestamp * 1_000_000_000)


# Function to convert datetime to Unix timestamp in US/Eastern timezone
def datetime_to_timestamp_US(date):
    """
    Convert a datetime string to a Unix timestamp in US/Eastern timezone.

    :param date: str, datetime in format '%Y-%m-%d %H:%M:%S'
    :return: int, Unix timestamp
    """
    tz = pytz.timezone("US/Eastern")
    time_array = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(time_array))
    timestamp = tz.localize(dt).timestamp()
    return int(timestamp)


# Function to initialize a database connection using psycopg2
def init_database_connection():
    """
    Initialize a connection to the PostgreSQL database.

    :return: tuple, (cursor, connection) to interact with the database
    """
    if host is not None:
        connect = psycopg2.connect(
            database=database, host=host, user=user, password=password, port=port
        )
    else:
        connect = psycopg2.connect(
            database=database, user=user, password=password, port=port
        )
    cur = connect.cursor()
    return cur, connect


# Function to create a mapping from node IDs to messages
def gen_nodeid2msg(cur):
    """
    Generate a dictionary mapping node IDs to their associated messages.

    :param cur: psycopg2 cursor object
    :return: dict, node ID to message mapping
    """
    sql = "SELECT * FROM node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for row in rows:
        nodeid2msg[row[0]] = row[-1]
        nodeid2msg[row[-1]] = {row[1]: row[2]}
    return nodeid2msg


# Function to find an element in a tensor
def tensor_find(t, x):
    """
    Find the index of an element in a PyTorch tensor.

    :param t: torch.Tensor, tensor to search
    :param x: element to find
    :return: int, index of the element (1-based index)
    """
    t_np = t.cpu().numpy()
    idx = np.argwhere(t_np == x)
    return idx[0][0] + 1


# Function to compute standard deviation of a list
def std(t):
    """
    Calculate the standard deviation of a list.

    :param t: list or array-like, input values
    :return: float, standard deviation
    """
    return np.std(np.array(t))


# Function to compute variance of a list
def var(t):
    """
    Calculate the variance of a list.

    :param t: list or array-like, input values
    :return: float, variance
    """
    return np.var(np.array(t))


# Function to compute mean of a list
def mean(t):
    """
    Calculate the mean of a list.

    :param t: list or array-like, input values
    :return: float, mean
    """
    return np.mean(np.array(t))


# Function to generate a hash value from a list
def hashgen(l):
    """
    Generate a single hash value from a list of strings.

    :param l: list of str, elements to hash
    :return: int, hashed integer value
    """
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()
