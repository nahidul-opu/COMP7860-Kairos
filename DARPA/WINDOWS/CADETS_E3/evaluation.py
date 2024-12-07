from sklearn.metrics import confusion_matrix, roc_auc_score
import logging
import os
import torch
from kairos_utils import *  # Utility functions for the system
from config import *  # Configuration settings
from model import *  # Model definitions

# Setting for logging
logger = logging.getLogger("evaluation_logger")  # Initialize logger for evaluation
logger.setLevel(logging.INFO)  # Set logging level to INFO

# Ensure cross-platform paths for the log file
artifact_path = os.path.join(artifact_dir, "evaluation.log")
file_handler = logging.FileHandler(artifact_path)  # Output log messages to a file
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def sanitize_filename(filename):
    """
    Replace invalid characters in a filename with underscores.
    This ensures filenames are safe for storage in a filesystem.

        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    return filename.translate(str.maketrans({":": "_", "\\": "_", "/": "_"}))


def classifier_evaluation(y_test, y_test_pred):
    """
    Evaluate the performance of a classifier using standard metrics.

        y_test (list): Ground truth labels.
        y_test_pred (list): Predicted labels.

    Returns:
        tuple: Metrics including precision, recall, F1-score, accuracy, and AUC.
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Log confusion matrix components
    logger.info(f"tn: {tn}")
    logger.info(f"fp: {fp}")
    logger.info(f"fn: {fn}")
    logger.info(f"tp: {tp}")

    # Calculate performance metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fscore = 2 * (precision * recall) / (precision + recall)
    auc_val = roc_auc_score(y_test, y_test_pred)

    # Log calculated metrics
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fscore: {fscore}")
    logger.info(f"accuracy: {accuracy}")
    logger.info(f"auc_val: {auc_val}")

    return precision, recall, fscore, accuracy, auc_val


def ground_truth_label():
    """
    Generate ground truth labels for validation and testing datasets.

    Returns:
        dict: Mapping of filenames to their corresponding ground truth labels (0 or 1).
    """
    labels = {}

    # Set all files in graph_4_6 directory to label 0 (non-attack)
    graph_4_6_path = os.path.join(artifact_dir, "graph_4_6")
    filelist = os.listdir(graph_4_6_path)
    for f in filelist:
        labels[f] = 0

    # Set all files in graph_4_7 directory to label 0 (non-attack)
    graph_4_7_path = os.path.join(artifact_dir, "graph_4_7")
    filelist = os.listdir(graph_4_7_path)
    for f in filelist:
        labels[f] = 0

    # Mark specific files as attack events (label 1)
    attack_list = [
        "2018-04-06 11_18_26.126177915~2018-04-06 11_33_35.116170745.txt",
        "2018-04-06 11_33_35.116170745~2018-04-06 11_48_42.606135188.txt",
        "2018-04-06 11_48_42.606135188~2018-04-06 12_03_50.186115455.txt",
        "2018-04-06 12_03_50.186115455~2018-04-06 14_01_32.489584227.txt",
    ]
    for i in attack_list:
        labels[i] = 1

    return labels


def calc_attack_edges():
    """
    Count the number of attack edges in a given set of files.
    An edge is considered an attack edge if it matches certain attack keywords.
    """

    def keyword_hit(line):
        """
        Check if a line contains any of the attack-related keywords.

        Args:
            line (str): A line of text.

        Returns:
            bool: True if the line contains any attack keywords, False otherwise.
        """
        attack_nodes = [
            "vUgefal",
            "/var/log/devc",
            "nginx",
            "81.49.200.166",
            "78.205.235.65",
            "200.36.109.214",
            "139.123.0.113",
            "152.111.159.139",
            "61.167.39.128",
        ]
        return any(i in line for i in attack_nodes)

    files = []

    # List attack-related files in graph_4_6
    graph_4_6_path = os.path.join(artifact_dir, "graph_4_6")
    attack_list = [
        "2018-04-06 11_18_26.126177915~2018-04-06 11_33_35.116170745.txt",
        "2018-04-06 11_33_35.116170745~2018-04-06 11_48_42.606135188.txt",
        "2018-04-06 11_48_42.606135188~2018-04-06 12_03_50.186115455.txt",
        "2018-04-06 12_03_50.186115455~2018-04-06 14_01_32.489584227.txt",
    ]
    for f in attack_list:
        files.append(os.path.join(graph_4_6_path, f))

    attack_edge_count = 0  # Initialize counter for attack edges
    for fpath in files:  # Iterate over attack files
        with open(fpath, "r") as f:
            for line in f:
                if keyword_hit(line):  # Check for attack-related keywords
                    attack_edge_count += 1

    logger.info(f"Num of attack edges: {attack_edge_count}")


if __name__ == "__main__":
    logger.info("Start logging.")

    # Calculate anomaly scores for the validation set
    anomalous_queue_scores = []
    history_list_path = os.path.join(artifact_dir, "graph_4_5_history_list")
    history_list = torch.load(history_list_path)

    # Compute anomaly scores from history list
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq["loss"] + 1)
            else:
                anomaly_score *= hq["loss"] + 1
        anomalous_queue_scores.append(anomaly_score)

    logger.info(
        f"The largest anomaly score in validation set is: {max(anomalous_queue_scores)}\n"
    )

    # Predict labels for the testing set
    pred_label = {}

    # Process files in graph_4_6 directory
    graph_4_6_path = os.path.join(artifact_dir, "graph_4_6")
    filelist = os.listdir(graph_4_6_path)
    for f in filelist:
        pred_label[sanitize_filename(f)] = 0

    # Process files in graph_4_7 directory
    graph_4_7_path = os.path.join(artifact_dir, "graph_4_7")
    filelist = os.listdir(graph_4_7_path)
    for f in filelist:
        pred_label[sanitize_filename(f)] = 0

    # Update predictions using history list for graph_4_6
    history_list_path = os.path.join(artifact_dir, "graph_4_6_history_list")
    history_list = torch.load(history_list_path)
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq["loss"] + 1)
            else:
                anomaly_score *= hq["loss"] + 1
        if anomaly_score > beta_day6:  # Threshold for day 6
            for i in hl:
                pred_label[sanitize_filename(i["name"])] = 1

    # Update predictions using history list for graph_4_7
    history_list_path = os.path.join(artifact_dir, "graph_4_7_history_list")
    history_list = torch.load(history_list_path)
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq["loss"] + 1)
            else:
                anomaly_score *= hq["loss"] + 1
        if anomaly_score > beta_day7:  # Threshold for day 7
            for i in hl:
                pred_label[sanitize_filename(i["name"])] = 1

    # Evaluate the classifier using ground truth and predictions
    labels = ground_truth_label()
    y = [labels[sanitize_filename(i)] for i in labels]
    y_pred = [pred_label[sanitize_filename(i)] for i in labels]
    classifier_evaluation(y, y_pred)
