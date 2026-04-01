import tensorflow as tf
import numpy as np
import os
import random

def boundary_negative_contrastive_loss(query_y_pred, y_true_q, support_y_pred, cfg):
    """
    Computes the Boundary Negative Contrastive Loss to enhance inter-class separation.
    
    Args:
        query_y_pred: Predicted embeddings for query samples (Nq, D)
        y_true_q: Ground truth labels for query samples (Nq, C)
        support_y_pred: Predicted embeddings for support samples (Ns, D)
        cfg: Configuration dictionary
    """
    embedding_dim = tf.shape(support_y_pred)[-1]
    # Reshape support embeddings to (Class, Shot, Dimension)
    support_y_pred_n = tf.reshape(support_y_pred, (cfg["number_class"], cfg["support_n_shot"], embedding_dim))
    # Compute prototypes (mean of support embeddings per class)
    support_feature = tf.reduce_mean(support_y_pred_n, axis=1)

    # Calculate boundary radius for each class using Euclidean distance
    diff = support_y_pred_n - tf.expand_dims(support_feature, axis=1)
    support_boundary_metric = tf.norm(diff, ord="euclidean", axis=2) 
    # Define class radius as the mean distance of support samples from the prototype
    support_boundary = tf.reduce_mean(support_boundary_metric, axis=1) # (C,)

    # Compute distance between query samples and prototypes
    # Using Cosine distance as defined in the proposed framework
    q_expanded = tf.expand_dims(query_y_pred, axis=1) # (Nq, 1, D)
    s_expanded = tf.expand_dims(support_feature, axis=0) # (1, C, D)
    
    # Cosine Similarity
    normalize_q = tf.nn.l2_normalize(q_expanded, axis=2)
    normalize_s = tf.nn.l2_normalize(s_expanded, axis=2)
    cosine_sim = tf.reduce_sum(normalize_q * normalize_s, axis=2) # (Nq, C)
    
    # Distance = 1 - Similarity
    dist_matrix = 1.0 - cosine_sim

    # Penalize query samples that fall outside the boundary of their true class
    y_true_indices = tf.argmax(y_true_q, axis=1)
    batch_indices = tf.range(tf.shape(y_true_q)[0], dtype=tf.int64)
    indices = tf.stack([batch_indices, y_true_indices], axis=1)
    
    true_class_dist = tf.gather_nd(dist_matrix, indices)
    true_class_boundary = tf.gather(support_boundary, y_true_indices)

    # Boundary Loss: Max(0, Distance - Radius)
    loss = tf.reduce_mean(tf.maximum(0.0, true_class_dist - true_class_boundary))
    return loss

def get_samples(data, cfg, add_noise=False):
    """
    Generates N-way N-shot episodes from the dataset.
    Includes Gaussian noise injection strategy for robustness evaluation.
    """
    # [Implementation of episode sampling logic...]
    # If add_noise is True, add Gaussian noise with std=0.02 as described in the paper
    pass