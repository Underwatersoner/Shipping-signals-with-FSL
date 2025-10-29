import os
import random
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from tensorflow import keras
from tensorflow.keras import models, layers, Input, Model


# =========================
# Config
# =========================
@dataclass
class CFG:
    n_way: int = 5
    support_n_shot: int = 5
    query_n_shot: int = 16
    files_per_fold: int = 21
    train_files: int = 16
    val_files: int = 5
    input_shape: tuple = (224, 224, 3)
    feat_dim: int = 512
    train_epochs: int = 700
    test_epochs: int = 500
    lr: float = 2e-5
    data_root: str = "/root/data/"
    csv_name: str = "save_result.csv"


# =========================
# Pair / Distance / Loss
# =========================
def make_contrastive_pairs_tf(query_vecs, query_labels, support_vecs, support_labels):
    """Build query-support pairs with positive(1)/negative(0) labels."""
    num_query = tf.shape(query_vecs)[0]
    num_support = tf.shape(support_vecs)[0]

    pair_1_array = tf.TensorArray(dtype=tf.float32, size=num_query * num_support)
    pair_2_array = tf.TensorArray(dtype=tf.float32, size=num_query * num_support)
    label_array  = tf.TensorArray(dtype=tf.float32, size=num_query * num_support)

    idx = 0
    for i in tf.range(num_query):
        q_vec   = query_vecs[i]
        q_label = query_labels[i]
        for j in tf.range(num_support):
            s_vec   = support_vecs[j]
            s_label = support_labels[j]

            pair_1_array = pair_1_array.write(idx, q_vec)
            pair_2_array = pair_2_array.write(idx, s_vec)

            is_same = tf.equal(q_label, s_label)
            label = tf.where(is_same, 1.0, 0.0)
            label_array = label_array.write(idx, tf.cast(label, tf.float32))
            idx += 1

    return pair_1_array.stack(), pair_2_array.stack(), label_array.stack()


def squared_euclidean_distance_pair(a, b):
    return tf.reduce_sum(tf.square(a - b), axis=1)


def contrastive_loss(y_true, y_pred, margin=0.3):
    """Contrastive loss: positive->d^2, negative->max(margin-d,0)^2."""
    y_true = tf.cast(y_true, tf.float32)
    pos = tf.square(y_pred)
    neg = tf.square(tf.maximum(margin - y_pred, 0.0))
    return tf.reduce_mean(y_true * pos + (1.0 - y_true) * neg)


def cosine_distance_tf(x, y):
    """Cosine similarity logits between x(N,D) and y(M,D)."""
    x_exp = tf.expand_dims(x, axis=1)
    y_exp = tf.expand_dims(y, axis=0)
    x_n = tf.nn.l2_normalize(x_exp, axis=2)
    y_n = tf.nn.l2_normalize(y_exp, axis=2)
    return tf.reduce_sum(x_n * y_n, axis=2) * 50.0


def calculate_dist(x_out, cfg: CFG):
    """Return (self, original, inter logits, contrastive loss)."""
    support_y_pred, query_y_pred = x_out[0], x_out[1]
    D = tf.shape(support_y_pred)[1]

    support_feature_n = tf.reshape(support_y_pred, (cfg.n_way, cfg.support_n_shot, D))
    support_feature   = tf.reduce_mean(support_feature_n, axis=1)

    query_feature_n   = tf.reshape(query_y_pred, (cfg.n_way, cfg.query_n_shot, D))
    query_feature     = tf.reduce_mean(query_feature_n, axis=1)

    self_dist     = cosine_distance_tf(support_y_pred, support_feature)
    original_dist = cosine_distance_tf(query_y_pred,   support_feature)
    inter_dist    = cosine_distance_tf(support_y_pred, query_feature)

    query_labels   = tf.repeat(tf.range(cfg.n_way), cfg.query_n_shot)
    support_labels = tf.range(cfg.n_way)
    x_pair, y_pair, lbl = make_contrastive_pairs_tf(query_y_pred, query_labels,
                                                    support_feature, support_labels)
    cont_dist = squared_euclidean_distance_pair(x_pair, y_pair)
    cont_loss = contrastive_loss(lbl, cont_dist)

    return self_dist, original_dist, inter_dist, cont_loss


calculate_dist_val = calculate_dist  # same logic for validation


def calculate_loss(y_true_s, y_true_q, self_dist, original_dist, inter_dist, cont_original_dist_loss):
    """Weighted sum of cross-entropy losses + contrastive loss."""
    lossC = keras.losses.CategoricalCrossentropy(from_logits=True)
    self_loss     = lossC(y_true_s, self_dist)
    original_loss = lossC(y_true_q, original_dist)
    inter_loss    = lossC(y_true_s, inter_dist)
    return (self_loss * 2.0) + (original_loss * 1.0) + (inter_loss * 5.0) + (cont_original_dist_loss * 1.0)


def calculate_acc(y_true_q, original_dist):
    """Query accuracy."""
    y_true_q_ids = tf.argmax(y_true_q, axis=1)
    pred_q_ids   = tf.argmax(original_dist, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(pred_q_ids, y_true_q_ids), tf.float32))


# =========================
# Data helpers
# =========================
def load_npy_files_from_folder(folder_path, fold_num, cfg: CFG):
    random.seed(fold_num); np.random.seed(fold_num); tf.random.set_seed(fold_num)

    existing_files = sorted(os.listdir(folder_path))
    assert len(existing_files) >= cfg.files_per_fold, "Not enough files in folder."

    selected_files = random.sample(existing_files, cfg.files_per_fold)
    train_files = selected_files[:cfg.train_files]
    val_files   = selected_files[cfg.train_files:]

    train_split, val_split = [], []
    for f in train_files:
        arr = np.load(os.path.join(folder_path, f))
        train_split.append(arr / 255.0)
    for f in val_files:
        arr = np.load(os.path.join(folder_path, f))
        val_split.append(arr / 255.0)

    return train_split, val_split


def get_samples(data_list, cfg: CFG):
    chosen_classes = random.sample(data_list, cfg.n_way)
    support, query = [], []

    for data in chosen_classes:
        idx = np.random.choice(len(data), cfg.support_n_shot + cfg.query_n_shot, replace=False)
        support.extend(data[idx[:cfg.support_n_shot]])
        query.extend(data[idx[cfg.support_n_shot:]])

    y_s_ids = np.repeat(np.arange(cfg.n_way), cfg.support_n_shot)
    y_q_ids = np.repeat(np.arange(cfg.n_way), cfg.query_n_shot)
    y_s = tf.keras.utils.to_categorical(y_s_ids, num_classes=cfg.n_way)
    y_q = tf.keras.utils.to_categorical(y_q_ids, num_classes=cfg.n_way)

    return np.array(support), np.array(query), y_s, y_q


# =========================
# Models
# =========================
def build_vgg16(input_shape=(224, 224, 3), include_top=True):
    model = models.Sequential()
    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 4
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 5
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    if include_top:
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='linear'))
    return model


def get_siamese_model(base_model, input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    support_basic = base_model(left_input)
    query_basic   = base_model(right_input)
    return Model(inputs=[left_input, right_input], outputs=[support_basic, query_basic])
