import numpy as np
import tensorflow as tf
import scipy

from correlation_analysis import CCA

def compute_distance_metric(S, U):
    Ps = np.eye(S.shape[1]) - tf.transpose(S)@np.linalg.inv(S@tf.transpose(S))@S
    Q = scipy.linalg.orth(tf.transpose(U))
    dist = np.linalg.norm(Ps@Q, ord=2)
    return dist


def compute_similarity_metric_v1(S, U):
    _, _, _, _, ccor = CCA(tf.transpose(S), tf.transpose(U), 5)
    return np.mean(ccor)


def compute_similarity_metric_v2(S1, U1, S2, U2):
    _, _, _, _, ccor_1 = CCA(tf.transpose(S1), tf.transpose(U1), 5)
    _, _, _, _, ccor_2 = CCA(tf.transpose(S2), tf.transpose(U2), 5)
    return np.mean(ccor_1+ccor_2)


def compute_rademacher_and_L1(model):
    inter_w_ids = list()
    inter_b_ids = list()

    for idx, trainable_var in enumerate(model.trainable_variables):
        if 'Encoder' in trainable_var.name:
            if 'kernel' in trainable_var.name:
                inter_w_ids.append(idx)
            elif 'bias' in trainable_var.name:
                inter_b_ids.append(idx)
            else:
                raise IOError

    inter_w_vars = [model.trainable_variables[idx] for idx in inter_w_ids]
    inter_b_vars = [model.trainable_variables[idx] for idx in inter_b_ids]

    for idx, var in enumerate(zip(inter_w_vars, inter_b_vars)):
        assert var[0].name.split('/')[0] == var[1].name.split('/')[0]

    L1_terms, Rademacher_terms = list(), list()
    for idx, var in enumerate(zip(inter_w_vars, inter_b_vars)):
        Rademacher_terms.append(tf.math.reduce_max(
            tf.norm(tf.concat([tf.squeeze(var[0]), var[1]], axis=0), ord=np.inf, axis=0)))
        L1_terms.append(tf.reduce_sum(tf.abs(tf.concat([tf.squeeze(var[0]), var[1]], axis=0))))

    L1_loss = tf.reduce_sum(L1_terms)
    Rademacher_loss = tf.math.reduce_prod(Rademacher_terms)

    return L1_loss, Rademacher_loss