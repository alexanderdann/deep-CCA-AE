import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time


def CCA(view1, view2, shared_dim):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    r1 = 0.0
    r2 = 0.0

    assert V1.shape[0] == V2.shape[0]
    M = tf.constant(V1.shape[0], dtype=tf.float32)
    ddim_1 = tf.constant(V1.shape[1], dtype=tf.int16)
    ddim_2 = tf.constant(V2.shape[1], dtype=tf.int16)
    # check mean and variance

    mean_V1 = tf.reduce_mean(V1, 0)
    mean_V2 = tf.reduce_mean(V2, 0)

    V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
    V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) #+ r1 * tf.eye(ddim_1)
    Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) #+ r2 * tf.eye(ddim_2)

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

    C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    D, U, V = tf.linalg.svd(C, full_matrices=True)

    A = tf.matmul(tf.transpose(U)[:shared_dim], Sigma11_root_inv)
    B = tf.matmul(tf.transpose(V)[:shared_dim], Sigma22_root_inv)

    epsilon = tf.matmul(A, tf.transpose(V1_bar))
    omega = tf.matmul(B, tf.transpose(V2_bar))

    return A, B, epsilon, omega, D


def PCC_Matrix(view1, view2, observations):
    assert tf.shape(view1)[1] == observations
    assert tf.shape(view1)[1] == tf.shape(view2)[1]
    calc_cov = tfp.stats.correlation(view1, view2, sample_axis=1, event_axis=0)

    return tf.math.abs(calc_cov), tf.shape(view1)[0], tf.shape(view2)[0]
