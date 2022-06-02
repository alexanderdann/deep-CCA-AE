import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.svm import LinearSVC as SVM
from sklearn.metrics import accuracy_score


def _create_mlp(input_dim, hidden_layers, ae_layers, shared_dim, view_idx, activation='sigmoid'):
    '''
        Creating one view. Hidden layers and AE part are connected by repeatedly reassigning 
        a temporary variable and using knowledge about the network structure.
    '''
    input_layer = tf.keras.Input(shape=(input_dim, ), name=f'view{view_idx}_input_layer')
    
    for layer_idx, layer_dim in enumerate(hidden_layers[view_idx], start=1):
        if layer_idx == 1:
            tmp_hidden_layer = tf.keras.layers.Dense(layer_dim, activation, name=f'view_{view_idx}_hidden_layer_{layer_idx}')\
            (input_layer)
            
        else:
            tmp_hidden_layer = tf.keras.layers.Dense(layer_dim, activation, name=f'view_{view_idx}_hidden_layer_{layer_idx}')\
            (tmp_hidden_layer)
            
    output_layer = tf.keras.layers.Dense(shared_dim, activation='linear', name=f'view_{view_idx}_output_layer')\
                    (tmp_hidden_layer)
    
    for layer_idx, layer_dim in enumerate(ae_layers[view_idx], start=1):
        if layer_idx == 1:
            tmp_ae_layer = tf.keras.layers.Dense(layer_dim, activation, name=f'view_{view_idx}_ae_layer_{layer_idx}')\
                (output_layer)
            
        else:
            tmp_ae_layer = tf.keras.layers.Dense(layer_dim, activation, name=f'view_{view_idx}_ae_layer_{layer_idx}')\
                (tmp_ae_layer)
            
    ae_output_layer = tf.keras.layers.Dense(input_dim, activation='linear', name=f'view_{view_idx}_ae_output_layer')\
                    (tmp_ae_layer)
    
    return input_layer, [output_layer, ae_output_layer]


def build_deepCCA_AE_model(input_dims, hidden_layers, ae_layers, shared_dim, activation, learning_rate=0.001, display_model=False):
    '''
        input_dims:     type is list with integers, len(input_dims) >= 2
        hidden_layers:  type is list with integers for specifying nodes per layer, same for all views
        ae_layers:      type is list with integers for specifying nodes for each Auto Encoder layers
        shared_dim:     final output dimension, integer, same for both views
        activation:     activation function of interest, string (i.e. 'sigmoid, 'relu', ..)
    '''
    inputs, outputs, ae_outputs = list(), list(), list()
    
    for view_idx, view_input_dim in enumerate(input_dims):
        input_layer, [output_layer, ae_output_layer] = _create_mlp(view_input_dim, hidden_layers, ae_layers, shared_dim, view_idx, activation)
        inputs.append(input_layer)
        outputs.append(output_layer)
        ae_outputs.append(ae_output_layer)

    model = tf.keras.Model(inputs=inputs, outputs=[outputs, ae_outputs], name='deepCCA-AE')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    model.summary()
    
    if display_model:
        plot_model(model, to_file='deepCCA-AE model.png', show_shapes=True, show_layer_activations=True)

    return model


def compute_loss(view1, view2, reg_param):
    V1 = tf.cast(view1, dtype=tf.float32)
    V2 = tf.cast(view2, dtype=tf.float32)

    assert V1.shape[0] == V2.shape[0]

    M = tf.constant(V1.shape[0], dtype=tf.float32)
    ddim = tf.constant(V1.shape[1], dtype=tf.int16)

    meanV1 = tf.reduce_mean(V1, axis=0, keepdims=True)
    meanV2 = tf.reduce_mean(V2, axis=0, keepdims=True)

    V1_bar = V1 - tf.tile(meanV1, [M, 1])
    V2_bar = V2 - tf.tile(meanV2, [M, 1])

    Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
    Sigma11 = tf.add(tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1), reg_param * tf.eye(ddim))
    Sigma22 = tf.add(tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1), reg_param * tf.eye(ddim))

    Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
    Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
    Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)
    
    T = tf.matmul(tf.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
    TT = tf.matmul(tf.transpose(T), T)
    corr = tf.linalg.trace(tf.linalg.sqrtm(TT))
    return -corr


def compute_reconstruction(inputs_view1, inputs_view2, rec_view1, rec_view2, lambda_param):
    rec1 = tf.norm(tf.math.subtract(inputs_view1, rec_view1), ord=2, axis=1)
    rec2 = tf.norm(tf.math.subtract(inputs_view2, rec_view2), ord=2, axis=1)
    return lambda_param * tf.math.reduce_mean(tf.add(rec1, rec2))

    
def compute_regularization(model, lambda_reg):
    regularizer = tf.keras.regularizers.L2(lambda_reg)
    W = np.array([])
    
    for trainable_var in model.trainable_variables:
        W = np.append(W, tf.reshape(trainable_var, [-1]).numpy())
            
    return regularizer(tf.convert_to_tensor(W, dtype=tf.float32))


