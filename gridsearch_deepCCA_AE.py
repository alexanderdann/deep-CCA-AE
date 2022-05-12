import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from correlation_analysis import CCA
from models import build_deepCCA_AE_model, compute_loss, compute_regularization, compute_reconstruction
from tensorboard_utillities import write_scalar_summary, write_image_summary, write_PCC_summary, write_gradients_summary_mean, write_poly
from tensorboard_utillities import create_grid_writer
from utilities import load_data, prepare_data, track_validation_progress, track_test_progress

np.random.seed(3333)


def train(hidden_layers, ae_layers, shared_dim, data, max_epochs, log_path, model_path, batch_size=None, lambda_reg=1e-6, lambda_rec=1e-6, activation='sigmoid', iter_idx=1):
    '''
        Used to capture information about architecture and log this information
    '''
    params = ['deepCCA-AE', f'v{iter_idx}', 
              f'{len(hidden_layers)} layers',
              f'{hidden_layers[0]} hidden nodes',
              f'{ae_layers[0]} ae nodes',
              f'shared dim {shared_dim}',
              f'reg-lambda {str(lambda_reg)}', 
              f'rec-lambda {str(lambda_rec)}',
              f'activation {activation}']
    
    writer = create_grid_writer(root_dir=log_path, params=params)
    
    '''
        Preparation of data. More information in the file utilities.py. Only used full batch approach
        thus always used batch_size=None.
    '''
    final_data, batch_sizes = prepare_data(data, batch_size)
    num_views = len(final_data)
    
    '''
        Used to extract information of input dimensions based on the data given, so no specification
        of input dimensions is needed.
    '''
    input_dims = list()
    for idx, chunk in enumerate(final_data):
        _, _, dim = tf.shape(chunk['train'])
        input_dims.append(int(dim))
    
    model = build_deepCCA_AE_model(input_dims, hidden_layers, ae_layers, shared_dim, activation)
    termination_condition = False
    score_history = dict(zip(['validation'], [list()]))
  
    
    for epoch in range(max_epochs):
        if termination_condition:
            break
        
        intermediate_outputs = list()
        
        for batch_idx in range(batch_sizes['train']):
            y_1, y_2 = final_data[0]['train'],  final_data[1]['train']
            batch_y1, batch_y2 = y_1[batch_idx], y_2[batch_idx]
    
            with tf.GradientTape() as tape:
                tape.watch([batch_y1, batch_y2])
                [fy_1, fy_2], [rec_y_1, rec_y_2] = model([batch_y1, batch_y2])
                cca_loss = compute_loss(fy_1, fy_2)
                reg_loss = compute_regularization(model, lambda_reg=lambda_reg)
                rec_loss = compute_reconstruction(batch_y1, batch_y2, rec_y_1, rec_y_2, lambda_param=lambda_rec)

                if epoch > 1:
                    loss = cca_loss + reg_loss + rec_loss
                else:
                    loss = cca_loss + rec_loss
                    
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                intermediate_outputs.append((fy_1, fy_2))
        
        
        if epoch % 10 == 0:
            '''
                Using all data to track the SVM accuracy and different losses over time in TensorBoard.
            '''
            validation_score = track_validation_progress(model, final_data, data[0]['train']['labels'], data[0]['validation']['labels'])    
            score_history['validation'].append(validation_score)
            
            static_part = [(loss, 'Loss/Total'),
                           (cca_loss, 'Loss/CCA'),
                           (reg_loss, 'Loss/Regularization'),
                           (rec_loss, 'Loss/Reconstruction'),
                           (validation_score, 'Score/Validation Accuracy')
                          ]
            
            write_scalar_summary(
                writer=writer,
                epoch=epoch,
                list_of_tuples=static_part
            )
                
        if epoch % 75 == 0:
            tmp = list()
            for batch_idx in range(batch_sizes['train']):
                batched_fy_1, batched_fy_2 = intermediate_outputs[batch_idx]
                B1, B2, epsilon, omega, ccor = CCA(batched_fy_1, batched_fy_2)
                tmp.append(ccor)
                
            avg_ccor = tf.math.reduce_mean(tmp, axis=0)
            dynamic_part = [(cval, f'Canonical correlation/{idx})') for idx, cval in enumerate(avg_ccor)]
            
            write_scalar_summary(
                writer=writer,
                epoch=epoch,
                list_of_tuples=dynamic_part
            )
        
        if epoch > 100:
            if (tf.math.reduce_std(score_history['validation'][-100:]) < 1e-3):
                std = tf.math.reduce_std(score_history['validation'][-100:])
                termination_condition = True
    
    '''
        Save the whole architecture in a final step.
    '''
    try:
        os.makedirs(model_path)
    except FileExistsError:
        print('MODELS PATH exists, saving data.')
    finally:
        model_name = '-'.join(params)
        model.save(f'{model_path}/{model_name}', overwrite=True)
            
    return track_test_progress(model, final_data, data[0]['train']['labels'], data[0]['validation']['labels'], data[0]['test']['labels'])


desc = f'GridSearch'
LOGROOT = f'{os.getcwd()}/LOG/{desc}'
MODELSPATH = f'{os.getcwd()}/MODELS/{desc}'

eeg_data, meg_data, labels = load_data(artefact_removal=True)
num_folds = 5

num_layers = [2]
shared_dims = [5, 10]
lambdas_rec = [1e-4, 1e-5, 1e-6]
lambdas_reg = [1e-3, 1e-4]
hidden_dims = [256, 512]
ae_num_layers = [1, 2]
activation_functions = ['sigmoid']

HP_NUM_LAYERS = hp.HParam('number of layers', hp.Discrete(num_layers))
HP_HIDDEN_DIMENSION = hp.HParam('hidden dimension', hp.Discrete(hidden_dims))
HP_AE_LAYERS = hp.HParam('number of ae layers', hp.Discrete(ae_num_layers))
HP_SHARED_DIMENSION = hp.HParam('shared dimension', hp.Discrete(shared_dims))
HP_LAMBDA_REC = hp.HParam('rec-lambda', hp.Discrete(lambdas_rec))
HP_LAMBDA_REG = hp.HParam('reg-lambda', hp.Discrete(lambdas_reg))
HP_ACTIVATION = hp.HParam('activation function', hp.Discrete(activation_functions))

METRIC_ACCURACY = 'Accuracy'

hyperparameters = [HP_NUM_LAYERS, HP_HIDDEN_DIMENSION, HP_AE_LAYERS, HP_SHARED_DIMENSION, HP_LAMBDA_REC, HP_LAMBDA_REG, HP_ACTIVATION]
hp_metric = [hp.Metric(METRIC_ACCURACY, display_name=f'Mean SVM accuracy over {num_folds}-folds')]

with tf.summary.create_file_writer(LOGROOT).as_default():
    hp.hparams_config(
    hparams=hyperparameters,
    metrics=hp_metric,
  )

def start_grid_search(hparams):
    tf.random.set_seed(3333)
    accs = list()
    for fold_idx in range(num_folds):
        
        try:
            fin_acc = train(hidden_layers=[hparams[HP_HIDDEN_DIMENSION] for _ in range(hparams[HP_NUM_LAYERS])], 
                            ae_layers = [hparams[HP_HIDDEN_DIMENSION] for _ in range(hparams[HP_AE_LAYERS])],
                            shared_dim=hparams[HP_SHARED_DIMENSION],
                            activation=hparams[HP_ACTIVATION],
                            data=[eeg_data[fold_idx], meg_data[fold_idx]],
			    lambda_reg=hparams[HP_LAMBDA_REG],
                            lambda_rec=hparams[HP_LAMBDA_REC],
                            max_epochs=5000, 
                            log_path=LOGPATH, model_path=MODELSPATH, 
                            batch_size=None, 
                            iter_idx=fold_idx)
            accs.append(fin_acc)
        
        except Exception as e:
            print(e)
            accs.append(0)
            
    return tf.math.reduce_mean(accs)


def run(log_path, hparams):
    with tf.summary.create_file_writer(log_path).as_default():
        hp.hparams(hparams)
        accuracy = start_grid_search(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

        
session_num = 0
for num_layers_idx, num_layers in enumerate(HP_NUM_LAYERS.domain.values, 0):
    for hdim_idx, hdim in enumerate(HP_HIDDEN_DIMENSION.domain.values, 0):
        for ae_idx, ae_num_layers in enumerate(HP_AE_LAYERS.domain.values, 0):
            for sdim_idx, sdim in enumerate(HP_SHARED_DIMENSION.domain.values, 0):
                for lrec_idx, lrec in enumerate(HP_LAMBDA_REC.domain.values, 0):
                    for lreg_idx, lreg in enumerate(HP_LAMBDA_REG.domain.values, 0):
                    	for afunc_idx, afunc in enumerate(HP_ACTIVATION.domain.values, 0):

                            hparams = {
                                HP_NUM_LAYERS: num_layers,
                                HP_HIDDEN_DIMENSION: hdim,
                                HP_AE_LAYERS: ae_num_layers,
                                HP_SHARED_DIMENSION: sdim,
                                HP_LAMBDA_REC: lrec,
                                HP_LAMBDA_REG: lreg,
                                HP_ACTIVATION: afunc
                            }

                            LOGPATH = f'{LOGROOT}/Grid {session_num}'
                            run(LOGPATH, hparams)

                            session_num += 1
