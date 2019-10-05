import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import codecs
import json
import logging
import numpy as np
import os
import re
import subprocess
import inspect 

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from model_def import get_model, HEIGHT, WIDTH, DEPTH, NUM_CLASSES
from utilities import process_input

import tensorflow
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
print(optimizer_v2.OptimizerV2.from_config)

def from_config(cls, config=None, custom_objects=None):
    print("What is going on here\n")
#     assert False, cls
    print(cls)
    print(config)
    print("Are you done?\n")
    config = config.copy()  # Make a copy, since we mutate config
    config['optimizer'] = optimizers.deserialize(
        config['optimizer'], custom_objects=custom_objects)
    config['loss_scale'] = keras_loss_scale_module.deserialize(
        config['loss_scale'], custom_objects=custom_objects)
    print(inspect.getargspec(cls)[0])
    return cls(**config)

# optimizer_v2.OptimizerV2.from_config = from_config



from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import saving
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.util.tf_export import tf_export

# API entries importable from `keras.models`:
Model = training.Model  # pylint: disable=invalid-name
Sequential = sequential.Sequential  # pylint: disable=invalid-name
save_model = saving.save_model
load_model = saving.load_model
model_from_config = saving.model_from_config
model_from_yaml = saving.model_from_yaml
model_from_json = saving.model_from_json


from tensorflow.python.keras import models


def clone_and_build_model(
    model, input_tensors=None, target_tensors=None, custom_objects=None,
    compile_clone=True, in_place_reset=False, optimizer_iterations=None):
  """1.13"""
  if compile_clone and not model.optimizer:
    raise ValueError(
        'Error when cloning model: compile_clone was set to True, but the '
        'original model has not been compiled.')

  if model._is_graph_network or isinstance(model, Sequential):
    if custom_objects:
      with CustomObjectScope(custom_objects):
        clone = models.clone_model(model, input_tensors=input_tensors)
    else:
      clone = models.clone_model(model, input_tensors=input_tensors)

    if all([isinstance(clone, Sequential),
            not models.clone._is_graph_network,
            getattr(model, '_build_input_shape', None) is not None]):
      clone._set_inputs(
          K.placeholder(model._build_input_shape, dtype=model.inputs[0].dtype))
  else:
    if not in_place_reset:
      raise ValueError('.')
    clone = model
    _in_place_subclassed_model_reset(clone)
    if input_tensors is not None:
      if isinstance(input_tensors, (list, tuple)) and len(input_tensors) == 1:
        input_tensors = input_tensors[0]
      models.clone._set_inputs(input_tensors)

  if compile_clone and model.optimizer:
    if isinstance(model.optimizer, optimizers.TFOptimizer):
      optimizer = optimizers.TFOptimizer(
          model.optimizer.optimizer, optimizer_iterations)
      K.track_tf_optimizer(optimizer)
    else:
      optimizer_config = model.optimizer.get_config()
      optimizer = model.optimizer.__class__.from_config(optimizer_config)
      if optimizer_iterations is not None:
        optimizer.iterations = optimizer_iterations

    models.clone.compile(
        optimizer,
        model.loss,
        metrics=metrics_module.clone_metrics(model._compile_metrics),
        loss_weights=model.loss_weights,
        sample_weight_mode=model.sample_weight_mode,
        weighted_metrics=metrics_module.clone_metrics(
            model._compile_weighted_metrics),
        target_tensors=target_tensors)

  return clone



models.clone_and_build_model = clone_and_build_model


logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)


#  Copy inference pre/post-processing script so it will be included in the model package
os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')


class CustomTensorBoardCallback(TensorBoard):
    
    def on_batch_end(self, batch, logs=None):
        pass

    
def save_history(path, history):

    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            history_for_json[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float32 or type(history.history[key][0]) == np.float64:
               history_for_json[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(',', ':'), sort_keys=True, indent=4) 


def save_model(model, output):

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    print("\n\n")
    print(model.optimizer)
    print(model.optimizer.__init__)
    print("\n\n")
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    logging.info("Model successfully saved at: {}".format(output))
    return


def main(args):

    mpi = False
    if 'sourcedir.tar.gz' in args.tensorboard_dir:
        tensorboard_dir = re.sub('source/sourcedir.tar.gz', 'model', args.tensorboard_dir)
    else:
        tensorboard_dir = args.tensorboard_dir
    logging.info("Writing TensorBoard logs to {}".format(tensorboard_dir))
    
    if 'sagemaker_mpi_enabled' in args.fw_params:
        if args.fw_params['sagemaker_mpi_enabled']:
            import horovod.tensorflow.keras as hvd
            mpi = True
            # Horovod: initialize Horovod.
            hvd.init()

            # Horovod: pin GPU to be used to process local rank (one GPU per process)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = str(hvd.local_rank())
            K.set_session(tf.Session(config=config))
    else:
        hvd = None

    logging.info("Running with MPI={}".format(mpi))
    logging.info("getting data")
    train_dataset = process_input(args.epochs, args.batch_size, args.train, 'train', args.data_config)
    eval_dataset = process_input(args.epochs, args.batch_size, args.eval, 'eval', args.data_config)
    validation_dataset = process_input(args.epochs, args.batch_size, args.validation, 'validation', args.data_config)

    logging.info("configuring model")
    model = get_model(args.learning_rate, args.weight_decay, args.optimizer, args.momentum, 1, mpi, hvd)
    callbacks = []
    if mpi:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(args.output_data_dir + '/checkpoint-{epoch}.h5'))
            callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))
    else:
        callbacks.append(ModelCheckpoint(args.output_data_dir + '/checkpoint-{epoch}.h5'))
        callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))
        
    logging.info("Starting training")
    size = 1
    if mpi:
        size = hvd.size()
        
    history = model.fit(x=train_dataset[0], 
              y=train_dataset[1],
              steps_per_epoch=(num_examples_per_epoch('train') // args.batch_size) // size,
              epochs=args.epochs, 
              validation_data=validation_dataset,
              validation_steps=(num_examples_per_epoch('validation') // args.batch_size) // size,
              callbacks=callbacks)

    score = model.evaluate(eval_dataset[0], 
                           eval_dataset[1], 
                           steps=num_examples_per_epoch('eval') // args.batch_size,
                           verbose=0)

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))

    # Horovod: Save model and history only on worker 0 (i.e. master)
    if mpi:
        if hvd.rank() == 0:
#             save_history(args.model_dir + "/hvd_history.p", history)
            subprocess.call("rm -rf /opt/ml/model/*", shell=True)
            return save_model(model, args.model_output_dir)
    else:
#         save_history(args.model_dir + "/hvd_history.p", history)
        subprocess.call("rm -rf /opt/ml/model/*", shell=True)
        return save_model(model, args.model_output_dir)


def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 40000
    elif subset == 'validation':
        return 10000
    elif subset == 'eval':
        return 10000
    else:
        raise ValueError('Invalid data subset "%s"' % subset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir',type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir',type=str,default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay',type=float,default=2e-4,help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--data-config',type=json.loads,default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--momentum',type=float,default='0.9')
    
    args = parser.parse_args()

    main(args)
