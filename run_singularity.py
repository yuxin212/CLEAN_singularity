import os
import sys
import pathlib
import signal
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
from spython.main import Client

import tempfile
import subprocess

singularity_image = Client.load(os.path.join(os.environ['CLEAN_DIR'], 'train_clean.sif'))

# Path to a directory that will store the results.
if 'TMP' in os.environ:
    output_dir = os.environ['TMP']
elif 'TMPDIR' in os.environ:
    output_dir = os.environ['TMPDIR']
else:
    output_dir = tempfile.mkdtemp(dir='/tmp', prefix='train-clean-')

# set tmp dir the same as output dir
tmp_dir = output_dir

#### END USER CONFIGURATION ####


flags.DEFINE_bool(
    'use_gpu', True, 'whether run with GPUs'
)
flags.DEFINE_float(
    'learning_rate', 5e-4, 'learning rate'
)
flags.DEFINE_integer(
    'epoch', 2000, 'number of epochs'
)
flags.DEFINE_string(
    'model_name', 'split10_triplet', 'name of the model'
)
flags.DEFINE_string(
    'training_data', 'split10', 'name of the training data'
)
flags.DEFINE_integer(
    'hidden_dim', 512, 'hidden dimension'
)
flags.DEFINE_integer(
    'out_dim', 128, 'output dimension'
)
flags.DEFINE_integer(
    'adaptive_rate', 100, 'adaptive rate'
)
flags.DEFINE_bool(
    'verbose', False, 'whether print verbose'
)
FLAGS = flags.FLAGS

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  binds = []
  command_args = []

  binds.extend([
     f'{tmp_dir}/CLEAN/model/:/app/CLEAN/data/model/',
     f'{tmp_dir}/CLEAN/esm_data/:/app/CLEAN/data/esm_data/',
     f'{tmp_dir}/CLEAN/distance_map/:/app/CLEAN/data/distance_map/',
     f'{tmp_dir}/CLEAN/:/app/CLEAN/data/'
  ])

  command_args.extend([
        f'-l={FLAGS.learning_rate}',
        f'-e={FLAGS.epoch}',
        f'-n={FLAGS.model_name}',
        f'-t={FLAGS.training_data}',
        f'-d={FLAGS.hidden_dim}',
        f'-o={FLAGS.out_dim}',
        f'--adaptive_rate={FLAGS.adaptive_rate}',
        f'--verbose={FLAGS.verbose}',
  ])

  options = [
    '--bind', f'{",".join(binds)}',
    '--pwd', '/app/CLEAN',
  ]

  # Run the container.
  # Result is a dict with keys "message" (value = all output as a single string),
  # and "return_code" (value = integer return code)
  result = Client.run(
             singularity_image,
             command_args,
             nv=True if FLAGS.use_gpu else None,
             return_result=True,
             options=options
           )


if __name__ == '__main__':
  app.run(main)