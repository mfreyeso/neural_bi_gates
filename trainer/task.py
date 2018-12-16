import argparse
import glob
import os

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model

from tensorflow.python.lib.io import file_io

import trainer.model as model

CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
LOGICAL_NN_MODEL = 'neural_gates.hdf5'


class ContinuousEval(Callback):
    """Continuous eval callback to evaluate the checkpoint once every so many epochs."""
    def __init__(self, eval_frequency, learning_rate, job_dir, steps):
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.steps = args.eval_steps

    def on_epoch_begin(self, epoch, logs={}):
        """Compile and save model."""
        if epoch > 0 and epoch % self.eval_frequency == 0:
            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith('gs://'):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                logic_nn_model = load_model(checkpoints[-1])
                logic_nn_model = model.compile_model(logic_nn_model, self.learning_rate)
                loss, acc = logic_nn_model.evaluate_generator(
                    model.generator_input(), steps=self.steps)
                print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
                    epoch, loss, acc, logic_nn_model.metrics_names))
                if self.job_dir.startswith('gs://'):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))


def train_and_evaluate(args):
    logic_nn_model = model.model_fn(**vars(args))
    try:
        os.makedirs(args.job_dir)
    except:
        pass

    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = CHECKPOINT_FILE_PATH
    if not args.job_dir.startswith('gs://'):
        checkpoint_path = os.path.join(args.job_dir, checkpoint_path)

    # Model checkpoint callback.
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=args.checkpoint_epochs,
        mode='min')

    # Continuous eval callback.
    evaluation = ContinuousEval(args.eval_frequency, args.learning_rate, args.job_dir, args.eval_steps)

    # Tensorboard logs callback.
    tb_log = TensorBoard(
        log_dir=os.path.join(args.job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, evaluation, tb_log]

    logic_nn_model.fit_generator(
        model.generator_input(),
        steps_per_epoch=args.train_steps,
        epochs=args.num_epochs,
        callbacks=callbacks)

    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if args.job_dir.startswith('gs://'):
        logic_nn_model.save(LOGICAL_NN_MODEL)
        copy_file_to_gcs(args.job_dir, LOGICAL_NN_MODEL)
    else:
        logic_nn_model.save(os.path.join(args.job_dir, LOGICAL_NN_MODEL))

    # Convert the Keras model to TensorFlow SavedModel.
    model.to_saved_model(logic_nn_model, os.path.join(args.job_dir, 'export'))
    print("...Finished actions for build and train with model...")


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='/tmp/neural-gates-keras')
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int)
    parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help="""\
            Maximum number of training steps to perform
            Training steps are in the units of training-batch-size.
            So if train-steps is 500 and train-batch-size if 100 then
            at most 500 * 100 training instances will be used to train.""")
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.3,
        help='Learning rate for SGD')
    parser.add_argument(
        '--eval-frequency',
        default=10,
        help='Perform one evaluation per n epochs')
    parser.add_argument(
        '--eval-num-epochs',
        type=int,
        default=5,
        help='Number of epochs during evaluation')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Maximum number of epochs on which to train')
    parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        default=5,
        help='Checkpoint per n training epochs')

    args, _ = parser.parse_known_args()
    train_and_evaluate(args)
