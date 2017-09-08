from __future__ import print_function
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

import TensorflowUtils as utils
import read_PascalVocData as dataset
#import read_MITSceneParsingData as dataset
import datetime
import BatchDatsetReader as batchdata
import inception_v3_fcn
from six.moves import xrange
import colorize


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "10", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/all", "path to logs directory")
#tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("data_dir", "Data_zoo/Pascal_Voc/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "0.00032", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode: train/ test/ visualize")
tf.flags.DEFINE_string(
    'train_dir', 'logs/all',
    'Directory where checkpoints and event logs are written to.')
tf.flags.DEFINE_string(
    'skip_layers', '8s',
    'Skip architecture to use: 32s/ 16s/ 8s')

#####################
# Fine-Tuning Flags #
#####################
tf.flags.DEFINE_string(
    'checkpoint_path', 'logs',
    'The path to a checkpoint from which to fine-tune.')
# this should be 'logs', used to be None

tf.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'InceptionV3/Logits,InceptionV3/AuxLogits',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
# this should be 'InceptionV3/Logits,InceptionV3/AuxLogits', used to be None

tf.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
# this should be 'InceptionV3/Logits,InceptionV3/Upsampling', used to be None

tf.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
# this may be True


MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 22
IMAGE_SIZE = 224


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore, ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits, end_points = inception_v3_fcn.inception_v3_fcn(image, num_classes=NUM_OF_CLASSESS, dropout_keep_prob=keep_probability, skip=FLAGS.skip_layers)
    probabilities = tf.nn.softmax(logits)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("pixelwise loss", loss)

    # TODO change the variables that are trainable
    # Variables to train.
    variables_to_train = _get_variables_to_train()
    if FLAGS.debug:
        for var in variables_to_train:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, variables_to_train)


    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = dataset.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = batchdata.BatchDatset(train_records, image_options)
    validation_dataset_reader = batchdata.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')
    sess.run(tf.global_variables_initializer())
    # maybe this is the place to do init_fn
    # TODO i dont know how to add init_op in this framework
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        init_fn = _get_init_fn()
        init_fn(sess)

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 100 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred, prob = sess.run([pred_annotation, probabilities], feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(itr))
            colorize.save_result(str(itr), valid_images[itr], prob[itr], NUM_OF_CLASSESS, True)
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "test":
        valid_images, valid_annotations = validation_dataset_reader.get_consecutive_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        print(pred.shape)

        t_px = np.zeros(NUM_OF_CLASSESS)
        n_px = np.zeros(NUM_OF_CLASSESS)
        n1_px = np.zeros(NUM_OF_CLASSESS)

        for itr in range(FLAGS.batch_size):
            tmp_t, tmp_n, tmp_n1 = colorize.single_metrics(valid_annotations[itr].astype(np.uint8), pred[itr].astype(np.uint8), NUM_OF_CLASSESS)
            t_px += tmp_t
            n_px += tmp_n
            n1_px += tmp_n1

        t_sum = np.sum(t_px)
        n_sum = np.sum(n_px)
        px_acc = n_sum/t_sum
        condition_1 = t_px != 0
        c_n1 = np.extract(condition_1, n_px)
        c_t1 = np.extract(condition_1, t_px)
        condition_2 = (t_px + n1_px - n_px) != 0
        c_n2 = np.extract(condition_2, n_px)
        c_d2 = np.extract(condition_2, (t_px + n1_px - n_px))
        mean_acc = np.sum(np.divide(c_n1, c_t1))/NUM_OF_CLASSESS
        mean_IU = np.sum(np.divide(c_n2, c_d2))/NUM_OF_CLASSESS
        fw_IU = np.sum(np.divide(np.extract(condition_2, np.multiply(t_px, n_px)), c_d2))/t_sum

        text_file = open("metrics.txt", "w")
        text_file.write("pixel accuracy: %s\n" % str(px_acc))
        text_file.write("mean accuracy: %s\n" % str(mean_acc))
        text_file.write("mean IU: %s\n" % str(mean_IU))
        text_file.write("frequency weighted IU: %s" % str(fw_IU))
        text_file.close()
        print("Successfully write metrics to text file")
  
        print("========= metrics =========")
        print("pixel accuracy: " + str(px_acc))
        print("mean accuracy: " + str(mean_acc))
        print("mean IU: " + str(mean_IU))
        print("frequency weighted IU: " + str(fw_IU))
        print("")

if __name__ == "__main__":
    tf.app.run()
