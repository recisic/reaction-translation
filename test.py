from __future__ import absolute_import, division, print_function

import math, os, random, sys, time
import cPickle, gzip
import progressbar

import numpy as np
from six.moves import xrange
import tensorflow as tf

from rdkit.Chem import AllChem
import parser.Smipar as Smipar

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 600, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("reactant_vocab_size", 326, "Reactant vocabulary size.")
tf.app.flags.DEFINE_integer("product_vocab_size", 197, "Product vocabulary size.")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(54, 54), (70, 60), (90, 65), (150, 80)]

# vocab loader

with gzip.open('data/vocab/vocab_list.pkl.gz', 'rb') as list_file:
    reactants_token_list, products_token_list = cPickle.load(list_file)


def create_model(session, forward_only):
  model = seq2seq_model.Seq2SeqModel(
    FLAGS.reactant_vocab_size, FLAGS.product_vocab_size, _buckets,
    FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
    FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
    forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    rsmi = sys.stdin.readline()
    while rsmi:
      # Get token-ids for the input sentence.
      reactant_list = []
      agent_list = []
      split_rsmi = rsmi.split('>')
      reactants = split_rsmi[0].split('.')
      agents = split_rsmi[1].split('.')
      # TODO: remove mapping and normalize before process
      for reactant in reactants:
        reactant_list += Smipar.parser_list(reactant)
        reactant_list += '.'
      for agent in agents:
        agent_list += Smipar.parser_list(agent)
        agent_list += '.'
      reactant_list.pop() # to pop last '.'
      agent_list.pop()
      reactant_list += '>'
      reactant_list += agent_list
      token_ids = [reactants_token_list.index(r) for r in reactant_list]
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out the reaction smiles
      products = ''.join([tf.compat.as_str(products_token_list[output])
      											for output in outputs])
      print(products)
      print("> ", end="")
      sys.stdout.flush()
      rsmi = sys.stdin.readline()


def main(_):
	decode()

if __name__ == "__main__":
	tf.app.run()
