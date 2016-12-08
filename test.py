from __future__ import absolute_import, division, print_function

import math, os, random, sys, time
import cPickle, gzip
import progressbar
import pprint
import glob, shutil

import numpy as np
from six.moves import xrange
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from tensorflow.models.rnn.translate import data_utils
import seq2seq_model

from rdkit import Chem
from rdkit.Chem import AllChem
import parser.Smipar as Smipar


pp = pprint.PrettyPrinter()
flags = tf.app.flags

flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
flags.DEFINE_integer("size", 600, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
flags.DEFINE_integer("reactant_vocab_size", 311, "Reactant vocabulary size.")
flags.DEFINE_integer("product_vocab_size", 180, "Product vocabulary size.")
flags.DEFINE_string("train_dir", "checkpoint/saved_models", "Training dir.")
flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = flags.FLAGS
pp.pprint(flags.FLAGS.__flags)

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

def cano(smiles): # canonicalize smiles by MolToSmiles function
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if (smiles != '') else ''

def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1

    enc_embed = 'embedding_attention_seq2seq/RNN/EmbeddingWrapper/embedding:0'
    dec_embed = 'embedding_attention_seq2seq/embedding_attention_decoder/embedding:0'
    e, d = sess.run([enc_embed, dec_embed])
    print(e.shape, d.shape)
    #plt.imshow(e, interpolation='nearest')
    # t-SNE visualization
    plt.figure(figsize=(18, 18))
    ptns = 50 # plotting numbers
    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
    e_2d = tsne.fit_transform(e)
    for i, label in enumerate(reactants_token_list[:ptns]):
      x, y = e_2d[i, :]
      plt.scatter(x, y)
      plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                    ha='right', va='bottom')
    #plt.show()

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    rsmi = sys.stdin.readline()
    
    while rsmi:
      reactant_list = []
      agent_list = []
      split_rsmi = rsmi.split('>')
      reactants = cano(split_rsmi[0]).split('.')
      agents = cano(split_rsmi[1]).split('.')

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
      attn, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out the reaction smiles
      products = ''.join([tf.compat.as_str(products_token_list[output])
      											for output in outputs])

      attn_matrix = np.squeeze(np.stack(attn))
      print(attn_matrix.shape)
      plt.imshow(attn_matrix, interpolation='nearest')
      plt.show()

      print(products)
      print("> ", end="")
      sys.stdout.flush()
      rsmi = sys.stdin.readline()

def main(_):
	decode()

if __name__ == "__main__":
	tf.app.run()
