# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import math, os, random, sys, time
import cPickle, gzip
import progressbar
import pprint
import glob, shutil

import numpy as np
from six.moves import xrange
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model

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
flags.DEFINE_string("train_dir", "checkpoint", "Training directory.")
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


dev_set = [[] for _ in _buckets]
dev_gen_set = [[] for _ in _buckets]
train_set = [[] for _ in _buckets]


with gzip.open('data/dev.pkl.gz', 'rb') as dev_file:
    while 1:
        try:
            reactants, products = cPickle.load(dev_file)
            products.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(reactants) < source_size and len(products) < target_size:
                    dev_set[bucket_id].append([reactants, products])
                    break
        except EOFError:
            break

print("dev_set size:", [len(d) for d in dev_set], sum([len(d) for d in dev_set]))

with gzip.open('data/dev(gen_rxn).pkl.gz', 'rb') as dev_file:
    while 1:
        try:
            reactants, products = cPickle.load(dev_file)
            products.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(reactants) < source_size and len(products) < target_size:
                    dev_gen_set[bucket_id].append([reactants, products])
                    break
        except EOFError:
            break

print("dev_gen_set size:", [len(d) for d in dev_gen_set],
							sum([len(d) for d in dev_gen_set]))

with gzip.open('data/train(all).pkl.gz', 'rb') as train_file: # (gen_rxn)
    while 1:
        try:
            reactants, products = cPickle.load(train_file)
            products.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(reactants) < source_size and len(products) < target_size:
                    train_set[bucket_id].append([reactants, products])
                    break
        except EOFError:
            break

train_set_size = sum([len(t) for t in train_set])
print("train_set size:", [len(t) for t in train_set], train_set_size)
save_step = int(train_set_size / FLAGS.batch_size)
print("saving every", save_step, "steps")

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


def train():
    with tf.Session() as sess:
        # Create model
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)
     
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                for i in xrange(len(train_bucket_sizes))]

        # Training loop
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        bar = progressbar.ProgressBar(max_value=FLAGS.steps_per_checkpoint)

        while True:
            
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
            if train_buckets_scale[i] > random_number_01])

            # Batch and step
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                      train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                               target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            bar.update(current_step % FLAGS.steps_per_checkpoint)

            # Checkpoint and evaluation
            if current_step % FLAGS.steps_per_checkpoint == 0:
            	bar.finish()
            	bar = progressbar.ProgressBar(max_value=FLAGS.steps_per_checkpoint)

                 # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                        "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("    eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                    target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("    eval: bucket %d perplexity %.4f" % (bucket_id, eval_ppx))

                    if len(dev_gen_set[bucket_id]) == 0:
                        print("gen_eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_gen_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                    target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("gen_eval: bucket %d perplexity %.4f" %(bucket_id, eval_ppx))

            if current_step % save_step == 0:
            	checkpoint_path = os.path.join(FLAGS.train_dir, "gen.ckpt")
            	model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            	for fname in glob.iglob(os.path.join(FLAGS.train_dir, "gen.ckpt*")):
   					shutil.copy2(fname, os.path.join(FLAGS.train_dir, "trained_models"))


def main(_):
	train()

if __name__ == "__main__":
	tf.app.run()
