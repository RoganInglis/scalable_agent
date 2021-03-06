import os
import functools
import collections
import sonnet as snt
import tensorflow as tf

from agents.nets import convnet


nest = tf.contrib.framework.nest

# Structure to be sent from actors to learner.
#ActorOutput = collections.namedtuple(
#    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class Agent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(Agent, self).__init__(name='agent')

        self._num_actions = num_actions

        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)

    def _instruction(self, instruction):
        # Split string.
        splitted = tf.string_split(instruction)
        dense = tf.sparse_tensor_to_dense(splitted, default_value='')
        length = tf.reduce_sum(tf.to_int32(tf.not_equal(dense, '')), axis=1)

        # To int64 hash buckets. Small risk of having collisions. Alternatively, a
        # vocabulary can be used.
        num_hash_buckets = 1000
        buckets = tf.string_to_hash_bucket_fast(dense, num_hash_buckets)

        # Embed the instruction. Embedding size 20 seems to be enough.
        embedding_size = 20
        embedding = snt.Embed(num_hash_buckets, embedding_size)(buckets)

        # Pad to make sure there is at least one output.
        padding = tf.to_int32(tf.equal(tf.shape(embedding)[1], 0))
        embedding = tf.pad(embedding, [[0, 0], [0, padding], [0, 0]])

        core = tf.contrib.rnn.LSTMBlockCell(64, name='language_lstm')
        output, _ = tf.nn.dynamic_rnn(core, embedding, length, dtype=tf.float32)

        # Return last output.
        return tf.reverse_sequence(output, length, seq_axis=1)[:, 0]

    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, (frame, instruction) = env_output

        # Convert frame to floats and rescale
        frame = tf.to_float(frame)
        frame /= 255

        # Create convnet
        conv_out = convnet(frame)

        # Create language instuction net
        instruction_out = self._instruction(instruction)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        return tf.concat([conv_out, clipped_reward, one_hot_last_action, instruction_out], axis=1)

    def _head(self, core_output):
        policy_logits = snt.Linear(self._num_actions, name='policy_logits')(core_output)
        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

        # Sample an action from the policy.
        new_action = tf.multinomial(policy_logits, num_samples=1, output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='new_action')

        return AgentOutput(new_action, policy_logits, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0), (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d), initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)

        return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state