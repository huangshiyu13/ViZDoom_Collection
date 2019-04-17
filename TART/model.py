"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: model.py
"""
import pickle
import tensorflow as tf
import numpy as np
from .utils import check_dir, del_dir, create_dir
import os
# from tensorflow.python.ops.rnn import dynamic_rnn


def get_step(model_path):
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        if 'step' in model_dict:
            return model_dict['step']
    return 1

def transfer_params(from_scope, to_sope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_sope)

    trans_op = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        trans_op.append(to_var.assign(from_var))
    return tf.group(*trans_op)

def average_params(from_scopes, to_sope):
    for i, from_scope in enumerate(from_scopes):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)

        if i == 0:
            from_varses = [[] for _ in range(len(vars))]

        for j in range(len(from_varses)):
            from_varses[j].append(vars[j])

    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_sope)

    trans_op = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_vars, to_var in zip(from_varses, to_vars):
        trans_op.append(to_var.assign(tf.add_n(from_vars) / len(from_scopes)))

    return tf.group(*trans_op)


class Model():

    def __init__(self, name=None):
        self.name = name

    def get_assign_ops(self):
        self.assign_ops = {}
        variables = tf.trainable_variables(scope=self.name)
        for var in variables:
            var_holder = tf.placeholder(tf.float32, shape=var.get_shape())
            self.assign_ops[var.name] = [var.assign(var_holder), var_holder]

        return self.assign_ops

    # Layers
    ## Dense Layer
    def dense(self, bottom, name, in_size, out_size, use_relu=False, get_weight = False):
        with tf.variable_scope(name):
            weights, biases = self.get_dense_var_new(in_size, out_size)
            x = tf.reshape(bottom, [-1, in_size])
            output = tf.nn.bias_add(tf.matmul(x, weights), biases)
            if use_relu:
                output = tf.nn.relu(output)

            if get_weight:
                weight_dacay = tf.nn.l2_loss(weights, name='weight_dacay')
                return output, weight_dacay
            else:
                return output

    ## Conv Layer
    def conv2d(self, bottom, name, kernel_size=[3, 3], out_channel=512, stddev=0.01, use_relu=False, strides = [1,1,1,1], get_weight = False):

        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()[-1]

            filt = tf.get_variable(
                initializer=tf.random_normal([kernel_size[0], kernel_size[1], shape, out_channel], mean=0.0,
                                             stddev=stddev),
                name='filter')
            conv_biases = tf.get_variable(initializer=tf.zeros([out_channel]), name='biases')

            conv = tf.nn.conv2d(bottom, filt, strides, padding='SAME')
            output = tf.nn.bias_add(conv, conv_biases)

            if use_relu:
                output = tf.nn.relu(output)

            if get_weight:
                weight_dacay = tf.nn.l2_loss(filt, name='weight_dacay')
                return output, weight_dacay
            else:
                return output

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_dense_var_new(self, in_size, out_size):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, "weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, "biases")
        return weights, biases

    def get_var(self, initial_value, var_name):
        value = initial_value
        var = tf.get_variable(initializer=value, name=var_name)
        assert var.get_shape() == initial_value.get_shape()
        return var

    def GaussianNoise(self, x, sigma=0.01):
        x = x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma)
        return x

    # def lstm_layer(self, bottom_squence, name, hidden_size, seq_len, init_state=None, pre_forward_num=0):
    #
    #     lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
    #
    #
    #
    #     with tf.variable_scope(name):
    #         init_state = None
    #         rnn = tf.keras.layers.RNN(lstm_cell, return_state=True, stateful=True)
    #         rnn.reset_states(states=init_state)
    #
    #
    #         outputs_all = rnn(bottom_squence)
    #         outputs = outputs_all[0]
    #         hidden_states = outputs_all[1:]
    #
    #     return outputs, hidden_states

    def lstm_layer(self, bottom_squence, name, hidden_size, seq_len, init_state=None, pre_forward_num=0,
                   get_all_outputs=False):

        # with tf.variable_scope(name):
        # lstm = LSTMCell(hidden_size)
        lstm = tf.keras.layers.LSTMCell(hidden_size, unit_forget_bias=True, name=name)

        # self.lstm = lstm

        if init_state is not None:
            state = init_state
        else:
            state = lstm.zero_state(tf.shape(bottom_squence)[0], dtype=tf.float32)

        outputs = []
        hidden_states = []

        for t in range(seq_len):
            input_tensor = bottom_squence[:, t, :]
            all_outputs = lstm(input_tensor, states=state)  # outputs is different in different build mode
            # print(input_tensor.get_shape(), all_outputs)
            # print(input_tensor.get_shape(), [y0.get_shape().as_list() for y0 in all_outputs])
            # print(len(all_outputs))
            output = all_outputs[0]
            state = all_outputs[1]

            if t < pre_forward_num:
                output = tf.stop_gradient(output)
                state = (tf.stop_gradient(state[0]), tf.stop_gradient(state[1]))
            outputs.append(output)
            hidden_states.append(state)

        # weight_decay = tf.nn.l2_loss(lstm._kernel)
        if get_all_outputs:
            return outputs, hidden_states
        else:
            return outputs[-1], hidden_states[-1]

    def gru_layer(self, bottom_squence, name, hidden_size, seq_len, init_state=None, pre_forward_num=0,
                  get_all_outputs=False):

        gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size, name=name)

        if init_state is not None:
            state = init_state
        else:
            state = gru_cell.zero_state(tf.shape(bottom_squence)[0], dtype=tf.float32)

        outputs = []
        hidden_states = []

        for t in range(seq_len):
            input_tensor = bottom_squence[:, t, :]
            # print('t=',t,type(state))
            # print('state:',state.get_shape())
            all_outputs = gru_cell(input_tensor, state=state)  # outputs is different in different build mode
            # print('alloutputs:',len(all_outputs),all_outputs)

            output = all_outputs[0]
            state = all_outputs[1]

            if t < pre_forward_num:
                output = tf.stop_gradient(output)
                state = tf.stop_gradient(state)
            outputs.append(output)
            hidden_states.append(state)

        if get_all_outputs:
            return outputs, hidden_states
        else:
            return outputs[-1], hidden_states[-1]

    def embed(self, bottom, name, in_size, out_size, stddev=0.01,get_weight= False):
        with tf.variable_scope(name):
            v_embeddings = tf.Variable(tf.random_normal([in_size, out_size], mean=0.0, stddev=stddev), name='embedding')
            embedded_v = tf.nn.embedding_lookup(v_embeddings, bottom)
            weight_dacay = tf.nn.l2_loss(v_embeddings, name='weight_dacay')
            if get_weight:
                return embedded_v, weight_dacay
            else:
                return embedded_v

    def DropOut(self, bottom, keep_prob=0.5):
        # return tf.nn.dropout(bottom, keep_prob)
        return tf.nn.dropout(bottom, rate=1 - keep_prob)


def get_parameters(model_path):
    if type(model_path) == list:
        parameters = {}
        for model_path0 in model_path:
            with open(model_path0, 'rb') as f:
                model_dict0 = pickle.load(f)
            parameters.update(model_dict0)
    else:
        with open(model_path, 'rb') as f:
            parameters = pickle.load(f)
    return parameters

class Saver(object):
    def __init__(self, sess):
        self.sess = sess

    def get_variables(self, scope_name=None):
        vars = []
        with self.sess.as_default(), self.sess.graph.as_default():
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name):
                vars.append(var)
        return vars

    def set_parameters(self, parameters, model, strict=False, show_names=False, del_scope=False):
        if hasattr(model, 'assign_ops'):
            assign_ops = model.assign_ops
        else:
            assign_ops = model.get_assign_ops()

        if hasattr(model, 'name'):
            scope_name = model.name
        else:
            scope_name = None

        if show_names:
            print('var names:')
            for var_name in assign_ops:
                print(var_name)
            print('loaded model names:')
            for key in parameters:
                print(key)

        ass_ops = []
        feeds = {}
        for var_name in assign_ops:
            var_name0 = var_name
            if del_scope:
                assert scope_name is not None
                if var_name.startswith(scope_name):
                    var_name = var_name[len(scope_name) + 1:]
            if var_name in parameters:
                if np.shape(parameters[var_name]) == assign_ops[var_name0][1].get_shape():
                    ass_op = assign_ops[var_name0]
                    feeds.update({assign_ops[var_name0][1]: parameters[var_name]})
                    ass_ops.append(ass_op)
                else:
                    assert not strict, "{} shape is different:loaded model:{} var in graph:{}".format(var_name,
                                                                                                      np.shape(
                                                                                                          parameters[
                                                                                                              var_name]),
                                                                                                      assign_ops[
                                                                                                          var_name0][
                                                                                                          1].get_shape().get_shape())
            else:
                assert not strict, "loaded model has no value of {}".format(var_name)

        self.sess.run(ass_ops, feed_dict=feeds)

    def load(self, model_path, model, strict=False, show_names=False, del_scope=False):
        parameters = get_parameters(model_path)
        self.set_parameters(parameters=parameters, model=model, strict=strict,
                            show_names=show_names, del_scope=del_scope)

    # def load(self, model_path, assign_ops, scope_name=None, strict=False, show_names=False, del_scope=False):
    #     parameters = get_parameters(model_path)
    #     self.set_parameters(parameters=parameters, assign_ops=assign_ops, scope_name=scope_name, strict=strict,
    #                         show_names=show_names, del_scope=del_scope)

    def load_v0(self, model_path, scope_name=None, strict=False, show_names=False, del_scope=False):
        parameters = get_parameters(model_path)

        vars = self.get_variables(scope_name)

        if show_names:
            print('var names:')
            for var in vars:
                print(var.name)
            print('loaded model names:')
            for key in parameters:
                print(key)

        ass_ops = []
        for var in vars:
            var_name = var.name
            if del_scope and scope_name is not None:
                if var_name.startswith(scope_name):
                    var_name = var_name[len(scope_name) + 1:]
            if var_name in parameters:

                if np.shape(parameters[var_name]) == var.get_shape():
                    ass_op = var.assign(parameters[var_name])
                    ass_ops.append(ass_op)
                    if show_names:
                        print('assign {} to {}'.format(var_name, var.name))
                else:
                    assert not strict, "{} shape is different:loaded model:{} var in graph:{}".format(var_name,
                                                                                                      np.shape(
                                                                                                          parameters[
                                                                                                              var_name]),
                                                                                                      var.get_shape())

            else:
                assert not strict, "loaded model has no value of {}".format(var.name)

        self.sess.run(ass_ops)

    def save(self, save_path,save_scope_name=False, scope_name=None):
        params = {}
        with self.sess.as_default(), self.sess.graph.as_default():

            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name):
                # param_name = var.name.split(':')[0]
                param_name = var.name
                if not save_scope_name and scope_name is not None:
                    assert param_name.split('/')[0] == scope_name
                params[param_name] = self.sess.run(var)
        try:
            np.save(save_path, params)
        except:
            print('can\'t save model to path:{}'.format(save_path))

    def _auto_save(self, save_path, step=None):
        params = {}
        for key in self.save_params:
            params[key] = self.sess.run(self.save_params[key])
        # np.save(save_path, params)
        if step is not None:
            params['step'] = step
        with open(save_path, 'wb') as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    def auto_save_init(self, save_dir, save_interval, max_keep=5, save_scope_name=False, scope_name=None,
                       continue_train=False):
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.max_keep = max_keep

        self.model_step_dir = save_dir + 'steps/'


        if continue_train == 'True' or continue_train == 'true' or continue_train:
            assert check_dir(self.model_step_dir)

        else:
            if check_dir(self.model_step_dir):
                del_dir(self.model_step_dir)
            create_dir(self.model_step_dir)


        self.save_params = {}

        with self.sess.as_default(), self.sess.graph.as_default():
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name):
                # param_name = var.name.split(':')[0]
                param_name = var.name
                if not save_scope_name:
                    assert scope_name is not None
                    assert param_name.split('/')[0] == scope_name
                    param_name = '/'.join(param_name.split('/')[1:])
                self.save_params[param_name] = var

    def auto_save(self, step):
        if step > 0 and step % self.save_interval == 0:

            self._auto_save('{}/{}.pkl'.format(self.model_step_dir, step), step)

            os.system('mv {} {}'.format(self.save_dir + 'init.pkl', self.save_dir + 'init_bak.pkl'))

            os.system('cp {}/{}.pkl {}'.format(self.model_step_dir, step, self.save_dir + 'init.pkl'))

            if step > self.max_keep * self.save_interval:
                rm_step = step - self.max_keep * self.save_interval
                os.system('rm {}/{}.pkl'.format(self.model_step_dir, rm_step))



    def get_parameters(self,save_scope_name=False, scope_name=None):
        params = {}
        with self.sess.as_default(), self.sess.graph.as_default():

            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name):
                # param_name = var.name.split(':')[0]

                param_name = var.name
                if not save_scope_name and scope_name is not None:
                    assert param_name.split('/')[0] == scope_name
                    param_name = '/'.join(param_name.split('/')[1:])
                params[param_name] = self.sess.run(var)

        return params

    def set_parameters_v0(self, parameters, scope_name=None, strict=False, show_names=False, del_scope=False):
        vars = self.get_variables(scope_name)

        ass_ops = []

        if show_names:
            print('var names:')
            for var in vars:
                print(var.name)
            print('loaded model names:')
            for key in parameters:
                print(key)

        for var in vars:
            var_name = var.name
            if del_scope:
                assert scope_name is not None
                if var_name.startswith(scope_name):
                    var_name = var_name[len(scope_name) + 1:]
            if var_name in parameters:
                ass_op = var.assign(parameters[var_name])
                ass_ops.append(ass_op)

            else:
                assert not strict, "loaded model has no value of {}".format(var.name)

        self.sess.run(ass_ops)


if __name__ == '__main__':
    hidden_size = 3

    lstm_cell = tf.keras.layers.LSTMCell(hidden_size)

    layer = tf.keras.layers.RNN(lstm_cell)

    x = tf.keras.layers.Input((None, 5))

    y = lstm_cell(x)

    print(x.get_shape(), y.get_shape())
