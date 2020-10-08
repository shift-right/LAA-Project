"""
Using:
Tensorflow: 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=1.0,
            replace_target_iter=200,
            memory_size=2000,
            batch_size=512,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon = 1.0
        self.epsilon_increment =0.000005
        self.memory_counter = 18000000
        
        
        self.double_q = double_q    # decide to use double q or not
        
        
        
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.saver = tf.train.Saver()
        
        #load model
        self.saver = tf.train.import_meta_graph('./DoubleDQN/ckpt/model.ckpt-18000000.meta')
        self.saver.restore(self.sess,tf.train.latest_checkpoint('./DoubleDQN/ckpt'))
        #self.sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("eval_net/l1/w1:0")
        b1 = graph.get_tensor_by_name("eval_net/l1/b1:0")
        l1 = graph.get_tensor_by_name("eval_net/l1/Relu:0")
        w2 = graph.get_tensor_by_name("eval_net/l2/w2:0")
        b2 = graph.get_tensor_by_name("eval_net/l2/b2:0")
        l2 = graph.get_tensor_by_name("eval_net/l2/Relu:0")
        w3 = graph.get_tensor_by_name("eval_net/l3/w3:0")
        b3 = graph.get_tensor_by_name("eval_net/l3/b3:0")
        l3 = graph.get_tensor_by_name("eval_net/l3/Relu:0")
        w4 = graph.get_tensor_by_name("eval_net/l4/w4:0")
        b4 = graph.get_tensor_by_name("eval_net/l4/b4:0")
        l4 = graph.get_tensor_by_name("eval_net/l4/Relu:0")
        w5 = graph.get_tensor_by_name("eval_net/l5/w5:0")
        b5 = graph.get_tensor_by_name("eval_net/l5/b5:0")
        l5 = graph.get_tensor_by_name("eval_net/l5/Relu:0")
        w6 = graph.get_tensor_by_name("eval_net/l6/w6:0")
        b6 = graph.get_tensor_by_name("eval_net/l6/b6:0")
        self.q_eval = graph.get_tensor_by_name("eval_net/l6/add:0")
        
        w1 = graph.get_tensor_by_name("target_net/l1/w1:0")
        b1 = graph.get_tensor_by_name("target_net/l1/b1:0")
        l1 = graph.get_tensor_by_name("target_net/l1/Relu:0")
        w2 = graph.get_tensor_by_name("target_net/l2/w2:0")
        b2 = graph.get_tensor_by_name("target_net/l2/b2:0")
        l2 = graph.get_tensor_by_name("target_net/l2/Relu:0")
        w3 = graph.get_tensor_by_name("target_net/l3/w3:0")
        b3 = graph.get_tensor_by_name("target_net/l3/b3:0")
        l3 = graph.get_tensor_by_name("target_net/l3/Relu:0")
        w4 = graph.get_tensor_by_name("target_net/l4/w4:0")
        b4 = graph.get_tensor_by_name("target_net/l4/b4:0")
        l4 = graph.get_tensor_by_name("target_net/l4/Relu:0")
        w5 = graph.get_tensor_by_name("target_net/l5/w5:0")
        b5 = graph.get_tensor_by_name("target_net/l5/b5:0")
        l5 = graph.get_tensor_by_name("target_net/l5/Relu:0")
        w6 = graph.get_tensor_by_name("target_net/l6/w6:0")
        b6 = graph.get_tensor_by_name("target_net/l6/b6:0")
        self.q_next = graph.get_tensor_by_name("target_net/l6/add:0")

        
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, n_l2, n_l3,n_l4, n_l5, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l3, n_l4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, n_l4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l4, n_l5], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, n_l5], initializer=b_initializer, collections=c_names)
                l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)

            with tf.variable_scope('l6'):
                w6 = tf.get_variable('w6', [n_l5, self.n_actions], initializer=w_initializer, collections=c_names)
                b6 = tf.get_variable('b6', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l5, w6) + b6
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names,n_l1, n_l2, n_l3,n_l4, n_l5, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 1024, 512, 512, 256, 64, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, n_l2, n_l3,n_l4, n_l5, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names,n_l1, n_l2, n_l3,n_l4, n_l5, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
            
           
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        if (self.memory_counter >=20000)and (self.memory_counter %20000==0):
            self.saver.save(self.sess,"./DoubleDQN/ckpt/model.ckpt",global_step = self.memory_counter)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        # action = np.argmax(actions_value)

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        q_values = actions_value - np.max(actions_value)
        p_a_s = np.exp(1*q_values)/np.sum(np.exp(1*q_values))
        p_a_s =p_a_s.flatten()
        action = np.random.choice(a=15,p=p_a_s)
        print("action:",action)
        
        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        #self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        
        
        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        print("loss",self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


