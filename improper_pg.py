# class defs for agent, environment, experiment

import numpy as np
import tensorflow as tf
from attention_models import AttentionModel
import math

# improper learning loss of an attention model wrt a trajectory and a menu of controllers
def loss(attn_model, trajectory, controllers):
    m = len(controllers)
    H = len(trajectory)

    # make tail losses
    rewards = [transition[3] for transition in trajectory]
    neg_tail_rewards = [-sum(rewards[i:H]) for i in range(H)]
    tail_losses = tf.convert_to_tensor(neg_tail_rewards, dtype=tf.float32)    

    # # get matrix K_j(a_t|s_t): each controller j's prob of playing action a_t @ state s_t over all times t
    K = np.zeros((m, H))
    for j in range(m):
        for t in range(H):
            K[j,t] = controllers[j].get_prob(trajectory[t][0], trajectory[t][1])
    K_transp = tf.convert_to_tensor(K.T, dtype=tf.float32)
    # get attention model probabilities per controller per state in the trajectory
    # print("loss calling compute ...")
    W = tf.convert_to_tensor([attn_model.compute(transition[0]) for transition in trajectory],
                                dtype=tf.float32)
    W = tf.squeeze(W)

    # W and K_transp should have shape (H,m)
    mixture_probs = tf.multiply(W, K_transp)
    mixture_probs = tf.reduce_sum(mixture_probs, axis=1) # should result in shape (H,)
    log_mixture_probs = tf.math.log(mixture_probs)
    # print("************ loss: log_mixture probs: {}, tail_losses: {}".format(log_mixture_probs, tail_losses))
    pg_loss = tf.tensordot(log_mixture_probs, tail_losses, 1) # note: targets are losses, not rewards!!
    # print("loss: attn_model weights: {}; W: {}, K.T: {}, tail_losses: {}, pg_loss: {}".format(
    #     attn_model.trainable_variables, W, K_transp, tail_losses, pg_loss))

    return pg_loss  

class Agent:
    """
    Improper PG learning agent
    """
    def __init__(self, state_dim=1, n_hidden_1=10, n_hidden_2=10, 
                    controllers=[], learn_rate=0.01, seed=43, attention_model=None,
                    optimizer='sgd'):
        self.state_dim = state_dim # state space dimension
        self.num_controllers = len(controllers) # action space dimension
        self.controllers = controllers # controller library, each controller must implement get_prob(), get_action()
        self.t = 0 # total environment steps elapsed
        self.k = 0 # no. of episodes since last gradient update
        self.rng = np.random.RandomState(seed)
        self.tf_rng = tf.random.experimental.Generator.from_seed(seed)
        self.trajectory = []
        self.trajectoryfull = []
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.lr = learn_rate
        if optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        elif optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        else:
            print("!!!!!!!!!!!!!Error!!!!!!!!!!!!!!!! Unrecognized optimizer") 

        if attention_model is not None:
            self.attn_model = attention_model
        else:
            self.attn_model = AttentionModel(output_dim=self.num_controllers, 
                            name='attn_model', seed=43, 
                            n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2)
 
    def get_action(self, state):
        pi = self.attn_model.compute(state).numpy().flatten()
        # first sample a controller
        controller_idx = self.rng.choice(self.num_controllers, p=pi)
        if controller_idx == 0:
            print('LQR')
        elif controller_idx == 1:
            print('NonlinearSwingup')
        else: 
            print('Apply Brakes Linear')    
        # else:
        #     print('ApplyBrakes Angular')
        # query the sampled controller for its action suggestion 
        return self.controllers[controller_idx].get_action(state)

    def learn(self, trajectory):
        # print("learn: got trajectory: {}".format(trajectory))
        with tf.GradientTape() as tape:
            current_loss = loss(self.attn_model, 
                                trajectory=trajectory, controllers=self.controllers)
        grad = tape.gradient(current_loss, self.attn_model.trainable_variables)
        # print("learn: trainable vars before update: {}; grad: {}".format(
        #     self.attn_model.trainable_variables, grad))

        # gradient descent step
        self.optimizer.apply_gradients(zip(grad, self.attn_model.trainable_variables))
        
        # print("learn: trainable vars after update: {}".format(
        #     self.attn_model.trainable_variables))
        return current_loss, grad

    def observe(self, cur_state, action, next_state, reward, end_of_episode=False):
        self.t += 1
        transition = (cur_state, action, next_state, reward)
        self.trajectory.append(transition)
        self.trajectoryfull.append(transition)

        if end_of_episode:
            self.k += 1 
            # update policy with one episode worth of data
            cur_loss, gradient = self.learn(self.trajectory)
            # print("current loss: {}, current gradient: {}".format(cur_loss, gradient))
            print("current loss: {}".format(cur_loss))
            # print(self.trajectory)
            self.trajectory = []

    def get_info(self):
        info = {
            "model": self.attn_model.trainable_variables
        }
        return info

        
class ManualAgent:
    """
    Improper PG with manual switching
    """
    def __init__(self, controllers=[], threshold=0.5):
        self.controllers = controllers
        self.threshold = threshold 
        self.t = 0 # total environment steps elapsed
        self.trajectory = []
        self.flag = 0
        # self.k = 0 # no. of episodes since last gradient update
        
    def get_action(self, state):
        
        
        if np.abs(state[2])<(2/9)*(2*math.pi):
            controller_idx = 0
        else:
            controller_idx = 1
        
        
        
        # if np.abs(state[1])>2:
        #     self.flag = 2
        # elif (state[2]%2*math.pi)<(2/9)*math.pi:
        #     #controller_idx = 0
        #     #print('LQR')
        #     self.flag = 1
        # else:
        #     self.flag = 0
        # # else:
        
        # if self.flag == 1:
        #     controller_idx = 0
        #     print('LQR')
        # elif self.flag == 0:
        #     controller_idx = 1
        #     print('Nonlinear')

        # else:
        #     controller_idx = 2
        #     print('ApplyBrakes')
            
        #pi = self.attn_model.compute(state).numpy().flatten()
        # first sample a controller
        #controller_idx = self.rng.choice(self.num_controllers, p=pi)
        # query the sampled controller for its action suggestion 
        return self.controllers[controller_idx].get_action(state)

    # def learn(self, trajectory):
    #     # print("learn: got trajectory: {}".format(trajectory))
    #     with tf.GradientTape() as tape:
    #         current_loss = loss(self.attn_model, 
    #                             trajectory=trajectory, controllers=self.controllers)
    #     grad = tape.gradient(current_loss, self.attn_model.trainable_variables)
    #     # print("learn: trainable vars before update: {}; grad: {}".format(
    #     #     self.attn_model.trainable_variables, grad))

    #     # gradient descent step
    #     self.optimizer.apply_gradients(zip(grad, self.attn_model.trainable_variables))
        
    #     # print("learn: trainable vars after update: {}".format(
    #     #     self.attn_model.trainable_variables))
    #     return current_loss, grad

    def observe(self, cur_state, action, next_state, reward, end_of_episode=False):
        self.t += 1
        transition = (cur_state, action, next_state, reward)
        self.trajectory.append(transition)

        # if end_of_episode:
        #     self.k += 1 
        #     # update policy with one episode worth of data
        #     cur_loss, gradient = self.learn(self.trajectory)
        #     # print("current loss: {}, current gradient: {}".format(cur_loss, gradient))
        #     print("current loss: {}".format(cur_loss))
        #     self.trajectory = []

    def get_info(self):
        info = {
            "model": self.attn_model.trainable_variables
        }
        return info
