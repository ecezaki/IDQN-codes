import argparse
from simple_dqn_tf2_improper import Agent
import numpy as np
import gym
from utils import plotLearning
import tensorflow as tf
from cartpole_swingup import CartPoleSwingUpEnv
import math
from controllers import LinearController, NonLinearControllerEnergyShaping, ApplyBrakesLinearVelocity
from scipy import linalg
import matplotlib.pyplot as plt




#############################################################################################################################
def get_optimal_cartpole_balance(env):
    # linearize the dynamics about the upright equilibrium
    # state matrix
    g = env.gravity
    mp = env.masspole
    mk = env.masscart
    lp = env.length
    mt = env.total_mass
    # env.x_threshold = 0.1 #testing the behavior of the policy when different stopping conditions are used.

    a = g / (lp * (4.0 / 3 - mp / (mp + mk)))
    A = np.array([[0, 1, 0, 0],
                [0, 0, a, 0],
                [0, 0, 0, 1],
                [0, 0, a, 0]])

    # input matrix
    b = -1 / (lp * (4.0 / 3 - mp / (mp + mk)))
    B = np.array([[0], [1 / mt], [0], [b]])

    # calculate optimal controller
    R = 5 * np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5 * np.eye(4, dtype=int)  # choose Q (weight for state)

    # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K_star = -np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    K_star = np.squeeze(K_star)
    return K_star

#############################################################################################################################
if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = CartPoleSwingUpEnv(rng_seed=112233)
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-lr', type=float, default=0.0003)
    # if you supply it, then true
    args = parser.parse_args()
    #########################################################################################################################
    
    K_star = get_optimal_cartpole_balance(env=env)
    
    seed_main = 112233
    rng_main = np.random.RandomState(seed_main)
    
    controllers = []
    
    # for i in range(7):
    #     controller = LinearController(rng_main.normal(size=4))
    #     controllers.append(controller)
    # # also add the optimal cartpole controller
    controllers.append(LinearController(K_star))
    #controllers.append(LinearController(np.array([-5., -5., 0, 5.])))
    controllers.append(NonLinearControllerEnergyShaping(env))
    controllers.append(ApplyBrakesLinearVelocity(env))

##############################################################################################################################


    #lr = 0.0003
    lr = args.lr
    n_games = 2000
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, controllers=controllers, 
                input_dims=env.observation_space.shape,
                mem_size=1000000, batch_size=64,
                epsilon_end=0.01)
    scores = []
    eps_history = []
    best_score = env.reward_range[0]
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action, controller_idx = agent.choose_action(observation)
            action_direction = 0
            if action > 0:
                action_direction = 1
            # limit magnitude of applied action force
            abs_force = abs(float(np.clip(action, -10, 10)))
            # change magnitude of the applied force in CartPole
            env.force_mag = abs_force
            observation_, reward, done, info = env.step(action_direction)
            score += reward
            agent.store_transition(observation, controller_idx, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            best_score = avg_score
            print('... saving model ...')
            agent.save_model()
        print('episode: ', i, 'score %.2f' % score,
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    #filename = 'cartpole_improper_3.png'
    #x = [i+1 for i in range(n_games)]
    #plotLearning(x, scores, eps_history, filename)
    fname = 'Const_lr_' + str(lr) + '.npy'
    np.save(fname, np.array(scores))
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        
    x_axis = np.arange(len(scores))
    plt.plot(x_axis, running_avg, 'r--', label=str(lr))
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    figname = 'Const_lr_' + str(lr) + '.png'
    plt.savefig(figname)
    plt.close()
