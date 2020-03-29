
import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers
import tensorflow.keras.optimizers as ko
import time


"""
input is stacking states [T-3, T-2, T-1, T]
update with loss calculated based on one step, meaning rewards and actions are not stacked up

for example, stack_size = 4
input observation is (4*128)
output is (4*6) 6 is the action_space.n

action selection and loss are only based on the last row of output

"""


np.random.seed(1)
tf.random.set_seed(1)

class MyModel(tf.keras.Model):

	def __init__(self, num_states, num_actions, hidden_units=128):
		super(MyModel, self).__init__(name = 'basic_ddqn')

        ## btach_size * size_state
		self.state_shape = num_state
		self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
		self.fc1 = tf.keras.layers.Dense(hidden_units, activation = 'relu',kernel_initializer='RandomNormal')  # kernel_initializer = 'he_uniform'
		self.fc2 = tf.keras.layers.Dense(hidden_units, activation = 'relu',kernel_initializer='RandomNormal')
		self.output_layer = tf.keras.layers.Dense(num_actions, name = 'q_values')
 
    
	@tf.function
	def call(self, inputs, training = None):
		x = self.input_layer(inputs)
		x = self.fc1(x)
		x = self.fc2(x)	
		output_ = self.output_layer(x)

		return output_

	# @tf.function
	def action_value(self, state):
		q_values = self.predict(state)
		q_values = q_values[-1]
		best_action = np.argmax(q_values, axis = -1)
		return best_action, q_values
		

def normalize_obs(obs, scale = 256):

	return obs/scale


def test_model():

    env = gym.make('Breakout-ram-v4')
    print('num_actions: ', env.action_space.n)

    model = MyModel(128, env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs)
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]



class DQNAgent:

	def __init__(self, model, target_model, env, buffer_size=50000, learning_rate=.0015, epsilon=0.9, epsilon_dacay=0.999,
                 min_epsilon=.1, gamma=.95, batch_size=64, target_update_iter=1000, learn_every_n_step=1, episode_num = 1000,
                 start_learning=1000, save_every_n_step = 5000, stack_interval = 4, stack_size = 4): 

		self.model = model
		self.target_model = target_model
		self.opt = tf.keras.optimizers.RMSprop(learning_rate = learning_rate, clipvalue = 1.0) #, clipvalue = 10.0
		self.model.compile(optimizer = self.opt, loss = 'huber_loss')

		self.env = env
		self.lr = learning_rate
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_dacay
		self.min_epsilon = min_epsilon
		self.gamma = gamma
		self.batch_size = batch_size
		self.target_update_iter = target_update_iter
		self.num_in_buffer = 0
		self.buffer_size = buffer_size
		self.start_learning = start_learning
		self.learn_every_n_step = learn_every_n_step
		self.save_every_n_step = save_every_n_step
		self.stack_interval = stack_interval
		self.stack_size = stack_size

		self.obs = np.empty((self.buffer_size,)+ (self.stack_size, self.env.reset().shape[0],))
		self.actions = np.empty((self.buffer_size), dtype = np.int8)
		self.rewards = np.empty((self.buffer_size), dtype = np.float32)
		self.dones = np.empty((self.buffer_size), dtype = np.bool)
		self.next_states = np.empty((self.buffer_size,)+ (self.stack_size, self.env.reset().shape[0],))
		self.next_idx = 0
		self.episode_num = episode_num
		self.loss_stat = []
		self.reward_his = []
		self.obs_stack = self.init_stacking()

	def train(self, model_path_dir):

		episode = 0
		step = 0
		loss = 0
		
		while episode < self.episode_num:

			obs = self.env.reset()
			# obs = normalize_obs(obs)

			done = False
			episode_reward =0.0
			pre_step = step

			## initialize the state stack
			self.obs_stack = self.init_stacking()
			self.stacking(obs)

			while not done:

				step += 1
				best_action, q_values = self.model.action_value(self.obs_stack)
				
				# best_action, q_values = self.model.action_value(obs[None])
				self.epsilon = max(self.epsilon, self.min_epsilon)
				action = self.get_action(best_action)
				next_obs, reward, done, info = self.env.step(action)
				# next_obs = normalize_obs(next_obs)

				temp_curr_obs = self.obs_stack
				self.stacking(next_obs)
				episode_reward += reward
				
				# self.store_transition(obs, action, reward, next_obs, done)
				self.store_transition(temp_curr_obs, action, reward, self.obs_stack, done)
				# obs = next_obs

				self.num_in_buffer = min(self.num_in_buffer+1, self.buffer_size)
		
				if step > self.start_learning:
					if not step % self.learn_every_n_step:
						losses = self.train_step()
						self.loss_stat.append(losses)
					if step % self.save_every_n_step == 0:
						print(' losses each {} steps: {}'.format(step, losses))
						self.save_model(model_path_dir)

					if step % self.target_update_iter == 0:
						self.update_target_model()
			
				if step > self.start_learning:
					self.e_decay()

			print("--episode: ", episode, '-- step: ', step-pre_step,  '--reward: ', episode_reward)
			episode += 1

			self.reward_his.append(episode_reward)


	def stacking(self, next_obs):

		self.obs_stack = np.roll(self.obs_stack, -1, axis = 0)
		self.obs_stack[-1, :] = next_obs

	def init_stacking(self):

		return np.zeros((self.stack_size, self.model.state_shape))

	def train_step(self):
		idxes = self.sample(self.batch_size)
		s_batch = self.obs[idxes]
		a_batch = self.actions[idxes]
		r_batch = self.rewards[idxes]
		ns_batch = self.next_states[idxes]
		done_batch = self.dones[idxes]

		mask = np.ones((self.batch_size, self.stack_size))
		mask[:, -1] = 1-done_batch

		r_batch_ = np.ones((self.stack_size,1))*r_batch
		r_batch_ = r_batch_.T

		target_q =  r_batch_ + self.gamma * np.amax(self.get_target_value(ns_batch), axis = -1)*mask
		target_f = self.model.predict(s_batch)

		for i, val in enumerate(a_batch):
			target_f[i][-1][val] = target_q[i][-1]

		losses = self.model.train_on_batch(s_batch, target_f)
		return losses


	def evaluation(self, env, render = False):
		obs, done, ep_reward = env.reset(), False, 0
		while not done:
			action, q_values = self.model.action_value(obs[None])
			obs, reward, done, info = env.step(action)
			ep_reward += reward
			if render:
				env.render()
			time.sleep(0.05)
		env.close()
		return ep_reward

	def store_transition(self, obs, action, reward, next_state, done):

		n_idx = self.next_idx % self.buffer_size
		self.obs[n_idx] = obs
		self.actions[n_idx] = action
		self.rewards[n_idx] = reward
		self.next_states[n_idx] =  next_state
		self.dones[n_idx] =  done 
		self.next_idx  = (self.next_idx+1)%self.buffer_size


	# sample n different indexes
	def sample(self, n):

		assert n<self.num_in_buffer
		return np.random.choice(self.num_in_buffer, self.batch_size, replace = False)
	
    # e-greedy
	def get_action(self, best_action):
		if np.random.rand() < self.epsilon:
			action = self.env.action_space.sample()
		else:
			action = best_action
		return action

    # assign the current network parameters to target network
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def get_target_value(self, obs):
		return self.target_model.predict(obs)

	def e_decay(self):
		self.epsilon = self.epsilon * self.epsilon_decay

	def save_model(self, model_path_dir):

		# tf.keras.models.save_model(self.model, model_path_dir)
		# tf.saved_model.save(self.model, model_path_dir)
		self.model.save_weights(model_path_dir)


if __name__ == '__main__':

	# test_model()
	
	env = gym.make("Pong-ram-v0")
	env = wrappers.Monitor(env, os.path.join(os.getcwd(), 'video_Pong'), force = True)
	num_actions = env.action_space.n
	num_state = env.reset().shape[0]

	model = MyModel(num_state, num_actions)
	target_model = MyModel(num_state, num_actions)

	# model.load_weights('new_dqn/Walker/dqn_checkpoint')
	# target_model.load_weights('new_dqn/Walker/dqn_checkpoint')

	agent = DQNAgent(model, target_model,  env)
  
	agent.train("new_dqn/Pong/dqn_checkpoint")
	print("train is over and model is saved")
	np.save('dqn_agent_train_lost_pong.npy', agent.reward_his)
   





















