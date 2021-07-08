import gym
import numpy as np
from collections import deque
import warnings

# imports the cartpole environment
env=gym.make('CartPole-v1')

def relu(mat):
    return np.multiply(mat,(mat>0))

def relu_derivative(mat):
    return (mat>0)*1


class NNLayer:
    # class representing a neural net layer
    def __init__(self, input_size, output_size, activation=None, lr = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
        self.activation_function = activation
        self.lr = lr

    def forward(self, inputs, remember_for_backprop=True):
        # input has shape batch_size x layer_input_size 
        input_with_bias = np.append(inputs, 1)
        unactivated = np.dot(input_with_bias, self.weights)
        # store variables for backward pass
        output = unactivated
        if self.activation_function != None:
            # assuming here the activation function is relu, this can be made more robust
            output = self.activation_function(output)
        if remember_for_backprop:
            # store variables for backward pass     
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
            
        return output
    
    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        # this is pointwise
        if self.activation_function != None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out),gradient_from_above)
            
        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))), np.reshape(adjusted_mul, (1,len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i)
        return delta_i
    
    def update_weights(self, gradient):
        self.weights = self.weights - self.lr*gradient

class RLAgent:

    # class representing a reinforcement learning agent
    env = None
    def __init__(self, env):
        self.env = env
        self.hidden_size = 24
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.num_hidden_layers = 2
        self.epsilon = 1.0
        self.gamma = 0.95
        self.memory = deque([],1000000)
        self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu)]
        for i in range(self.num_hidden_layers-1):
            self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=relu))
        self.layers.append(NNLayer(self.hidden_size+1, self.output_size))
                
    def select_action(self, observation):
        # the action values accociated with that state
        values = self.forward(np.asmatrix(observation))
        # epsilon greedy
        if (np.random.random() > self.epsilon):
            # expliotation 
            return np.argmax(values)
        else:
            # exploration
            return np.random.randint(self.env.action_space.n)
        
    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals

    def remember(self, done, action, observation, prev_obs):
        self.memory.append([done, action, observation, prev_obs])
        
    def experience_replay(self, update_size=20):
        # checks to see if we have enough experiences to start learning 
        # if not we wait until we do
        if (len(self.memory) < update_size):
            return
        else: 
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, prev_obs = self.memory[index]
                action_values = self.forward(prev_obs, remember_for_backprop=True)
                next_action_values = self.forward(new_obs, remember_for_backprop=False)
                experimental_values = np.copy(action_values)
                if done:
                    experimental_values[action_selected] = -1
                else:
                    experimental_values[action_selected] = 1 + self.gamma*np.max(next_action_values)
                self.backward(action_values, experimental_values)
        self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon*0.995
        for layer in self.layers:
            layer.lr = layer.lr if layer.lr < 0.0001 else layer.lr*0.995
        
    def backward(self, calculated_values, experimental_values): 
        # values are batched = batch_size x output_size
        delta = (calculated_values - experimental_values)
        # print('delta = {}'.format(delta))
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
                

# Global variables
NUM_EPISODES = 5000
MAX_TIMESTEPS = 500
# creates an instance of the class RLAgent called model
model = RLAgent(env)
# The main program loop with NUM_EPISODES iterations
for i_episode in range(NUM_EPISODES):
    # each episode the environment is reset
    observation = env.reset()
    # Iterating through time steps within an episode
    for t in range(MAX_TIMESTEPS):
        env.render()
        # provides model with observation to select an action
        action = model.select_action(observation)
        #stores the previous observation in prev_observation
        prev_obs = observation
        # takes the action selected by the Agent and applies it to the environment
        observation, reward, done, info = env.step(action)
        # Keep a store of the agent's experiences
        model.remember(done, action, observation, prev_obs)
        model.experience_replay(20)
        # epsilon decay
        model.epsilon = model.epsilon if model.epsilon < 0.01 else model.epsilon*0.995
        if done:
            # If the pole has tipped over, end this episode
            print('Episode {} ended after {} timesteps, current exploration is {}'.format(i_episode+1, t+1,model.epsilon))
            print(model.layers[0].lr)
            break