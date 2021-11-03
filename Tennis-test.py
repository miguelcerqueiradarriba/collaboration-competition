#!/usr/bin/env python

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch

env = UnityEnvironment(file_name="Tennis_Windows_x86_64\Tennis.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# In[5]:


from ddpg_agent import Agent

agent = Agent(state_size=state_size*num_agents, action_size=action_size*num_agents, random_seed=10)

agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
agent.actor_local.eval()
agent.critic_local.eval()

for i in range(1, 6):  # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]
    agent.reset()
    state = env_info.vector_observations  # get the current state
    score = 0

    while True:
        state = np.array(np.array(state).flatten())
        action = agent.act(state)  # select an action

        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = np.array(env_info.vector_observations).flatten()  # get the next state
        reward = np.max(env_info.rewards)  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        agent.step(state, action, reward, next_state, done)  # take step with agent (including learning)
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

env.close()