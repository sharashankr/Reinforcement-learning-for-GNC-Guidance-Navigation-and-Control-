#!/usr/bin/env python

"""LunarLander environment class for RL-Glue-py.
"""

from environment import BaseEnvironment
import numpy as np
import gym

class LunarLanderEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        self.env = gym.make("LunarLander-v2")

        # Use the new way of setting the seed
        seed_value = env_info.get("seed", 0)
        self.env.action_space.seed(seed_value)
        self.env.reset(seed=seed_value)

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """        
        reward = 0.0
        observation = self.env.reset()  # Get the initial observation
        is_terminal = False
        
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return the first state observation from the environment
        return self.reward_obs_term[1]  # Returning only the observation
        
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        current_state, reward, is_terminal, _ = self.env.step(action)
        self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward_obs_term  # Return the updated reward, state, and terminal flag
