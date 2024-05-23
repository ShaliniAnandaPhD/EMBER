# dynamic_route_optimization_drl.py

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

class CustomRouteEnv(gym.Env):
    """
    Custom environment for route optimization using deep reinforcement learning.

    The environment represents a grid-based map where the agent needs to find the optimal evacuation route.

    Observation:
        Type: Box(4)
        Num    Observation               Min            Max
        0      Agent's x-coordinate      0              grid_size - 1
        1      Agent's y-coordinate      0              grid_size - 1
        2      Destination x-coordinate  0              grid_size - 1
        3      Destination y-coordinate  0              grid_size - 1

    Actions:
        Type: Discrete(4)
        Num    Action
        0      Move up
        1      Move right
        2      Move down
        3      Move left

    Reward:
        Reward of -1 for each step taken, and 0 when reaching the destination.

    Starting State:
        Agent starts at a random position on the grid.

    Episode Termination:
        Agent reaches the destination or exceeds the maximum number of steps.
    """

    def __init__(self, grid_size=10, max_steps=100):
        super(CustomRouteEnv, self).__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Define the observation and action spaces
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)

        # Initialize the agent's and destination's positions
        self.agent_pos = None
        self.destination_pos = None
        self.current_step = 0

    def reset(self):
        # Reset the environment and return the initial observation
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)
        self.destination_pos = np.random.randint(0, self.grid_size, size=2)
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        # Execute one step in the environment based on the given action
        self._take_action(action)
        self.current_step += 1

        # Check if the agent has reached the destination
        done = np.array_equal(self.agent_pos, self.destination_pos) or self.current_step >= self.max_steps

        # Calculate the reward based on the agent's position and destination
        reward = 0 if np.array_equal(self.agent_pos, self.destination_pos) else -1

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Get the current observation
        return np.concatenate((self.agent_pos, self.destination_pos))

    def _take_action(self, action):
        # Update the agent's position based on the given action
        if action == 0:  # Move up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Move right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Move down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 3:  # Move left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

    def render(self, mode='human'):
        # Render the environment (optional)
        pass

    def close(self):
        # Close the environment and perform any necessary cleanup (optional)
        pass

def train_drl_route_optimizer(env, algo='dqn', total_timesteps=10000, save_path='route_optimizer_model'):
    """
    Train a DRL model for route optimization.

    Args:
        env (gym.Env): The route optimization environment.
        algo (str): The DRL algorithm to use ('dqn' or 'ppo').
        total_timesteps (int): The total number of timesteps to train the model.
        save_path (str): The path to save the trained model.

    Returns:
        The trained DRL model.
    """
    try:
        # Check if the environment is valid
        check_env(env)

        # Create the DRL model
        if algo == 'dqn':
            model = DQN('MlpPolicy', env, verbose=1)
        elif algo == 'ppo':
            model = PPO('MlpPolicy', env, verbose=1)
        else:
            raise ValueError(f"Invalid algorithm: {algo}. Supported algorithms: 'dqn', 'ppo'")

        # Train the model
        model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        model.save(save_path)

        return model

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def optimize_route(model, env, state):
    """
    Optimize the evacuation route using the trained DRL model.

    Args:
        model: The trained DRL model.
        env (gym.Env): The route optimization environment.
        state (numpy.ndarray): The current state of the environment.

    Returns:
        The optimal action to take based on the current state.
    """
    try:
        # Get the optimal action from the DRL model
        action, _ = model.predict(state, deterministic=True)
        return action

    except Exception as e:
        print(f"Error during route optimization: {str(e)}")
        return None

def evaluate_model(model, env, n_eval_episodes=10):
    """
    Evaluate the performance of the trained DRL model.

    Args:
        model: The trained DRL model.
        env (gym.Env): The route optimization environment.
        n_eval_episodes (int): The number of episodes to evaluate the model.

    Returns:
        The mean reward achieved by the model over the evaluation episodes.
    """
    try:
        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
        return mean_reward

    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        return None

def main():
    # Create the route optimization environment
    env = CustomRouteEnv(grid_size=10, max_steps=100)

    # Train the DRL model
    model = train_drl_route_optimizer(env, algo='dqn', total_timesteps=10000, save_path='route_optimizer_model')

    if model is None:
        print("Failed to train the model.")
        return

    # Evaluate the trained model
    mean_reward = evaluate_model(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")

    # Example usage of the trained model for route optimization
    state = env.reset()
    done = False
    while not done:
        action = optimize_route(model, env, state)
        state, reward, done, _ = env.step(action)
        print(f"Current state: {state}, Action: {action}, Reward: {reward}, Done: {done}")

if __name__ == '__main__':
    main()
