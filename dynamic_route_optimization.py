# dynamic_route_optimization.py

import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

class EvacuationRouteEnv(gym.Env):
    """
    Custom environment for evacuation route optimization using deep reinforcement learning.

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
        super(EvacuationRouteEnv, self).__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Define the observation and action spaces
        self.observation_space = gym.spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

        # Initialize the agent's and destination's positions
        self.agent_pos = None
        self.destination_pos = None

        self.current_step = 0

    def reset(self):
        # Reset the environment and return the initial observation
        self.agent_pos = self._get_random_position()
        self.destination_pos = self._get_random_position()
        self.current_step = 0

        return self._get_observation()

    def step(self, action):
        # Execute one step in the environment based on the given action
        self._move_agent(action)
        self.current_step += 1

        # Check if the agent has reached the destination
        done = self._is_at_destination() or self.current_step >= self.max_steps

        # Calculate the reward based on the agent's position and destination
        reward = 0 if self._is_at_destination() else -1

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        # Render the environment (optional)
        pass

    def close(self):
        # Close the environment and perform any necessary cleanup (optional)
        pass

    def _get_random_position(self):
        # Generate a random position within the grid
        return np.random.randint(0, self.grid_size, size=2)

    def _move_agent(self, action):
        # Update the agent's position based on the given action
        if action == 0:  # Move up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Move right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Move down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 3:  # Move left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

    def _is_at_destination(self):
        # Check if the agent has reached the destination
        return np.array_equal(self.agent_pos, self.destination_pos)

    def _get_observation(self):
        # Get the current observation based on the agent's and destination's positions
        return np.concatenate((self.agent_pos, self.destination_pos))


def optimize_routes(env, model_name="dqn_evacuation_route", total_timesteps=10000, save_model=True):
    """
    Optimize evacuation routes using deep reinforcement learning (DRL).

    Args:
        env (gym.Env): The evacuation route optimization environment.
        model_name (str): The name of the DQN model to be saved or loaded.
        total_timesteps (int): The total number of timesteps to train the model.
        save_model (bool): Whether to save the trained model.

    Returns:
        model (stable_baselines3.DQN): The trained DQN model.
    """
    try:
        # Check if the environment is valid
        check_env(env, warn=True)

        # Create the DQN model
        model = DQN("MlpPolicy", env, verbose=1)

        # Train the model
        model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        if save_model:
            model.save(model_name)

        return model

    except ValueError as e:
        print(f"Error: Invalid environment or model configuration: {str(e)}")
        return None

    except FileNotFoundError as e:
        print(f"Error: File not found: {str(e)}")
        return None

    except Exception as e:
        print(f"Error: An unexpected error occurred during model training: {str(e)}")
        return None


def evaluate_model(env, model, num_episodes=10):
    """
    Evaluate the trained model on the evacuation route optimization environment.

    Args:
        env (gym.Env): The evacuation route optimization environment.
        model (stable_baselines3.DQN): The trained DQN model.
        num_episodes (int): The number of episodes to evaluate the model.

    Returns:
        avg_reward (float): The average reward obtained by the model over the evaluated episodes.
        avg_steps (float): The average number of steps taken by the model over the evaluated episodes.
    """
    total_reward = 0
    total_steps = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1

        total_reward += episode_reward
        total_steps += episode_steps

    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes

    return avg_reward, avg_steps


def main():
    # Create the evacuation route optimization environment
    env = EvacuationRouteEnv(grid_size=10, max_steps=100)

    # Optimize the evacuation routes using DRL
    model = optimize_routes(env, model_name="dqn_evacuation_route", total_timesteps=10000)

    if model is None:
        print("Error: Failed to optimize routes.")
        return

    # Evaluate the trained model
    avg_reward, avg_steps = evaluate_model(env, model, num_episodes=10)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
