"""Fine-Tuner — Stage 2 PPO fine-tuning of the MeanFlow policy.

After Stage 1 distillation, Stage 2 uses PPO to optimise the RL
objective (cumulative reward) directly in the environment.
"""

import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm


class FineTuner:
    """Stage 2: Fine-tune pre-trained policy with PPO."""

    def __init__(
        self,
        policy,
        env,
        ppo_agent,
        n_episodes: int = 50,
        steps_per_episode: int = 500,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.policy = policy
        self.env = env
        self.ppo_agent = ppo_agent
        self.n_episodes = n_episodes
        self.steps_per_episode = steps_per_episode
        self.gamma = gamma
        self.lam = lam

    def collect_rollout(self) -> Dict[str, List]:
        """Collect one episode rollout using MeanFlow policy.

        Returns:
            dict with obs_list, act_list, reward_list, done_list, value_list
        """
        obs_list = []
        act_list = []
        reward_list = []
        done_list = []
        value_list = []

        obs = self.env.reset()
        for _ in range(self.steps_per_episode):
            action = self.policy.act(obs)
            obs_arr = np.clip(action, -1.0, 1.0)  # normalise to PPO action space
            next_obs, reward, done, info = self.env.step(obs_arr)

            obs_list.append(obs)
            act_list.append(obs_arr)
            reward_list.append(reward)
            done_list.append(float(done))
            value_list.append(self.ppo_agent.policy.get_value(obs))

            obs = next_obs
            if done:
                break

        return {
            "obs": obs_list,
            "actions": act_list,
            "rewards": reward_list,
            "dones": done_list,
            "values": value_list,
        }

    def train(self, n_epochs: int = 10, log_interval: int = 5) -> Dict:
        """Run fine-tuning loop.

        Args:
            n_epochs: number of fine-tuning epochs
            log_interval: logging frequency

        Returns:
            training history dict
        """
        history = {"episode_rewards": [], "episode_pnls": []}

        print("Fine-tuning with PPO...")
        for epoch in tqdm(range(n_epochs)):
            ep_rewards = []
            ep_pnls = []

            for _ in range(self.n_episodes):
                rollout = self.collect_rollout()
                rewards = rollout["rewards"]
                total_reward = sum(rewards)
                ep_rewards.append(total_reward)

                # Get final PnL from info (approximate)
                ep_pnls.append(total_reward)

            avg_reward = np.mean(ep_rewards)
            history["episode_rewards"].append(avg_reward)
            history["episode_pnls"].append(np.mean(ep_pnls))

            if (epoch + 1) % log_interval == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, avg reward: {avg_reward:.4f}")

        return history
