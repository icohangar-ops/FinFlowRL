"""Pre-Trainer — Stage 1 expert distillation via flow matching.

Collects expert demonstrations and trains the MeanFlow policy to
match expert actions via the flow-matching loss.
"""

import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm


class PreTrainer:
    """Stage 1: Distill expert policies into MeanFlow via flow matching."""

    def __init__(
        self,
        policy,
        env,
        expert,
        n_episodes: int = 100,
        steps_per_episode: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
    ):
        self.policy = policy
        self.env = env
        self.expert = expert
        self.n_episodes = n_episodes
        self.steps_per_episode = steps_per_episode
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Replay buffer for (obs, expert_action) pairs
        self.obs_buffer: List[np.ndarray] = []
        self.action_buffer: List[np.ndarray] = []

    def collect_expert_demos(self) -> None:
        """Collect expert demonstrations by running expert in env."""
        self.obs_buffer.clear()
        self.action_buffer.clear()

        for ep in range(self.n_episodes):
            obs = self.env.reset()
            total_reward = 0.0
            for step in range(self.steps_per_episode):
                # Build state dict for expert
                state = {
                    "inventory": obs[0] * self.env.max_position,
                    "mid_price": self.env.mid_price,
                    "prev_mid_price": self.env.prev_mid_price,
                    "spread": obs[2],
                    "volatility": obs[3] / 10.0,
                    "order_imbalance": obs[4] * 10.0,
                    "hawkes_intensity": obs[5] * 20.0,
                }

                expert_output = self.expert.act(state)

                # Different expert output formats
                if "target_position" in expert_output:
                    expert_action = np.array([expert_output["target_position"] / self.env.max_position])
                elif "half_spread" in expert_output:
                    # Convert bid/ask to position signal
                    mid = self.env.mid_price
                    spread = expert_output.get("half_spread", 0.01) * 2
                    signal = (expert_output["ask_price"] + expert_output["bid_price"]) / 2 - mid
                    expert_action = np.array([np.clip(signal * 100, -1, 1)])
                else:
                    expert_action = np.zeros(self.env.act_dim)

                self.obs_buffer.append(obs.copy())
                self.action_buffer.append(expert_action)

                # Take a random action to explore
                action = np.random.randn(self.env.act_dim) * 0.1
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break

    def train_step(self) -> float:
        """Perform one training step on a random batch.

        Returns:
            average flow-matching loss
        """
        n = len(self.obs_buffer)
        if n < self.batch_size:
            indices = np.arange(n)
        else:
            indices = np.random.choice(n, self.batch_size, replace=False)

        total_loss = 0.0
        for idx in indices:
            loss = self.policy.flow_loss(self.obs_buffer[idx], self.action_buffer[idx])
            total_loss += loss

            # Simple gradient-free update: nudge parameters
            self._nudge_params(loss)

        return total_loss / len(indices)

    def _nudge_params(self, loss: float, scale: float = 0.01) -> None:
        """Simple parameter update via small random perturbation (gradient-free)."""
        params = self.policy.get_params()
        for key in params:
            if isinstance(params[key], list):
                for i in range(len(params[key])):
                    params[key][i] -= scale * loss * np.sign(np.random.randn(*params[key][i].shape))
            elif isinstance(params[key], np.ndarray):
                params[key] -= scale * loss * np.sign(np.random.randn(*params[key].shape))
        self.policy.set_params(params)

    def train(self, n_iterations: int = 1000, log_interval: int = 100) -> Dict:
        """Run full pre-training loop.

        Args:
            n_iterations: number of training steps
            log_interval: how often to print progress

        Returns:
            training history dict
        """
        print("Collecting expert demonstrations...")
        self.collect_expert_demos()
        print(f"Collected {len(self.obs_buffer)} expert transitions.")

        history = {"loss": []}
        print("Pre-training MeanFlow policy via flow matching...")
        for i in tqdm(range(n_iterations)):
            loss = self.train_step()
            history["loss"].append(loss)
            if (i + 1) % log_interval == 0:
                avg_loss = np.mean(history["loss"][-log_interval:])
                print(f"  Step {i+1}/{n_iterations}, avg loss: {avg_loss:.6f}")

        return history
