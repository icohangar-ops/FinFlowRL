"""
FinFlowRL — MLflow Training Integration

Wraps the two-stage RL training pipeline (pretrain + finetune) with
MLflow experiment tracking on Databricks:

  Stage 1: Expert distillation via flow-matching (MeanFlow model)
  Stage 2: PPO fine-tuning against simulated market rewards

Logs hyperparameters, training curves, model checkpoints, and
evaluation metrics (PnL, Sharpe, max drawdown) to MLflow.
"""

import mlflow
import mlflow.pyfunc
import json
import os
import time
import numpy as np
from dataclasses import dataclass, asdict


EXPERIMENT_NAME = "/Shared/FinFlowRL/training"


@dataclass
class TrainingConfig:
    hidden_sizes: list[int]
    pretrain_epochs: int
    pretrain_lr: float
    finetune_episodes: int
    ppo_lr: float
    ppo_clip: float
    gamma: float
    simulator_s0: float
    simulator_mu: float
    simulator_sigma: float
    jump_intensity: float
    expert_strategy: str


DEFAULT_CONFIG = TrainingConfig(
    hidden_sizes=[128, 128, 64],
    pretrain_epochs=100,
    pretrain_lr=1e-3,
    finetune_episodes=1000,
    ppo_lr=3e-4,
    ppo_clip=0.2,
    gamma=0.99,
    simulator_s0=100.0,
    simulator_mu=0.0,
    simulator_sigma=0.02,
    jump_intensity=0.1,
    expert_strategy="avellaneda_stoikov",
)


class MeanFlowMLflowModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for the MeanFlow conditional flow-matching model."""

    def __init__(self, params=None):
        self.params = params

    def load_context(self, context):
        with open(context.artifacts["params_path"], "r") as f:
            self.params = json.load(f)

    def predict(self, context, model_input):
        state = model_input.values if hasattr(model_input, "values") else model_input
        # Forward pass through flow-matching network
        # In production this would run the ODE integration
        return {"action": "placeholder", "state_dim": len(state[0]) if len(state) > 0 else 0}


def log_pretrain_stage(config: TrainingConfig, metrics: dict, params_path: str):
    """Log the expert distillation (pretrain) stage to MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="pretrain_flow_matching") as run:
        mlflow.set_tag("stage", "pretrain")
        mlflow.set_tag("expert", config.expert_strategy)
        mlflow.set_tag("model_type", "MeanFlow")

        mlflow.log_params({
            "hidden_sizes": str(config.hidden_sizes),
            "pretrain_epochs": config.pretrain_epochs,
            "pretrain_lr": config.pretrain_lr,
            "expert_strategy": config.expert_strategy,
            "simulator_s0": config.simulator_s0,
            "simulator_sigma": config.simulator_sigma,
        })

        for epoch, loss in enumerate(metrics.get("epoch_losses", [])):
            mlflow.log_metric("distillation_loss", loss, step=epoch)

        mlflow.log_metric("final_distillation_loss", metrics.get("final_loss", 0))
        mlflow.log_metric("expert_action_mse", metrics.get("action_mse", 0))
        mlflow.log_metric("model_params_count", metrics.get("param_count", 78000))

        if os.path.exists(params_path):
            mlflow.log_artifact(params_path, "model_checkpoints")
            mlflow.pyfunc.log_model(
                artifact_path="meanflow_model",
                python_model=MeanFlowMLflowModel(),
                artifacts={"params_path": params_path},
            )

        return run.info.run_id


def log_finetune_stage(
    config: TrainingConfig,
    pretrain_run_id: str,
    episode_metrics: list[dict],
    final_params_path: str,
):
    """Log the PPO fine-tuning stage to MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="finetune_ppo") as run:
        mlflow.set_tag("stage", "finetune")
        mlflow.set_tag("pretrain_run_id", pretrain_run_id)
        mlflow.set_tag("model_type", "MeanFlow+PPO")

        mlflow.log_params({
            "finetune_episodes": config.finetune_episodes,
            "ppo_lr": config.ppo_lr,
            "ppo_clip": config.ppo_clip,
            "gamma": config.gamma,
            "simulator_mu": config.simulator_mu,
            "jump_intensity": config.jump_intensity,
        })

        for ep in episode_metrics:
            step = ep["episode"]
            mlflow.log_metric("episode_pnl", ep.get("pnl", 0), step=step)
            mlflow.log_metric("episode_sharpe", ep.get("sharpe", 0), step=step)
            mlflow.log_metric("episode_max_drawdown", ep.get("max_drawdown", 0), step=step)
            mlflow.log_metric("policy_loss", ep.get("policy_loss", 0), step=step)
            mlflow.log_metric("value_loss", ep.get("value_loss", 0), step=step)
            mlflow.log_metric("entropy", ep.get("entropy", 0), step=step)

        if episode_metrics:
            final = episode_metrics[-1]
            mlflow.log_metric("final_pnl", final.get("pnl", 0))
            mlflow.log_metric("final_sharpe", final.get("sharpe", 0))
            mlflow.log_metric("final_max_drawdown", final.get("max_drawdown", 0))

        if os.path.exists(final_params_path):
            mlflow.log_artifact(final_params_path, "model_checkpoints")

        return run.info.run_id


def run_tracked_training(config: TrainingConfig = DEFAULT_CONFIG):
    """
    Execute the full two-stage training pipeline with MLflow tracking.

    In production, this calls the actual FinFlowRL training code.
    This scaffold demonstrates the MLflow integration pattern.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full_training_pipeline") as parent_run:
        mlflow.log_params(asdict(config))
        mlflow.set_tag("pipeline", "two_stage_rl")

        # Stage 1: Pretrain
        print("Stage 1: Expert distillation...")
        pretrain_metrics = {
            "epoch_losses": [float(1.0 / (i + 1)) for i in range(config.pretrain_epochs)],
            "final_loss": 0.01,
            "action_mse": 0.005,
            "param_count": 78000,
        }
        pretrain_run_id = log_pretrain_stage(
            config, pretrain_metrics, "checkpoints/meanflow_pretrain.json"
        )

        # Stage 2: Finetune
        print("Stage 2: PPO fine-tuning...")
        episode_metrics = [
            {
                "episode": i,
                "pnl": float(np.random.randn() * 100 + i * 0.5),
                "sharpe": float(0.5 + i * 0.002),
                "max_drawdown": float(max(0, 500 - i * 0.3)),
                "policy_loss": float(0.5 / (i + 1)),
                "value_loss": float(1.0 / (i + 1)),
                "entropy": float(2.0 - i * 0.001),
            }
            for i in range(min(100, config.finetune_episodes))
        ]
        finetune_run_id = log_finetune_stage(
            config, pretrain_run_id, episode_metrics, "checkpoints/meanflow_finetuned.json"
        )

        mlflow.set_tag("pretrain_run_id", pretrain_run_id)
        mlflow.set_tag("finetune_run_id", finetune_run_id)

        print(f"Training pipeline complete. Parent run: {parent_run.info.run_id}")
        return parent_run.info.run_id


if __name__ == "__main__":
    run_tracked_training()
