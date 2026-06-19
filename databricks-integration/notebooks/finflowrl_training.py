# Databricks notebook source
# MAGIC %md
# MAGIC # FinFlowRL — Databricks Training Pipeline
# MAGIC
# MAGIC Two-stage RL training for HFT market-making:
# MAGIC 1. **Data Pipeline**: Alpha Vantage → Bronze → Silver → Gold (Delta Lake)
# MAGIC 2. **Pretrain**: Expert distillation via conditional flow-matching
# MAGIC 3. **Finetune**: PPO optimization against simulated market rewards
# MAGIC
# MAGIC All experiments tracked via MLflow on Databricks.

# COMMAND ----------

# MAGIC %pip install requests pyyaml tqdm numpy

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

mlflow.set_experiment("/Shared/FinFlowRL/training")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure Cluster + Catalog

# COMMAND ----------

CATALOG = "finflowrl"
SCHEMA = "market_data"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run Data Pipeline (Bronze → Silver → Gold)

# COMMAND ----------

# MAGIC %run ../pipelines/tick_data_lakehouse

# COMMAND ----------

run_full_pipeline(symbol="SPY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify Delta Tables

# COMMAND ----------

bronze_df = spark.read.format("delta").table(f"{CATALOG}.{SCHEMA}.bronze_ohlcv")
silver_df = spark.read.format("delta").table(f"{CATALOG}.{SCHEMA}.silver_features")
gold_df = spark.read.format("delta").table(f"{CATALOG}.{SCHEMA}.gold_training")

print(f"Bronze: {bronze_df.count()} rows")
print(f"Silver: {silver_df.count()} rows")
print(f"Gold:   {gold_df.count()} rows")

display(silver_df.orderBy(F.desc("date")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Jump Detection Analysis

# COMMAND ----------

jumps = silver_df.filter(F.col("is_jump") == True)
print(f"Detected {jumps.count()} jump events")
display(jumps.select("symbol", "date", "log_return", "realized_vol").orderBy(F.desc("date")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Tracked Training Pipeline

# COMMAND ----------

# MAGIC %run ../pipelines/mlflow_training

# COMMAND ----------

from pipelines.mlflow_training import TrainingConfig, run_tracked_training

config = TrainingConfig(
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

run_id = run_tracked_training(config)
print(f"Training run: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Review MLflow Results

# COMMAND ----------

experiment = mlflow.get_experiment_by_name("/Shared/FinFlowRL/training")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
display(runs[["run_id", "tags.stage", "metrics.final_pnl", "metrics.final_sharpe", "status"]])
