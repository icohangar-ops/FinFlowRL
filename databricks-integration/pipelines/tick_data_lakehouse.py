"""
FinFlowRL — Delta Lake Tick Data Pipeline

Ingests tick data from Alpha Vantage and the Merton jump-diffusion simulator
into a Delta Lake medallion architecture:

  Bronze: Raw OHLCV + simulator output (append-only, partitioned by date)
  Silver: Cleaned ticks with realized volatility, jump detection, feature engineering
  Gold:   Aggregated training datasets ready for RL agent consumption

Designed to run as a Databricks job or notebook cell.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import mlflow
import requests
import json
import os
from datetime import datetime, timezone


CATALOG = os.environ.get("DATABRICKS_CATALOG", "finflowrl")
SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "market_data")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")


def get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


# ---------------------------------------------------------------------------
# Bronze Layer — Raw Ingestion
# ---------------------------------------------------------------------------


def ingest_alphavantage_daily(symbol: str = "SPY") -> DataFrame:
    """Fetch daily OHLCV from Alpha Vantage and write to Bronze."""
    spark = get_spark()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json().get("Time Series (Daily)", {})

    rows = []
    for date_str, values in data.items():
        rows.append({
            "symbol": symbol,
            "date": date_str,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "volume": int(values["5. volume"]),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })

    df = spark.createDataFrame(rows)
    df = df.withColumn("date", F.to_date("date"))
    df = df.withColumn("ingested_at", F.to_timestamp("ingested_at"))

    bronze_table = f"{CATALOG}.{SCHEMA}.bronze_ohlcv"
    df.write.format("delta").mode("append").partitionBy("date").saveAsTable(bronze_table)

    print(f"[Bronze] Ingested {len(rows)} rows for {symbol} -> {bronze_table}")
    return df


def ingest_simulator_output(sim_data: list[dict], sim_config: dict) -> DataFrame:
    """Write Merton jump-diffusion simulator output to Bronze."""
    spark = get_spark()

    for row in sim_data:
        row["sim_config"] = json.dumps(sim_config)
        row["ingested_at"] = datetime.now(timezone.utc).isoformat()

    df = spark.createDataFrame(sim_data)
    df = df.withColumn("ingested_at", F.to_timestamp("ingested_at"))

    bronze_table = f"{CATALOG}.{SCHEMA}.bronze_simulator"
    df.write.format("delta").mode("append").saveAsTable(bronze_table)

    print(f"[Bronze] Ingested {len(sim_data)} simulator ticks -> {bronze_table}")
    return df


# ---------------------------------------------------------------------------
# Silver Layer — Cleaning + Feature Engineering
# ---------------------------------------------------------------------------


def build_silver_features() -> DataFrame:
    """
    Clean bronze data and compute features used by the RL agent:
    - log returns
    - realized volatility (20-period rolling)
    - jump detection (>3 sigma moves)
    - bid-ask spread proxy (high-low range)
    - order flow imbalance proxy (volume delta)
    """
    spark = get_spark()

    bronze = spark.read.format("delta").table(f"{CATALOG}.{SCHEMA}.bronze_ohlcv")

    window_20 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)
    window_lag = Window.partitionBy("symbol").orderBy("date")

    silver = (
        bronze
        .withColumn("prev_close", F.lag("close").over(window_lag))
        .filter(F.col("prev_close").isNotNull())
        .withColumn("log_return", F.log(F.col("close") / F.col("prev_close")))
        .withColumn("realized_vol", F.stddev("log_return").over(window_20))
        .withColumn("vol_20_mean", F.mean("log_return").over(window_20))
        .withColumn(
            "is_jump",
            F.when(
                F.abs(F.col("log_return") - F.col("vol_20_mean")) > 3 * F.col("realized_vol"),
                True,
            ).otherwise(False),
        )
        .withColumn("spread_proxy", (F.col("high") - F.col("low")) / F.col("close"))
        .withColumn("prev_volume", F.lag("volume").over(window_lag))
        .withColumn(
            "volume_delta",
            F.when(
                F.col("prev_volume").isNotNull(),
                (F.col("volume") - F.col("prev_volume")) / F.col("prev_volume"),
            ).otherwise(0.0),
        )
        .drop("vol_20_mean", "prev_close", "prev_volume")
    )

    silver_table = f"{CATALOG}.{SCHEMA}.silver_features"
    silver.write.format("delta").mode("overwrite").saveAsTable(silver_table)

    print(f"[Silver] Built features -> {silver_table}")
    return silver


# ---------------------------------------------------------------------------
# Gold Layer — Training Datasets
# ---------------------------------------------------------------------------


def build_gold_training_dataset(lookback: int = 50) -> DataFrame:
    """
    Assemble gold-layer training dataset for the RL agent.
    Each row = one observation window with:
    - 6-dim state vector (matching hft_env.py observation space)
    - Expert labels from Avellaneda-Stoikov baseline
    """
    spark = get_spark()

    silver = spark.read.format("delta").table(f"{CATALOG}.{SCHEMA}.silver_features")

    window_lb = Window.partitionBy("symbol").orderBy("date").rowsBetween(-lookback + 1, 0)

    gold = (
        silver
        .withColumn("mid_price", (F.col("high") + F.col("low")) / 2)
        .withColumn("inventory", F.lit(0.0))  # neutral start
        .withColumn("pnl", F.sum("log_return").over(window_lb))
        .withColumn(
            "obs_state",
            F.struct(
                F.col("mid_price").alias("mid_price"),
                F.col("spread_proxy").alias("spread"),
                F.col("realized_vol").alias("volatility"),
                F.col("inventory").alias("inventory"),
                F.col("pnl").alias("pnl"),
                F.col("volume_delta").alias("order_flow"),
            ),
        )
        .withColumn("as_reservation_price", F.col("mid_price") - F.col("inventory") * F.col("realized_vol"))
        .withColumn("as_optimal_spread", F.col("realized_vol") + F.log(F.lit(2.0)) / F.lit(1.0))
        .withColumn(
            "expert_action",
            F.struct(
                F.col("as_reservation_price").alias("reservation_price"),
                F.col("as_optimal_spread").alias("optimal_spread"),
            ),
        )
        .select("symbol", "date", "obs_state", "expert_action", "log_return", "realized_vol", "is_jump")
    )

    gold_table = f"{CATALOG}.{SCHEMA}.gold_training"
    gold.write.format("delta").mode("overwrite").saveAsTable(gold_table)

    print(f"[Gold] Training dataset -> {gold_table} ({gold.count()} rows)")
    return gold


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(symbol: str = "SPY"):
    """Execute the complete Bronze → Silver → Gold pipeline."""
    with mlflow.start_run(run_name=f"tick_data_pipeline_{symbol}"):
        mlflow.set_tag("pipeline", "finflowrl_tick_data")
        mlflow.set_tag("symbol", symbol)

        bronze_df = ingest_alphavantage_daily(symbol)
        mlflow.log_metric("bronze_rows", bronze_df.count())

        silver_df = build_silver_features()
        mlflow.log_metric("silver_rows", silver_df.count())

        jump_count = silver_df.filter(F.col("is_jump") == True).count()
        mlflow.log_metric("jump_count", jump_count)

        gold_df = build_gold_training_dataset()
        mlflow.log_metric("gold_rows", gold_df.count())

        avg_vol = silver_df.agg(F.mean("realized_vol")).collect()[0][0]
        if avg_vol:
            mlflow.log_metric("avg_realized_vol", float(avg_vol))

        print(f"Pipeline complete for {symbol}")


if __name__ == "__main__":
    run_full_pipeline()
