"""
Airbyte-backed market data feeds for FinFlowRL.

Provides real historical market data as an alternative to the synthetic
MarketSimulator. When Airbyte connectors for financial data become available,
this module will serve as the integration point.

Current state: Airbyte's agent SDK has 48 connectors (CRM, support, billing,
marketing, dev tools, analytics). Financial market data connectors are not yet
available, but the integration pattern is established here for when they are.

For now, this module provides:
1. A bridge to fetch real data via direct API calls (Alpha Vantage, FRED)
2. The Airbyte SDK integration pattern ready for future connectors
3. A data adapter that converts real OHLCV data into the simulator's format

Setup:
    export AIRBYTE_CLIENT_ID=<your_client_id>
    export AIRBYTE_CLIENT_SECRET=<your_client_secret>
    export ALPHA_VANTAGE_API_KEY=<your_key>  # For direct API bridge
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Airbyte SDK bridge
# ---------------------------------------------------------------------------

_airbyte_available = False
try:
    from airbyte_agent_sdk import connect, Workspace, AirbyteError
    _airbyte_available = True
except ImportError:
    logger.info("airbyte-agent-sdk not installed; using direct API or synthetic data")


def is_airbyte_available() -> bool:
    if not _airbyte_available:
        return False
    return bool(os.environ.get("AIRBYTE_CLIENT_ID") and os.environ.get("AIRBYTE_CLIENT_SECRET"))


# ---------------------------------------------------------------------------
# Direct Alpha Vantage bridge (until Airbyte adds financial connectors)
# ---------------------------------------------------------------------------

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


def fetch_daily_ohlcv(
    symbol: str,
    api_key: Optional[str] = None,
    output_size: str = "compact",
) -> List[Dict[str, Any]]:
    """
    Fetch daily OHLCV data from Alpha Vantage.
    
    Returns list of dicts with keys:
        date, open, high, low, close, volume
    Sorted by date ascending (oldest first).
    """
    key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise ValueError("ALPHAVANTAGE_API_KEY is not configured")
    
    params = urlencode({
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": output_size,
        "apikey": key,
    })
    url = f"{ALPHA_VANTAGE_BASE}?{params}"
    
    with urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    ts_key = "Time Series (Daily)"
    if ts_key not in data:
        error = data.get("Note") or data.get("Error Message") or "Unknown error"
        raise ValueError(f"Alpha Vantage: {error}")
    
    records = []
    for date_str, values in sorted(data[ts_key].items()):
        records.append({
            "date": date_str,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "volume": int(values["5. volume"]),
        })
    
    return records


# ---------------------------------------------------------------------------
# Data adapter: real data → simulator format
# ---------------------------------------------------------------------------

def ohlcv_to_simulator_state(
    ohlcv_records: List[Dict[str, Any]],
    lookback: int = 100,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert OHLCV records to the format expected by MarketSimulator.
    
    Returns:
        Tuple of (mid_prices_array, metadata_dict)
        
    The mid_prices_array can be used to calibrate the simulator's
    parameters (mu, sigma, jump_intensity) from real data, or to
    create a RealDataMarket that replays historical prices.
    """
    if not ohlcv_records:
        raise ValueError("No OHLCV records provided")
    
    # Extract close prices
    closes = np.array([r["close"] for r in ohlcv_records[-lookback:]], dtype=np.float64)
    
    # Calculate realized volatility (daily returns std)
    returns = np.diff(np.log(closes))
    realized_vol = float(np.std(returns)) if len(returns) > 1 else 0.02
    
    # Calculate drift (mean daily return)
    realized_drift = float(np.mean(returns)) if len(returns) > 1 else 0.0
    
    # Detect jumps (returns > 3 sigma)
    jump_threshold = 3 * realized_vol
    jump_count = int(np.sum(np.abs(returns) > jump_threshold))
    jump_intensity = jump_count / len(returns) if len(returns) > 0 else 0.1
    
    metadata = {
        "symbol": ohlcv_records[-1].get("symbol", "UNKNOWN"),
        "start_date": ohlcv_records[0]["date"],
        "end_date": ohlcv_records[-1]["date"],
        "n_observations": len(closes),
        "realized_volatility_annualized": realized_vol * np.sqrt(252),
        "realized_drift_annualized": realized_drift * 252,
        "estimated_jump_intensity": jump_intensity,
        "latest_close": float(closes[-1]),
        "price_range": [float(np.min(closes)), float(np.max(closes))],
    }
    
    return closes, metadata


def calibrate_simulator_from_data(
    ohlcv_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calibrate MarketSimulator parameters from real market data.
    
    Returns a dict of parameters that can be passed to MarketSimulator.__init__():
        S0, mu, sigma, jump_intensity, jump_mean, jump_std
    """
    closes, meta = ohlcv_to_simulator_state(ohlcv_records)
    returns = np.diff(np.log(closes))
    
    # Fit jump-diffusion parameters
    jump_threshold = 3 * meta["realized_volatility_annualized"] / np.sqrt(252)
    jump_returns = returns[np.abs(returns) > jump_threshold]
    normal_returns = returns[np.abs(returns) <= jump_threshold]
    
    params = {
        "S0": float(closes[-1]),
        "mu": float(np.mean(normal_returns)) * 252 if len(normal_returns) > 0 else 0.0,
        "sigma": float(np.std(normal_returns)) * np.sqrt(252) if len(normal_returns) > 0 else 0.15,
        "jump_intensity": len(jump_returns) / len(returns) if len(returns) > 0 else 0.05,
        "jump_mean": float(np.mean(jump_returns)) if len(jump_returns) > 0 else -0.02,
        "jump_std": float(np.std(jump_returns)) if len(jump_returns) > 1 else 0.03,
    }
    
    logger.info(
        "Calibrated simulator from %d observations: vol=%.2f%%, drift=%.2f%%, jump_int=%.4f",
        len(closes),
        params["sigma"] * 100,
        params["mu"] * 100,
        params["jump_intensity"],
    )
    
    return params


# ---------------------------------------------------------------------------
# RealDataMarket: replays historical prices for RL training
# ---------------------------------------------------------------------------

class RealDataMarket:
    """
    A market environment backed by real historical data.
    
    Drop-in replacement for MarketSimulator that replays actual OHLCV data
    instead of generating synthetic prices. Useful for:
    - Backtesting RL strategies on real data
    - Validating simulator calibration
    - A/B testing synthetic vs real market conditions
    """
    
    def __init__(self, ohlcv_records: List[Dict[str, Any]], initial_cash: float = 1_000_000):
        self.closes = np.array([r["close"] for r in ohlcv_records], dtype=np.float64)
        self.highs = np.array([r["high"] for r in ohlcv_records], dtype=np.float64)
        self.lows = np.array([r["low"] for r in ohlcv_records], dtype=np.float64)
        self.volumes = np.array([r["volume"] for r in ohlcv_records], dtype=np.float64)
        self.dates = [r["date"] for r in ohlcv_records]
        self.initial_cash = initial_cash
        self.current_step = 0
        self.n_steps = len(self.closes)
    
    def reset(self) -> np.ndarray:
        """Reset to the beginning of the historical data."""
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Advance one time step.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
        
        Returns:
            (observation, reward, done, info)
        """
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Simple PnL reward
        if self.current_step > 0:
            price_change = self.closes[self.current_step] - self.closes[self.current_step - 1]
            position = (action - 1)  # -1=sell, 0=hold, 1=buy
            reward = position * price_change / self.closes[self.current_step - 1]
        else:
            reward = 0.0
        
        info = {
            "date": self.dates[self.current_step] if self.current_step < len(self.dates) else "",
            "price": float(self.closes[self.current_step]) if self.current_step < self.n_steps else 0.0,
            "step": self.current_step,
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current market state as observation vector."""
        idx = min(self.current_step, self.n_steps - 1)
        window = 10
        start = max(0, idx - window + 1)
        prices = self.closes[start:idx + 1]
        
        # Pad if needed
        if len(prices) < window:
            prices = np.concatenate([np.full(window - len(prices), prices[0]), prices])
        
        # Normalize: returns + normalized volume
        returns = np.diff(np.log(prices))
        norm_vol = self.volumes[idx] / (np.mean(self.volumes) + 1e-8)
        
        obs = np.concatenate([returns, [norm_vol, prices[-1] / prices[0] - 1]])
        return obs
    
    @property
    def observation_space(self) -> int:
        return 12  # 10 returns + volume + cumulative return


# ---------------------------------------------------------------------------
# Airbyte-powered data pipeline (future-ready)
# ---------------------------------------------------------------------------

async def fetch_market_data_via_airbyte(
    symbol: str,
    connector_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch market data via Airbyte connector (future).
    
    When Airbyte adds financial data connectors (Alpha Vantage, Polygon,
    Yahoo Finance), this function will use them. For now, it falls back
    to direct API calls.
    
    Pattern for when connectors become available:
    
        alpha_vantage = connect("alpha_vantage", connector_id=connector_id)
        result = await alpha_vantage.execute("equities", "daily", params={
            "symbol": symbol,
            "outputsize": "full",
        })
        return result.data
    """
    if is_airbyte_available():
        logger.info(
            "Airbyte SDK available but no financial data connector exists yet. "
            "Using direct Alpha Vantage API bridge."
        )
    
    # Direct API bridge
    return fetch_daily_ohlcv(symbol)


def get_mcp_config() -> Dict[str, Any]:
    """Return MCP server config for AI agent access to market data."""
    return {
        "mcp_server_url": "https://mcp.airbyte.ai/mcp",
        "setup": {
            "claude_code": "claude mcp add --transport http airbyte-agent https://mcp.airbyte.ai/mcp",
            "cursor": '{"mcpServers": {"Agent MCP": {"url": "https://mcp.airbyte.ai/mcp"}}}',
        },
        "future_connectors": [
            "Alpha Vantage (for OHLCV and technical indicators)",
            "Polygon.io (for real-time and historical market data)",
            "FRED (for macroeconomic indicators)",
        ],
    }
