#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
import statistics
import time
import anthropic
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import logging
import os
import warnings
import traceback
warnings.filterwarnings("ignore")

# Local imports
from utils.logger import logger
from config import config

class TechnicalIndicators:
    """Class for calculating technical indicators"""     

    @staticmethod
    def safe_max(sequence, default=None):
        """Safely get maximum value from a sequence, returning default if empty"""
        try:
            if not sequence or len(sequence) == 0:
                return default
            return max(sequence)
        except (ValueError, TypeError) as e:
            logger.log_error("TechnicalIndicators.safe_max", f"Error calculating max: {str(e)}")
            return default

    @staticmethod
    def safe_min(sequence, default=None):
        """Safely get minimum value from a sequence, returning default if empty"""
        try:
            if not sequence or len(sequence) == 0:
                return default
            return min(sequence)
        except (ValueError, TypeError) as e:
            logger.log_error("TechnicalIndicators.safe_min", f"Error calculating min: {str(e)}")
            return default
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index with improved error handling
        """
        try:
            if len(prices) < period + 1:
                return 50.0  # Default to neutral if not enough data
                
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Get gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Initial average gain and loss
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate for remaining periods
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                
            # Calculate RS and RSI
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.log_error("RSI Calculation", str(e))
            return 50.0  # Return neutral RSI on error
    
    @staticmethod
    def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence) with improved error handling
        Returns (macd_line, signal_line, histogram)
        """
        try:
            if len(prices) < slow_period + signal_period:
                return 0.0, 0.0, 0.0  # Default if not enough data
                
            # Convert to numpy array for efficiency
            prices_array = np.array(prices)
            
            # Calculate EMAs
            ema_fast = TechnicalIndicators.calculate_ema(prices_array, fast_period)
            ema_slow = TechnicalIndicators.calculate_ema(prices_array, slow_period)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate Signal line (EMA of MACD line)
            signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return macd_line[-1], signal_line[-1], histogram[-1]
        except Exception as e:
            logger.log_error("MACD Calculation", str(e))
            return 0.0, 0.0, 0.0  # Return neutral MACD on error
    
    @staticmethod
    def calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average with improved error handling
        """
        try:
            if len(values) == 0:
                return np.array([0.0])  # Return a default value for empty arrays
                
            if len(values) < 2 * period:
                # Pad with the first value if we don't have enough data
                padding = np.full(2 * period - len(values), values[0])
                values = np.concatenate((padding, values))
                
            alpha = 2 / (period + 1)
            
            # Calculate EMA
            ema = np.zeros_like(values)
            ema[0] = values[0]  # Initialize with first value
            
            for i in range(1, len(values)):
                ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
                
            return ema
        except Exception as e:
            logger.log_error("EMA Calculation", str(e))
            return np.array([values[0]]) if len(values) > 0 else np.array([0.0])
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands with improved error handling
        Returns (upper_band, middle_band, lower_band)
        """
        try:
            if not prices or len(prices) == 0:
                return 0.0, 0.0, 0.0  # Default for empty lists
                
            if len(prices) < period:
                # Not enough data, use last price with estimated bands
                last_price = prices[-1]
                estimated_volatility = 0.02 * last_price  # Estimate 2% volatility
                return (
                    last_price + num_std * estimated_volatility, 
                    last_price, 
                    last_price - num_std * estimated_volatility
                )
                
            # Calculate middle band (SMA)
            middle_band = sum(prices[-period:]) / period
            
            # Calculate standard deviation
            std = statistics.stdev(prices[-period:])
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            
            return upper_band, middle_band, lower_band
        except Exception as e:
            logger.log_error("Bollinger Bands Calculation", str(e))
            # Return default based on last price if available
            if prices and len(prices) > 0:
                last_price = prices[-1]
                return last_price * 1.02, last_price, last_price * 0.98
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def calculate_stochastic_oscillator(prices: List[float], highs: List[float], lows: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """
        Calculate Stochastic Oscillator with robust error handling
        Returns (%K, %D)
        """
        try:
            # Validate inputs
            if not prices or not highs or not lows:
                return 50.0, 50.0  # Default to mid-range if empty inputs
                
            if len(prices) < k_period or len(highs) < k_period or len(lows) < k_period:
                return 50.0, 50.0  # Default to mid-range if not enough data
                
            # Get last k_period prices, highs, lows
            recent_prices = prices[-k_period:]
            recent_highs = highs[-k_period:]
            recent_lows = lows[-k_period:]
            
            # Ensure we have the current price
            current_close = recent_prices[-1] if recent_prices else prices[-1]
            
            # Use safe methods to prevent empty sequence errors
            highest_high = TechnicalIndicators.safe_max(recent_highs, default=current_close)
            lowest_low = TechnicalIndicators.safe_min(recent_lows, default=current_close)
            
            # Avoid division by zero
            if highest_high == lowest_low:
                k = 50.0  # Default if there's no range
            else:
                k = 100 * ((current_close - lowest_low) / (highest_high - lowest_low))
                
            # Calculate %D (SMA of %K)
            if len(prices) < k_period + d_period - 1:
                d = k  # Not enough data for proper %D
            else:
                # We need historical %K values to calculate %D
                k_values = []
                
                # Safely calculate historical K values
                for i in range(d_period):
                    try:
                        idx = -(i + 1)  # Start from most recent and go backwards
                        
                        c = prices[idx]
                        
                        # Use safe min/max to avoid empty sequence errors
                        h = TechnicalIndicators.safe_max(highs[idx-k_period+1:idx+1], default=c)
                        l = TechnicalIndicators.safe_min(lows[idx-k_period+1:idx+1], default=c)
                        
                        if h == l:
                            k_values.append(50.0)
                        else:
                            k_values.append(100 * ((c - l) / (h - l)))
                    except (IndexError, ZeroDivisionError):
                        # Handle any unexpected errors
                        k_values.append(50.0)
                        
                # Average the k values to get %D
                d = sum(k_values) / len(k_values) if k_values else k
                
            return k, d
        except Exception as e:
            logger.log_error("Stochastic Oscillator Calculation", str(e))
            return 50.0, 50.0  # Return middle values on error
    
    @staticmethod
    def calculate_volume_profile(volumes: List[float], prices: List[float], num_levels: int = 10) -> Dict[str, float]:
        """
        Calculate Volume Profile with improved error handling
        Returns a dictionary mapping price levels to volume percentages
        """
        try:
            if not volumes or not prices or len(volumes) != len(prices):
                return {}
                
            # Get min and max price with safe methods
            min_price = TechnicalIndicators.safe_min(prices, default=0)
            max_price = TechnicalIndicators.safe_max(prices, default=0)
            
            if min_price == max_price:
                return {str(min_price): 100.0}
                
            # Create price levels
            bin_size = (max_price - min_price) / num_levels
            levels = [min_price + i * bin_size for i in range(num_levels + 1)]
            
            # Initialize volume profile
            volume_profile = {f"{round(levels[i], 2)}-{round(levels[i+1], 2)}": 0 for i in range(num_levels)}
            
            # Distribute volumes across price levels
            total_volume = sum(volumes)
            if total_volume == 0:
                return {key: 0.0 for key in volume_profile}
                
            for price, volume in zip(prices, volumes):
                # Find the bin this price belongs to
                for i in range(num_levels):
                    if levels[i] <= price < levels[i+1] or (i == num_levels - 1 and price == levels[i+1]):
                        key = f"{round(levels[i], 2)}-{round(levels[i+1], 2)}"
                        volume_profile[key] += volume
                        break
                        
            # Convert to percentages
            for key in volume_profile:
                volume_profile[key] = (volume_profile[key] / total_volume) * 100
                
            return volume_profile
        except Exception as e:
            logger.log_error("Volume Profile Calculation", str(e))
            return {}
    
    @staticmethod
    def calculate_obv(prices: List[float], volumes: List[float]) -> float:
        """
        Calculate On-Balance Volume (OBV) with improved error handling
        """
        try:
            if not prices or not volumes:
                return 0.0  # Default for empty lists
                
            if len(prices) < 2 or len(volumes) < 2:
                return volumes[0] if volumes else 0.0
                
            obv = volumes[0]
            
            # Calculate OBV
            for i in range(1, min(len(prices), len(volumes))):
                if prices[i] > prices[i-1]:
                    obv += volumes[i]
                elif prices[i] < prices[i-1]:
                    obv -= volumes[i]
                    
            return obv
        except Exception as e:
            logger.log_error("OBV Calculation", str(e))
            return 0.0
    
    @staticmethod
    def calculate_adx(highs: List[float], lows: List[float], prices: List[float], period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX) with improved error handling
        """
        try:
            # Validate inputs
            if not highs or not lows or not prices:
                return 25.0  # Default to moderate trend strength if empty inputs
                
            if len(highs) < 2 * period or len(lows) < 2 * period or len(prices) < 2 * period:
                return 25.0  # Default to moderate trend strength if not enough data
                
            # Calculate +DM and -DM
            plus_dm = []
            minus_dm = []
            
            for i in range(1, len(highs)):
                h_diff = highs[i] - highs[i-1]
                l_diff = lows[i-1] - lows[i]
                
                if h_diff > l_diff and h_diff > 0:
                    plus_dm.append(h_diff)
                else:
                    plus_dm.append(0)
                    
                if l_diff > h_diff and l_diff > 0:
                    minus_dm.append(l_diff)
                else:
                    minus_dm.append(0)
                    
            # Calculate True Range
            tr = []
            for i in range(1, len(prices)):
                tr1 = abs(highs[i] - lows[i])
                tr2 = abs(highs[i] - prices[i-1])
                tr3 = abs(lows[i] - prices[i-1])
                tr.append(max(tr1, tr2, tr3))
                
            # Handle case where tr is empty
            if not tr:
                return 25.0
                
            # Calculate ATR (Average True Range)
            atr = sum(tr[:period]) / period if period <= len(tr) else sum(tr) / len(tr)
            
            # Avoid division by zero
            if atr == 0:
                atr = 0.0001  # Small non-zero value
                
            # Calculate +DI and -DI
            plus_di = sum(plus_dm[:period]) / atr if period <= len(plus_dm) else sum(plus_dm) / atr
            minus_di = sum(minus_dm[:period]) / atr if period <= len(minus_dm) else sum(minus_dm) / atr
            
            # Calculate DX (Directional Index)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            
            # Calculate ADX (smoothed DX)
            adx = dx
            
            # Process the remaining periods
            max_period = min(len(tr), len(plus_dm), len(minus_dm))
            for i in range(period, max_period):
                # Update ATR
                atr = ((period - 1) * atr + tr[i]) / period
                
                # Avoid division by zero
                if atr == 0:
                    atr = 0.0001  # Small non-zero value
                
                # Update +DI and -DI
                plus_di = ((period - 1) * plus_di + plus_dm[i]) / period
                minus_di = ((period - 1) * minus_di + minus_dm[i]) / period
                
                # Update DX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
                
                # Smooth ADX
                adx = ((period - 1) * adx + dx) / period
                
            return adx
        except Exception as e:
            logger.log_error("ADX Calculation", str(e))
            return 25.0  # Return moderate trend strength on error
    
    @staticmethod
    def calculate_ichimoku(prices: List[float], highs: List[float], lows: List[float], 
                         tenkan_period: int = 9, kijun_period: int = 26, 
                         senkou_b_period: int = 52) -> Dict[str, float]:
        """
        Calculate Ichimoku Cloud components with improved error handling
        Returns key Ichimoku components
        """
        try:
            # Validate inputs
            if not prices or not highs or not lows:
                return {
                    "tenkan_sen": 0, 
                    "kijun_sen": 0,
                    "senkou_span_a": 0, 
                    "senkou_span_b": 0
                }
            
            # Get default value for calculations
            default_value = prices[-1] if prices else 0
            
            # Check if we have enough data
            if len(prices) < senkou_b_period or len(highs) < senkou_b_period or len(lows) < senkou_b_period:
                return {
                    "tenkan_sen": default_value, 
                    "kijun_sen": default_value,
                    "senkou_span_a": default_value, 
                    "senkou_span_b": default_value
                }
            
            # Calculate Tenkan-sen (Conversion Line) with safe methods
            high_tenkan = TechnicalIndicators.safe_max(highs[-tenkan_period:], default=default_value)
            low_tenkan = TechnicalIndicators.safe_min(lows[-tenkan_period:], default=default_value)
            tenkan_sen = (high_tenkan + low_tenkan) / 2
    
            # Calculate Kijun-sen (Base Line) with safe methods
            high_kijun = TechnicalIndicators.safe_max(highs[-kijun_period:], default=default_value)
            low_kijun = TechnicalIndicators.safe_min(lows[-kijun_period:], default=default_value)
            kijun_sen = (high_kijun + low_kijun) / 2
    
            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
            # Calculate Senkou Span B (Leading Span B) with safe methods
            high_senkou = TechnicalIndicators.safe_max(highs[-senkou_b_period:], default=default_value)
            low_senkou = TechnicalIndicators.safe_min(lows[-senkou_b_period:], default=default_value)
            senkou_span_b = (high_senkou + low_senkou) / 2
            
            return {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b
            }
        except Exception as e:
            logger.log_error("Ichimoku Calculation", str(e))
            default_value = prices[-1] if prices and len(prices) > 0 else 0
            return {
                "tenkan_sen": default_value, 
                "kijun_sen": default_value,
                "senkou_span_a": default_value, 
                "senkou_span_b": default_value
            }
    
    @staticmethod 
    def calculate_pivot_points(high: float, low: float, close: float, pivot_type: str = "standard") -> Dict[str, float]:
        """
        Calculate pivot points for support and resistance levels with improved error handling
        Supports standard, fibonacci, and woodie pivot types
        """
        try:
            # Default pivot point (avoid division by zero)
            if high == 0 and low == 0 and close == 0:
                return {
                    "pivot": 0,
                    "r1": 0, "r2": 0, "r3": 0,
                    "s1": 0, "s2": 0, "s3": 0
                }
                
            if pivot_type == "fibonacci":
                pivot = (high + low + close) / 3
                r1 = pivot + 0.382 * (high - low)
                r2 = pivot + 0.618 * (high - low)
                r3 = pivot + 1.0 * (high - low)
                s1 = pivot - 0.382 * (high - low)
                s2 = pivot - 0.618 * (high - low)
                s3 = pivot - 1.0 * (high - low)
            elif pivot_type == "woodie":
                pivot = (high + low + 2 * close) / 4
                r1 = 2 * pivot - low
                r2 = pivot + (high - low)
                s1 = 2 * pivot - high
                s2 = pivot - (high - low)
                r3 = r1 + (high - low)
                s3 = s1 - (high - low)
            else:  # standard
                pivot = (high + low + close) / 3
                r1 = 2 * pivot - low
                r2 = pivot + (high - low)
                r3 = r2 + (high - low)
                s1 = 2 * pivot - high
                s2 = pivot - (high - low)
                s3 = s2 - (high - low)
                
            return {
                "pivot": pivot,
                "r1": r1, "r2": r2, "r3": r3,
                "s1": s1, "s2": s2, "s3": s3
            }
        except Exception as e:
            logger.log_error("Pivot Points Calculation", str(e))
            # Return basic default values if calculation fails
            return {
                "pivot": close,
                "r1": close * 1.01, "r2": close * 1.02, "r3": close * 1.03,
                "s1": close * 0.99, "s2": close * 0.98, "s3": close * 0.97
            }
                
    @staticmethod
    def analyze_technical_indicators(prices: List[float], volumes: List[float], highs: List[float] = None, lows: List[float] = None, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze multiple technical indicators and return results with interpretations
        Improved error handling and support for different timeframes (1h, 24h, 7d)
        """
        try:
            # Validate inputs
            if not prices or len(prices) < 2:
                return {
                    "error": "Insufficient price data for technical analysis",
                    "overall_trend": "neutral",
                    "trend_strength": 50,
                    "signals": {
                        "rsi": "neutral",
                        "macd": "neutral",
                        "bollinger_bands": "neutral",
                        "stochastic": "neutral"
                    }
                }
                
                
            # Use closing prices for highs/lows if not provided
            if highs is None:
                highs = prices
            if lows is None:
                lows = prices
                
            # Adjust indicator parameters based on timeframe
            if timeframe == "24h":
                rsi_period = 14
                macd_fast, macd_slow, macd_signal = 12, 26, 9
                bb_period, bb_std = 20, 2.0
                stoch_k, stoch_d = 14, 3
            elif timeframe == "7d":
                rsi_period = 14
                macd_fast, macd_slow, macd_signal = 12, 26, 9
                bb_period, bb_std = 20, 2.0
                stoch_k, stoch_d = 14, 3
                # For weekly, we maintain similar parameters but apply them to weekly data
            else:  # 1h default
                rsi_period = 14
                macd_fast, macd_slow, macd_signal = 12, 26, 9
                bb_period, bb_std = 20, 2.0
                stoch_k, stoch_d = 14, 3
                
            # Calculate indicators with proper error handling
            rsi = TechnicalIndicators.calculate_rsi(prices, period=rsi_period)
            macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(
                prices, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal
            )
            upper_band, middle_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(
                prices, period=bb_period, num_std=bb_std
            )
            k, d = TechnicalIndicators.calculate_stochastic_oscillator(
                prices, highs, lows, k_period=stoch_k, d_period=stoch_d
            )
            obv = TechnicalIndicators.calculate_obv(prices, volumes) if volumes else 0
            
            # Calculate additional indicators for longer timeframes
            additional_indicators = {}
            if timeframe in ["24h", "7d"]:
                # Calculate ADX for trend strength
                adx = TechnicalIndicators.calculate_adx(highs, lows, prices)
                additional_indicators["adx"] = adx
                
                # Calculate Ichimoku Cloud for longer-term trend analysis
                ichimoku = TechnicalIndicators.calculate_ichimoku(prices, highs, lows)
                additional_indicators["ichimoku"] = ichimoku
                
                # Calculate Pivot Points for key support/resistance levels
                # Use recent high, low, close for pivot calculation
                if len(prices) >= 5:
                    high = TechnicalIndicators.safe_max(highs[-5:], default=prices[-1])
                    low = TechnicalIndicators.safe_min(lows[-5:], default=prices[-1])
                    close = prices[-1]
                    pivot_type = "fibonacci" if timeframe == "7d" else "standard"
                    pivots = TechnicalIndicators.calculate_pivot_points(high, low, close, pivot_type)
                    additional_indicators["pivot_points"] = pivots
            
            # Interpret RSI with timeframe context
            if timeframe == "1h":
                if rsi > 70:
                    rsi_signal = "overbought"
                elif rsi < 30:
                    rsi_signal = "oversold"
                else:
                    rsi_signal = "neutral"
            elif timeframe == "24h":
                # Slightly wider thresholds for daily
                if rsi > 75:
                    rsi_signal = "overbought"
                elif rsi < 25:
                    rsi_signal = "oversold"
                else:
                    rsi_signal = "neutral"
            else:  # 7d
                # Even wider thresholds for weekly
                if rsi > 80:
                    rsi_signal = "overbought"
                elif rsi < 20:
                    rsi_signal = "oversold"
                else:
                    rsi_signal = "neutral"
                
            # Interpret MACD
            if macd_line > signal_line and histogram > 0:
                macd_signal = "bullish"
            elif macd_line < signal_line and histogram < 0:
                macd_signal = "bearish"
            else:
                macd_signal = "neutral"
                
            # Interpret Bollinger Bands
            current_price = prices[-1]
            if current_price > upper_band:
                bb_signal = "overbought"
            elif current_price < lower_band:
                bb_signal = "oversold"
            else:
                # Check for Bollinger Band squeeze
                previous_bandwidth = (upper_band - lower_band) / middle_band if middle_band else 0.2
                if previous_bandwidth < 0.1:  # Tight bands indicate potential breakout
                    bb_signal = "squeeze"
                else:
                    bb_signal = "neutral"
                    
            # Interpret Stochastic
            if k > 80 and d > 80:
                stoch_signal = "overbought"
            elif k < 20 and d < 20:
                stoch_signal = "oversold"
            elif k > d:
                stoch_signal = "bullish"
            elif k < d:
                stoch_signal = "bearish"
            else:
                stoch_signal = "neutral"
                
            # Add ADX interpretation for longer timeframes
            adx_signal = "neutral"
            if timeframe in ["24h", "7d"] and "adx" in additional_indicators:
                adx_value = additional_indicators["adx"]
                if adx_value > 30:
                    adx_signal = "strong_trend"
                elif adx_value > 20:
                    adx_signal = "moderate_trend"
                else:
                    adx_signal = "weak_trend"
                    
            # Add Ichimoku interpretation for longer timeframes
            ichimoku_signal = "neutral"
            if timeframe in ["24h", "7d"] and "ichimoku" in additional_indicators:
                ichimoku_data = additional_indicators["ichimoku"]
                if (current_price > ichimoku_data["senkou_span_a"] and 
                    current_price > ichimoku_data["senkou_span_b"]):
                    ichimoku_signal = "bullish"
                elif (current_price < ichimoku_data["senkou_span_a"] and 
                      current_price < ichimoku_data["senkou_span_b"]):
                    ichimoku_signal = "bearish"
                else:
                    ichimoku_signal = "neutral"
                    
            # Determine overall signal
            signals = {
                "bullish": 0,
                "bearish": 0,
                "neutral": 0,
                "overbought": 0,
                "oversold": 0
            }
            
            # Count signals
            for signal in [rsi_signal, macd_signal, bb_signal, stoch_signal]:
                if signal in signals:
                    signals[signal] += 1
                
            # Add additional signals for longer timeframes
            if timeframe in ["24h", "7d"]:
                if adx_signal == "strong_trend" and macd_signal == "bullish":
                    signals["bullish"] += 1
                elif adx_signal == "strong_trend" and macd_signal == "bearish":
                    signals["bearish"] += 1
                    
                if ichimoku_signal == "bullish":
                    signals["bullish"] += 1
                elif ichimoku_signal == "bearish":
                    signals["bearish"] += 1
                
            # Determine trend strength and direction
            if signals["bullish"] + signals["oversold"] > signals["bearish"] + signals["overbought"]:
                if signals["bullish"] > signals["oversold"]:
                    trend = "strong_bullish" if signals["bullish"] >= 2 else "moderate_bullish"
                else:
                    trend = "potential_reversal_bullish"
            elif signals["bearish"] + signals["overbought"] > signals["bullish"] + signals["oversold"]:
                if signals["bearish"] > signals["overbought"]:
                    trend = "strong_bearish" if signals["bearish"] >= 2 else "moderate_bearish"
                else:
                    trend = "potential_reversal_bearish"
            else:
                trend = "neutral"
                
            # Calculate trend strength (0-100)
            bullish_strength = signals["bullish"] * 25 + signals["oversold"] * 15
            bearish_strength = signals["bearish"] * 25 + signals["overbought"] * 15
            
            if trend in ["strong_bullish", "moderate_bullish", "potential_reversal_bullish"]:
                trend_strength = bullish_strength
            elif trend in ["strong_bearish", "moderate_bearish", "potential_reversal_bearish"]:
                trend_strength = bearish_strength
            else:
                trend_strength = 50  # Neutral
                
            # Calculate price volatility
            if len(prices) > 20:
                recent_prices = prices[-20:]
                volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
            else:
                volatility = 5.0  # Default moderate volatility
            
            # Return all indicators and interpretations
            result = {
                "indicators": {
                    "rsi": rsi,
                    "macd": {
                        "macd_line": macd_line,
                        "signal_line": signal_line,
                        "histogram": histogram
                    },
                    "bollinger_bands": {
                        "upper": upper_band,
                        "middle": middle_band,
                        "lower": lower_band
                    },
                    "stochastic": {
                        "k": k,
                        "d": d
                    },
                    "obv": obv
                },
                "signals": {
                    "rsi": rsi_signal,
                    "macd": macd_signal,
                    "bollinger_bands": bb_signal,
                    "stochastic": stoch_signal
                },
                "overall_trend": trend,
                "trend_strength": trend_strength,
                "volatility": volatility,
                "timeframe": timeframe
            }
            
            # Add additional indicators for longer timeframes
            if timeframe in ["24h", "7d"]:
                result["indicators"].update(additional_indicators)
                result["signals"].update({
                    "adx": adx_signal,
                    "ichimoku": ichimoku_signal
                })
                
            return result
        except Exception as e:
            # Log detailed error
            logger.log_error("Technical Analysis", f"Error analyzing indicators: {str(e)}\n{traceback.format_exc()}")
            
            # Return fallback default result
            return {
                "overall_trend": "neutral",
                "trend_strength": 50,
                "volatility": 5.0,
                "timeframe": timeframe,
                "signals": {
                    "rsi": "neutral",
                    "macd": "neutral",
                    "bollinger_bands": "neutral",
                    "stochastic": "neutral"
                },
                "indicators": {
                    "rsi": 50,
                    "macd": {"macd_line": 0, "signal_line": 0, "histogram": 0},
                    "bollinger_bands": {"upper": 0, "middle": 0, "lower": 0},
                    "stochastic": {"k": 50, "d": 50},
                    "obv": 0
                },
                "error": str(e)
            }

class StatisticalModels:
    """Class for statistical forecasting models"""
    
    @staticmethod
    def arima_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        ARIMA forecasting model adjusted for different timeframes with robust error handling
        """
        try:
            # Validate inputs
            if not prices or len(prices) == 0:
                logger.logger.warning("ARIMA forecast received empty price list")
                return {
                    "forecast": [0.0] * forecast_steps,
                    "confidence_intervals": [{"95": [0.0, 0.0], "80": [0.0, 0.0]}] * forecast_steps,
                    "model_info": {"order": (0, 0, 0), "error": "No price data provided", "timeframe": timeframe}
                }
            
            # Adjust minimum data requirements based on timeframe
            min_data_points = 30
            if timeframe == "24h":
                min_data_points = 60  # Need more data for daily forecasts
            elif timeframe == "7d":
                min_data_points = 90  # Need even more data for weekly forecasts
                
            if len(prices) < min_data_points:
                logger.logger.warning(f"Insufficient data for ARIMA model with {timeframe} timeframe, using fallback")
                # Fall back to simpler model
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            # Adjust ARIMA parameters based on timeframe
            if timeframe == "1h":
                order = (5, 1, 0)  # Default for 1-hour
            elif timeframe == "24h":
                order = (5, 1, 1)  # Add MA component for daily
            else:  # 7d
                order = (7, 1, 1)  # More AR terms for weekly
            
            # Create and fit model
            model = ARIMA(prices, order=order)
            model_fit = model.fit()
            
            # Make forecast
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Calculate confidence intervals (simple approach)
            residuals = model_fit.resid
            resid_std = np.std(residuals)
            
            # Adjust confidence interval width based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20  # Wider for daily
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50  # Even wider for weekly
                ci_multiplier_80 = 1.80
                
            confidence_intervals = []
            for f in forecast:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * resid_std, f + ci_multiplier_95 * resid_std],
                    "80": [f - ci_multiplier_80 * resid_std, f + ci_multiplier_80 * resid_std]
                })
                
            return {
                "forecast": forecast.tolist(),
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "order": order,
                    "aic": model_fit.aic,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            # Log detailed error and use traceback for debugging
            error_msg = f"ARIMA Forecast ({timeframe}) error: {str(e)}"
            logger.log_error("ARIMA Forecast", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Return simple moving average forecast as fallback
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
    
    @staticmethod
    def moving_average_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h", window: int = None) -> Dict[str, Any]:
        """
        Simple moving average forecast with robust error handling (fallback method)
        Adjusted for different timeframes
        """
        try:
            # Validate inputs
            if not prices or len(prices) == 0:
                logger.logger.warning("Moving average forecast received empty price list")
                # Use default values for empty price list
                last_price = 0.0
                return {
                    "forecast": [last_price] * forecast_steps,
                    "confidence_intervals": [{
                        "95": [last_price * 0.95, last_price * 1.05],
                        "80": [last_price * 0.97, last_price * 1.03]
                    }] * forecast_steps,
                    "model_info": {
                        "method": "default_fallback",
                        "timeframe": timeframe
                    }
                }
                
            # Ensure we have at least one price
            last_price = prices[-1]
                
            # Set appropriate window size based on timeframe
            if window is None:
                if timeframe == "1h":
                    window = 5
                elif timeframe == "24h":
                    window = 7
                else:  # 7d
                    window = 4
            
            # Adjust window if we don't have enough data
            window = min(window, len(prices))
                
            if len(prices) < window or window <= 0:
                return {
                    "forecast": [last_price] * forecast_steps,
                    "confidence_intervals": [{
                        "95": [last_price * 0.95, last_price * 1.05],
                        "80": [last_price * 0.97, last_price * 1.03]
                    }] * forecast_steps,
                    "model_info": {
                        "method": "last_price_fallback",
                        "timeframe": timeframe
                    }
                }
                
            # Calculate moving average
            ma = sum(prices[-window:]) / window
            
            # Calculate standard deviation for confidence intervals
            std = np.std(prices[-window:])
            
            # Adjust confidence intervals based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50
                ci_multiplier_80 = 1.80
                
            # Generate forecast (all same value for MA)
            forecast = [ma] * forecast_steps
            
            # Generate confidence intervals
            confidence_intervals = []
            for _ in range(forecast_steps):
                confidence_intervals.append({
                    "95": [ma - ci_multiplier_95 * std, ma + ci_multiplier_95 * std],
                    "80": [ma - ci_multiplier_80 * std, ma + ci_multiplier_80 * std]
                })
                
            return {
                "forecast": forecast,
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "method": "moving_average",
                    "window": window,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            # Log detailed error
            error_msg = f"Moving Average Forecast ({timeframe}) error: {str(e)}"
            logger.log_error("Moving Average Forecast", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Fallback to last price
            if prices and len(prices) > 0:
                last_price = prices[-1]
            else:
                last_price = 0.0
                
            return {
                "forecast": [last_price] * forecast_steps,
                "confidence_intervals": [{
                    "95": [last_price * 0.95, last_price * 1.05],
                    "80": [last_price * 0.97, last_price * 1.03]
                }] * forecast_steps,
                "model_info": {
                    "method": "last_price_fallback",
                    "error": str(e),
                    "timeframe": timeframe
                }
            }
    
    @staticmethod
    def weighted_average_forecast(prices: List[float], volumes: List[float] = None, 
                                forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Volume-weighted average price forecast or linearly weighted forecast
        With robust error handling and adjusted for different timeframes
        """
        try:
            # Validate inputs
            if not prices or len(prices) == 0:
                if not volumes or len(volumes) == 0:
                    last_price = 0.0
                elif prices and len(prices) > 0:
                    last_price = prices[-1]
                else:
                    last_price = 0.0
                
                # Return fallback prediction    
                return {
                    "forecast": [last_price] * forecast_steps,
                    "confidence_intervals": [{
                        "95": [last_price * 0.95, last_price * 1.05],
                        "80": [last_price * 0.97, last_price * 1.03]
                    }] * forecast_steps,
                    "model_info": {
                        "method": "last_price_fallback",
                        "timeframe": timeframe
                    }
                }
            
            # Adjust window size based on timeframe
            if timeframe == "1h":
                window = 10
            elif timeframe == "24h":
                window = 14
            else:  # 7d
                window = 8
                
            # Adjust window if we don't have enough data
            window = min(window, len(prices))
                
            if len(prices) < window or window <= 0:
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            # If volumes available and match prices length, use volume-weighted average
            if volumes and len(volumes) == len(prices):
                # Get last window periods
                recent_prices = prices[-window:]
                recent_volumes = volumes[-window:] if len(volumes) >= window else [1.0] * window
                
                # Calculate VWAP, handling zero volumes gracefully
                total_volume = sum(recent_volumes)
                if total_volume > 0:
                    vwap = sum(p * v for p, v in zip(recent_prices, recent_volumes)) / total_volume
                else:
                    # Fall back to simple average if all volumes are zero
                    vwap = sum(recent_prices) / len(recent_prices)
                
                forecast = [vwap] * forecast_steps
                method = "volume_weighted"
            else:
                # Use linearly weighted average (more weight to recent prices)
                # Adjust weights based on timeframe for recency bias
                if timeframe == "1h":
                    weights = list(range(1, window + 1))  # Linear weights
                elif timeframe == "24h":
                    # Exponential weights for daily (more recency bias)
                    weights = [1.5 ** i for i in range(1, window + 1)]
                else:  # 7d
                    # Even more recency bias for weekly
                    weights = [2.0 ** i for i in range(1, window + 1)]
                    
                recent_prices = prices[-window:]
                
                # Calculate weighted average, handling empty weights
                sum_weights = sum(weights)
                if sum_weights > 0:
                    weighted_avg = sum(p * w for p, w in zip(recent_prices, weights)) / sum_weights
                else:
                    weighted_avg = sum(recent_prices) / len(recent_prices)
                
                forecast = [weighted_avg] * forecast_steps
                method = "weighted_average"
                
            # Calculate standard deviation for confidence intervals
            std = np.std(prices[-window:]) if len(prices) >= window else prices[-1] * 0.02
            
            # Adjust confidence intervals based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50
                ci_multiplier_80 = 1.80
                
            # Generate confidence intervals
            confidence_intervals = []
            for f in forecast:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * std, f + ci_multiplier_95 * std],
                    "80": [f - ci_multiplier_80 * std, f + ci_multiplier_80 * std]
                })
                
            return {
                "forecast": forecast,
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "method": method,
                    "window": window,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            # Log detailed error
            error_msg = f"Weighted Average Forecast ({timeframe}) error: {str(e)}"
            logger.log_error("Weighted Average Forecast", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Fall back to simpler method
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
        
    @staticmethod
    def holt_winters_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Holt-Winters exponential smoothing forecast with robust error handling
        Good for data with trend and seasonality
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Validate inputs
            if not prices or len(prices) == 0:
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
            
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 48,   # 2 days of hourly data
                "24h": 30,  # 1 month of daily data
                "7d": 16    # 4 months of weekly data
            }
            
            if len(prices) < min_data_points.get(timeframe, 48):
                return StatisticalModels.weighted_average_forecast(prices, None, forecast_steps, timeframe)
                
            # Determine seasonal_periods based on timeframe
            if timeframe == "1h":
                seasonal_periods = 24  # 24 hours in a day
            elif timeframe == "24h":
                seasonal_periods = 7   # 7 days in a week
            else:  # 7d
                seasonal_periods = 4   # 4 weeks in a month
            
            # Adjust seasonal_periods if we don't have enough data
            if len(prices) < 2 * seasonal_periods:
                # Fall back to non-seasonal model
                seasonal_periods = 1
                
            # Create and fit model with appropriate error handling
            try:
                model = ExponentialSmoothing(
                    prices, 
                    trend='add',
                    seasonal='add' if seasonal_periods > 1 else None, 
                    seasonal_periods=seasonal_periods if seasonal_periods > 1 else None,
                    use_boxcox=False  # Avoid potential errors with boxcox
                )
                model_fit = model.fit(optimized=True)
            except Exception as model_error:
                logger.logger.warning(f"Error fitting Holt-Winters model: {str(model_error)}. Trying simplified model.")
                # Try simpler model without seasonality
                try:
                    model = ExponentialSmoothing(prices, trend='add', seasonal=None)
                    model_fit = model.fit(optimized=True)
                except Exception as simple_error:
                    # If both fail, fall back to weighted average
                    logger.logger.warning(f"Error fitting simplified model: {str(simple_error)}. Using fallback.")
                    return StatisticalModels.weighted_average_forecast(prices, None, forecast_steps, timeframe)
            
            # Generate forecast
            forecast = model_fit.forecast(forecast_steps)
            
            # Calculate confidence intervals
            residuals = model_fit.resid
            resid_std = np.std(residuals)
            
            # Adjust confidence interval width based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50
                ci_multiplier_80 = 1.80
                
            confidence_intervals = []
            for f in forecast:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * resid_std, f + ci_multiplier_95 * resid_std],
                    "80": [f - ci_multiplier_80 * resid_std, f + ci_multiplier_80 * resid_std]
                })
                
            return {
                "forecast": forecast.tolist(),
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "method": "holt_winters",
                    "seasonal_periods": seasonal_periods,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            # Log detailed error
            error_msg = f"Holt-Winters Forecast ({timeframe}) error: {str(e)}"
            logger.log_error("Holt-Winters Forecast", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Fall back to weighted average forecast
            return StatisticalModels.weighted_average_forecast(prices, None, forecast_steps, timeframe)                    

class MachineLearningModels:
    """Class for machine learning forecasting models"""
    
    @staticmethod
    def create_features(prices: List[float], volumes: List[float] = None, timeframe: str = "1h") -> pd.DataFrame:
        """
        Create features for ML models from price and volume data
        With improved error handling and adjusted for different timeframes
        """
        try:
            # Validate inputs
            if not prices or len(prices) == 0:
                logger.logger.warning("Cannot create features: empty price data")
                return pd.DataFrame()
                
            # Adjust window sizes based on timeframe
            if timeframe == "1h":
                window_sizes = [5, 10, 20]
                max_lag = 6
            elif timeframe == "24h":
                window_sizes = [7, 14, 30]
                max_lag = 10
            else:  # 7d
                window_sizes = [4, 8, 12]
                max_lag = 8
            
            # Create base dataframe
            df = pd.DataFrame({'price': prices})
            
            # Add volume data if available
            if volumes and len(volumes) > 0:
                # Ensure volumes length matches prices
                vol_length = min(len(volumes), len(prices))
                df['volume'] = volumes[:vol_length]
                # If lengths don't match, fill remaining with last value or zeros
                if vol_length < len(prices):
                    df['volume'] = df['volume'].reindex(df.index, fill_value=volumes[-1] if volumes else 0)
            
            # Safely add lagged features
            try:
                # Adjust max_lag to prevent out-of-bounds errors
                max_lag = min(max_lag, len(prices) - 1)
                
                for lag in range(1, max_lag + 1):
                    df[f'price_lag_{lag}'] = df['price'].shift(lag)
            except Exception as lag_error:
                logger.logger.warning(f"Error creating lag features: {str(lag_error)}")
                
            # Safely add moving averages
            for window in window_sizes:
                # Skip windows larger than our data
                if window < len(prices):
                    try:
                        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
                    except Exception as ma_error:
                        logger.logger.warning(f"Error creating MA feature for window {window}: {str(ma_error)}")
                        
            # Safely add price momentum features
            for window in window_sizes:
                if window < len(prices) and f'ma_{window}' in df.columns:
                    try:
                        df[f'momentum_{window}'] = df['price'] - df[f'ma_{window}']
                    except Exception as momentum_error:
                        logger.logger.warning(f"Error creating momentum feature for window {window}: {str(momentum_error)}")
                        
            # Safely add relative price change
            for lag in range(1, max_lag + 1):
                if f'price_lag_{lag}' in df.columns:
                    try:
                        df[f'price_change_{lag}'] = (df['price'] / df[f'price_lag_{lag}'] - 1) * 100
                    except Exception as change_error:
                        logger.logger.warning(f"Error creating price change feature for lag {lag}: {str(change_error)}")
                        
            # Safely add volatility
            for window in window_sizes:
                if window < len(prices):
                    try:
                        df[f'volatility_{window}'] = df['price'].rolling(window=window).std()
                    except Exception as vol_error:
                        logger.logger.warning(f"Error creating volatility feature for window {window}: {str(vol_error)}")
                        
            # Safely add volume features if available
            if 'volume' in df.columns:
                for window in window_sizes:
                    if window < len(prices):
                        try:
                            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                            # Only create volume change if we have the moving average
                            if f'volume_ma_{window}' in df.columns:
                                df[f'volume_change_{window}'] = (df['volume'] / df[f'volume_ma_{window}'] - 1) * 100
                        except Exception as vol_feature_error:
                            logger.logger.warning(f"Error creating volume feature for window {window}: {str(vol_feature_error)}")
                            
            # Add timeframe-specific features
            try:
                if timeframe == "24h":
                    # Add day-of-week effect for daily data (if we have enough data)
                    if len(df) >= 7:
                        # Create day of week encoding (0-6, where 0 is Monday)
                        # This is a placeholder - in real implementation you would use actual dates
                        df['day_of_week'] = np.arange(len(df)) % 7
                        
                        # One-hot encode day of week
                        for day in range(7):
                            df[f'day_{day}'] = (df['day_of_week'] == day).astype(int)
                            
                elif timeframe == "7d":
                    # Add week-of-month or week-of-year features
                    if len(df) >= 4:
                        # Create week of month encoding (0-3)
                        # This is a placeholder - in real implementation you would use actual dates
                        df['week_of_month'] = np.arange(len(df)) % 4
                        
                        # One-hot encode week of month
                        for week in range(4):
                            df[f'week_{week}'] = (df['week_of_month'] == week).astype(int)
            except Exception as timeframe_features_error:
                logger.logger.warning(f"Error creating timeframe-specific features: {str(timeframe_features_error)}")
                        
            # Add additional technical indicators
            try:
                if len(prices) >= 14:
                    # RSI
                    delta = df['price'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    
                    rs = avg_gain / avg_loss
                    df['rsi_14'] = 100 - (100 / (1 + rs))
                    
                    # MACD components
                    ema_12 = df['price'].ewm(span=12, adjust=False).mean()
                    ema_26 = df['price'].ewm(span=26, adjust=False).mean()
                    df['macd'] = ema_12 - ema_26
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
            except Exception as tech_indicators_error:
                logger.logger.warning(f"Error creating technical indicators: {str(tech_indicators_error)}")
                
            # Drop NaN values
            df = df.dropna()
            
            return df
        except Exception as e:
            # Log detailed error
            error_msg = f"Feature Creation Error: {str(e)}"
            logger.log_error("ML Feature Creation", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Return empty DataFrame with price column at minimum
            return pd.DataFrame({'price': prices})
    
    @staticmethod
    def random_forest_forecast(prices: List[float], volumes: List[float] = None, 
                             forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Random Forest regression forecast with robust error handling
        Adjusted for different timeframes
        """
        try:
            # Validate inputs
            if not prices or len(prices) == 0:
                logger.logger.warning("Random Forest received empty price list")
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 48,   # 2 days of hourly data
                "24h": 30,  # 1 month of daily data
                "7d": 16    # 4 months of weekly data
            }
            
            if len(prices) < min_data_points.get(timeframe, 30):
                logger.logger.warning(f"Insufficient data for Random Forest model with {timeframe} timeframe")
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            # Create features with timeframe-specific settings
            df = MachineLearningModels.create_features(prices, volumes, timeframe)
            
            # Ensure we have enough features after preprocessing
            if len(df) < min_data_points.get(timeframe, 30) // 2:
                logger.logger.warning(f"Insufficient features after preprocessing for {timeframe} timeframe")
                return StatisticalModels.weighted_average_forecast(prices, volumes, forecast_steps, timeframe)
                
            # Prepare training data
            try:
                X = df.drop('price', axis=1)
                y = df['price']
                
                # Check if we have any features
                if X.empty:
                    logger.logger.warning("No features available for training")
                    return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                    
                # Create and train model with timeframe-specific parameters
                if timeframe == "1h":
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                elif timeframe == "24h":
                    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
                else:  # 7d
                    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
                    
                model.fit(X, y)
                
                # Prepare forecast data
                forecast_data = []
                last_known = df.iloc[-1:].copy()
                
                # Determine max lag based on features
                max_lag = 0
                for col in X.columns:
                    if col.startswith('price_lag_'):
                        lag_num = int(col.split('_')[-1])
                        max_lag = max(max_lag, lag_num)
                
                # Generate forecasts step by step
                for _ in range(forecast_steps):
                    try:
                        # Make prediction for next step
                        pred = model.predict(last_known.drop('price', axis=1))[0]
                        
                        # Update last_known for next step
                        new_row = last_known.copy()
                        new_row['price'] = pred
                        
                        # Update lags if they exist
                        for lag in range(max_lag, 0, -1):
                            lag_col = f'price_lag_{lag}'
                            if lag_col in new_row.columns:
                                if lag == 1:
                                    new_row[lag_col] = last_known['price'].values[0]
                                else:
                                    prev_lag_col = f'price_lag_{lag-1}'
                                    if prev_lag_col in last_known.columns:
                                        new_row[lag_col] = last_known[prev_lag_col].values[0]
                            
                        # Add prediction to results
                        forecast_data.append(pred)
                        
                        # Update last_known for next iteration
                        last_known = new_row
                    except Exception as step_error:
                        logger.logger.warning(f"Error in forecast step: {str(step_error)}")
                        # Fill with last prediction or price if error occurs
                        if forecast_data:
                            forecast_data.append(forecast_data[-1])
                        else:
                            forecast_data.append(prices[-1])
                
                # Calculate confidence intervals based on feature importance and model uncertainty
                feature_importance = model.feature_importances_.sum() if hasattr(model, 'feature_importances_') else 1.0
                
                # Higher importance = more confident = narrower intervals
                # Adjust confidence scale based on timeframe
                if timeframe == "1h":
                    base_confidence_scale = 1.0
                elif timeframe == "24h":
                    base_confidence_scale = 1.2  # Slightly less confident for daily
                else:  # 7d
                    base_confidence_scale = 1.5  # Even less confident for weekly
                    
                confidence_scale = max(0.5, min(2.0, base_confidence_scale / feature_importance))
                std = np.std(prices[-20:]) * confidence_scale
                
                # Adjust CI width based on timeframe
                if timeframe == "1h":
                    ci_multiplier_95 = 1.96
                    ci_multiplier_80 = 1.28
                elif timeframe == "24h":
                    ci_multiplier_95 = 2.20
                    ci_multiplier_80 = 1.50
                else:  # 7d
                    ci_multiplier_95 = 2.50
                    ci_multiplier_80 = 1.80
                
                confidence_intervals = []
                for f in forecast_data:
                    confidence_intervals.append({
                        "95": [f - ci_multiplier_95 * std, f + ci_multiplier_95 * std],
                        "80": [f - ci_multiplier_80 * std, f + ci_multiplier_80 * std]
                    })
                
                # Get feature importance for top features
                feature_importance_dict = {}
                if hasattr(model, 'feature_importances_'):
                    feature_importance_dict = dict(zip(X.columns, model.feature_importances_))
                    # Sort and limit to top 5
                    feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])
                    
                return {
                    "forecast": forecast_data,
                    "confidence_intervals": confidence_intervals,
                    "feature_importance": feature_importance_dict,
                    "model_info": {
                        "method": "random_forest",
                        "n_estimators": model.n_estimators,
                        "max_depth": model.max_depth,
                        "timeframe": timeframe
                    }
                }
            except Exception as model_error:
                # Log training/prediction error
                logger.log_error("Random Forest Model", str(model_error))
                logger.logger.debug(traceback.format_exc())
                # Fall back to statistical model
                return StatisticalModels.weighted_average_forecast(prices, volumes, forecast_steps, timeframe)
                
        except Exception as e:
            # Log detailed error
            error_msg = f"Random Forest Forecast ({timeframe}) error: {str(e)}"
            logger.log_error("Random Forest Forecast", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Fallback to moving average
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
    @staticmethod
    def lstm_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        LSTM neural network forecast for time series with robust error handling
        Returns forecast data in the same format as other prediction methods
        """
        try:
            # Import TensorFlow locally to avoid dependency issues if not available
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        
            # Check if prices array is sufficient for modeling
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 48,  # 2 days of hourly data
                "24h": 30,  # 1 month of daily data
                "7d": 16    # 4 months of weekly data
            }
        
            if len(prices) < min_data_points.get(timeframe, 48):
                logger.logger.warning(f"Insufficient data for LSTM model with {timeframe} timeframe: {len(prices)} points")
                # Fall back to RandomForest for insufficient data
                return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)
            
            # Prepare data for LSTM (with lookback window)
            # Adjust lookback based on timeframe
            if timeframe == "1h":
                lookback = 24  # 1 day
            elif timeframe == "24h":
                lookback = 14  # 2 weeks
            else:  # 7d
                lookback = 8   # 2 months
            
            # Make sure lookback is valid
            lookback = min(lookback, len(prices) // 2)
            if lookback < 3:
                lookback = 3  # Minimum lookback to avoid errors
        
            # Scale data (required for LSTM)
            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))
            except Exception as scale_error:
                logger.log_error(f"LSTM Data Scaling - {timeframe}", str(scale_error))
                return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)
        
            # Create dataset with lookback
            X, y = [], []
            try:
                for i in range(len(scaled_prices) - lookback):
                    X.append(scaled_prices[i:i+lookback, 0])
                    y.append(scaled_prices[i+lookback, 0])
                
                X, y = np.array(X), np.array(y)
                # Reshape for LSTM [samples, time steps, features]
                X = X.reshape(X.shape[0], X.shape[1], 1)
            except Exception as data_error:
                logger.log_error(f"LSTM Data Preparation - {timeframe}", str(data_error))
                return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)
        
            # Build and train LSTM model
            try:
                # Configure TensorFlow to avoid unnecessary warnings
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            
                # Clear previous Keras session
                tf.keras.backend.clear_session()
            
                # Create a simple but effective LSTM architecture
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
                model.add(Dropout(0.2))  # Add dropout to prevent overfitting
                model.add(LSTM(50))
                model.add(Dropout(0.2))
                model.add(Dense(1))
            
                # Compile model with appropriate loss and optimizer
                model.compile(optimizer='adam', loss='mean_squared_error')
            
                # Add early stopping to prevent overfitting
                early_stopping = EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True
                )
            
                # Train model with proper validation
                model.fit(
                    X, y,
                    epochs=50,
                    batch_size=32,
                    verbose=0,  # Silent training to avoid log spam
                    callbacks=[early_stopping],
                    validation_split=0.1  # Use 10% of data for validation
                )
            except Exception as train_error:
                logger.log_error(f"LSTM Training - {timeframe}", str(train_error))
                return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)
        
            # Generate predictions
            try:
                # Get the last window of observed prices
                last_window = scaled_prices[-lookback:].reshape(1, lookback, 1)
                forecast_scaled = []
            
                # Make the required number of forecast steps
                for _ in range(forecast_steps):
                    next_pred = model.predict(last_window, verbose=0)[0, 0]
                    forecast_scaled.append(next_pred)
                
                    # Update window by dropping oldest and adding newest prediction
                    next_pred_reshaped = np.array([[[next_pred]]])  # Shape: (1, 1, 1)
                    last_window = np.concatenate((last_window[:, 1:, :], next_pred_reshaped), axis=1)
                
                # Inverse transform to get actual price predictions
                forecast_data = scaler.inverse_transform(
                    np.array(forecast_scaled).reshape(-1, 1)
                ).flatten().tolist()
            
                # Calculate confidence intervals based on model's training error
                y_pred = model.predict(X, verbose=0).flatten()
                mse = np.mean((y - y_pred) ** 2)
            
                # Calculate prediction error in original scale
                price_range = max(prices) - min(prices)
                std_unscaled = np.sqrt(mse) * price_range
            
                # Adjust confidence intervals based on timeframe
                if timeframe == "1h":
                    ci_multiplier_95 = 1.96
                    ci_multiplier_80 = 1.28
                elif timeframe == "24h":
                    ci_multiplier_95 = 2.20
                    ci_multiplier_80 = 1.50
                else:  # 7d
                    ci_multiplier_95 = 2.50
                    ci_multiplier_80 = 1.80
                
                confidence_intervals = []
                for f in forecast_data:
                    confidence_intervals.append({
                        "95": [f - ci_multiplier_95 * std_unscaled, f + ci_multiplier_95 * std_unscaled],
                        "80": [f - ci_multiplier_80 * std_unscaled, f + ci_multiplier_80 * std_unscaled]
                    })
                
                # Clean up Keras/TF resources
                tf.keras.backend.clear_session()
            
                return {
                    "forecast": forecast_data,
                    "confidence_intervals": confidence_intervals,
                    "model_info": {
                        "method": "lstm",
                        "lookback": lookback,
                        "lstm_units": 50,
                        "timeframe": timeframe,
                        "epochs": 50
                    }
                }
            except Exception as pred_error:
                # Clean up resources even on error
                tf.keras.backend.clear_session()
            
                # Log and fall back to random forest
                logger.log_error(f"LSTM Prediction - {timeframe}", str(pred_error))
                return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)
            
        except ImportError as import_error:
            # Handle case where TensorFlow is not available
            logger.log_error("LSTM Import Error", str(import_error))
            logger.logger.warning("TensorFlow not available, falling back to Random Forest")
            return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)
        
        except Exception as e:
            # Log detailed error
            error_msg = f"LSTM Forecast ({timeframe}) error: {str(e)}"
            logger.log_error("LSTM Forecast", error_msg)
            logger.logger.debug(traceback.format_exc())
        
            # Fall back to random forest
            return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)

    @staticmethod
    def linear_regression_forecast(prices: List[float], volumes: List[float] = None, 
                                 forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Linear regression forecast with robust error handling
        Adjusted for different timeframes
        """
        try:
            # Validate inputs
            if not prices or len(prices) == 0:
                logger.logger.warning("Linear Regression received empty price list")
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 24,   # 1 day of hourly data
                "24h": 14,  # 2 weeks of daily data
                "7d": 8     # 2 months of weekly data
            }
            
            if len(prices) < min_data_points.get(timeframe, 20):
                logger.logger.warning(f"Insufficient data for Linear Regression model with {timeframe} timeframe")
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            # Create features with smaller window sizes for linear regression
            if timeframe == "1h":
                window_sizes = [3, 5, 10]
            elif timeframe == "24h":
                window_sizes = [3, 7, 14]
            else:  # 7d
                window_sizes = [2, 4, 8]
                
            # Create DataFrame for features
            try:
                df = pd.DataFrame({'price': prices})
                
                if volumes and len(volumes) > 0:
                    # Ensure volumes length matches prices
                    vol_length = min(len(volumes), len(prices))
                    df['volume'] = volumes[:vol_length]
                    # If lengths don't match, fill remaining with last value or zeros
                    if vol_length < len(prices):
                        df['volume'] = df['volume'].reindex(df.index, fill_value=volumes[-1] if volumes else 0)
                
                # Add lagged features (with safe max_lag)
                max_lag = 5
                max_lag = min(max_lag, len(prices) - 1)
                for lag in range(1, max_lag + 1):
                    df[f'price_lag_{lag}'] = df['price'].shift(lag)
                    
                # Add moving averages
                for window in window_sizes:
                    if window < len(prices):
                        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
                        
                # Add price momentum
                for window in window_sizes:
                    if window < len(prices) and f'ma_{window}' in df.columns:
                        df[f'momentum_{window}'] = df['price'] - df[f'ma_{window}']
                        
                # Add volume features if available
                if 'volume' in df.columns:
                    for window in window_sizes:
                        if window < len(prices):
                            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                            
                # Drop NaN values
                df = df.dropna()
            except Exception as feature_error:
                logger.log_error("Linear Regression Features", str(feature_error))
                # If feature creation fails, fall back to simpler model
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            if len(df) < 15:
                logger.logger.warning("Insufficient features after preprocessing")
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
            # Prepare training data
            try:
                X = df.drop('price', axis=1)
                y = df['price']
                
                # Create and train model
                model = LinearRegression()
                model.fit(X, y)
                
                # Prepare forecast data
                forecast_data = []
                last_known = df.iloc[-1:].copy()
                
                for _ in range(forecast_steps):
                    # Make prediction for next step
                    try:
                        pred = model.predict(last_known.drop('price', axis=1))[0]
                        
                        # Update last_known for next step
                        new_row = last_known.copy()
                        new_row['price'] = pred
                        
                        # Update lags
                        for lag in range(max_lag, 0, -1):
                            if lag == 1:
                                new_row[f'price_lag_{lag}'] = last_known['price'].values[0]
                            else:
                                new_row[f'price_lag_{lag}'] = last_known[f'price_lag_{lag-1}'].values[0]
                            
                        # Add prediction to results
                        forecast_data.append(pred)
                        
                        # Update last_known for next iteration
                        last_known = new_row
                    except Exception as step_error:
                        logger.logger.warning(f"Error in forecast step: {str(step_error)}")
                        # Fill with last prediction or price if error occurs
                        if forecast_data:
                            forecast_data.append(forecast_data[-1])
                        else:
                            forecast_data.append(prices[-1])
                
                # Calculate confidence intervals based on model's prediction error
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                std = np.sqrt(mse)
                
                # Adjust confidence intervals based on timeframe
                if timeframe == "1h":
                    ci_multiplier_95 = 1.96
                    ci_multiplier_80 = 1.28
                elif timeframe == "24h":
                    ci_multiplier_95 = 2.20
                    ci_multiplier_80 = 1.50
                else:  # 7d
                    ci_multiplier_95 = 2.50
                    ci_multiplier_80 = 1.80
                    
                confidence_intervals = []
                for f in forecast_data:
                    confidence_intervals.append({
                        "95": [f - ci_multiplier_95 * std, f + ci_multiplier_95 * std],
                        "80": [f - ci_multiplier_80 * std, f + ci_multiplier_80 * std]
                    })
                    
                # Get top coefficients
                coefficients = {}
                if hasattr(model, 'coef_'):
                    coefficients = dict(zip(X.columns, model.coef_))
                    # Sort and limit to top 5
                    coefficients = dict(sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
                    
                return {
                    "forecast": forecast_data,
                    "confidence_intervals": confidence_intervals,
                    "coefficients": coefficients,
                    "model_info": {
                        "method": "linear_regression",
                        "r2_score": model.score(X, y) if hasattr(model, 'score') else 0,
                        "timeframe": timeframe
                    }
                }
            except Exception as model_error:
                # Log model training/prediction error
                logger.log_error("Linear Regression Model", str(model_error))
                # Fall back to statistical model
                return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
                
        except Exception as e:
            # Log detailed error
            error_msg = f"Linear Regression Forecast ({timeframe}) error: {str(e)}"
            logger.log_error("Linear Regression Forecast", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Fallback to moving average
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
               

class ClaudeEnhancedPrediction:
    """Class for generating Claude AI enhanced predictions"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """Initialize Claude client with error handling"""
        try:
            self.client = anthropic.Client(api_key=api_key)
            self.model = model
            logger.logger.info(f"Claude Enhanced Prediction initialized with model: {model}")
        except Exception as e:
            logger.log_error("Claude Client Initialization", str(e))
            self.client = None
            self.model = model
            logger.logger.warning("Failed to initialize Claude client, will fall back to basic predictions")
        
    def generate_enhanced_prediction(self, 
                                     token: str, 
                                     current_price: float,
                                     technical_analysis: Dict[str, Any],
                                     statistical_forecast: Dict[str, Any],
                                     ml_forecast: Dict[str, Any],
                                     timeframe: str = "1h",
                                     price_history_24h: List[Dict[str, Any]] = None,
                                     market_conditions: Dict[str, Any] = None,
                                     recent_predictions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an enhanced prediction using Claude AI
        Improved error handling and fallback mechanisms
        Supports 1h, 24h, and 7d timeframes
        """
        try:
            # Check if client is available
            if not self.client:
                logger.logger.warning("Claude client unavailable, falling back to combined prediction")
                return self._generate_fallback_prediction(
                    token, current_price, technical_analysis, 
                    statistical_forecast, ml_forecast, timeframe
                )
            
            # Validate inputs
            if current_price <= 0:
                logger.logger.warning(f"Invalid current price for {token}: {current_price}")
                current_price = 1.0  # Safe default
                
            # Extract key info from technical analysis with error handling
            tech_signals = technical_analysis.get("signals", {})
            overall_trend = technical_analysis.get("overall_trend", "neutral")
            trend_strength = technical_analysis.get("trend_strength", 50)
            
            # Format forecasts with error handling
            try:
                # Extract statistical forecast
                stat_forecast_values = statistical_forecast.get("forecast", [current_price])
                stat_forecast = stat_forecast_values[0] if stat_forecast_values else current_price
                
                stat_confidence = statistical_forecast.get("confidence_intervals", [])
                if not stat_confidence:
                    stat_confidence = [{"80": [current_price*0.98, current_price*1.02]}]
                
                # Extract ML forecast
                ml_forecast_values = ml_forecast.get("forecast", [current_price])
                ml_forecast_val = ml_forecast_values[0] if ml_forecast_values else current_price
                
                ml_confidence = ml_forecast.get("confidence_intervals", [])
                if not ml_confidence:
                    ml_confidence = [{"80": [current_price*0.98, current_price*1.02]}]
                
                # Calculate average forecast
                avg_forecast = (stat_forecast + ml_forecast_val) / 2
            except Exception as forecast_error:
                logger.log_error("Claude Forecast Extraction", str(forecast_error))
                # Use current price as fallback
                stat_forecast = current_price
                ml_forecast_val = current_price
                avg_forecast = current_price
                stat_confidence = [{"80": [current_price*0.98, current_price*1.02]}]
                ml_confidence = [{"80": [current_price*0.98, current_price*1.02]}]
            
            # Prepare historical context
            historical_context = ""
            try:
                if price_history_24h:
                    # Get min and max prices over 24h
                    prices = [entry.get("price", 0) for entry in price_history_24h]
                    volumes = [entry.get("volume", 0) for entry in price_history_24h]
                    
                    if prices:
                        min_price = min(prices)
                        max_price = max(prices)
                        avg_price = sum(prices) / len(prices)
                        total_volume = sum(volumes)
                        
                        # Adjust display based on timeframe
                        if timeframe == "1h":
                            period_desc = "24-Hour"
                        elif timeframe == "24h":
                            period_desc = "7-Day"
                        else:  # 7d
                            period_desc = "30-Day"
                            
                        historical_context = f"""
{period_desc} Price Data:
- Current: ${current_price}
- Average: ${avg_price}
- High: ${max_price}
- Low: ${min_price}
- Range: ${max_price - min_price} ({((max_price - min_price) / min_price) * 100:.2f}%)
- Total Volume: ${total_volume}
"""
            except Exception as history_error:
                logger.log_error("Claude Historical Context", str(history_error))
                historical_context = ""
            
            # Market conditions context
            market_context = ""
            try:
                if market_conditions:
                    market_context = f"""
Market Conditions:
- Overall market trend: {market_conditions.get('market_trend', 'unknown')}
- BTC dominance: {market_conditions.get('btc_dominance', 'unknown')}
- Market volatility: {market_conditions.get('market_volatility', 'unknown')}
- Sector performance: {market_conditions.get('sector_performance', 'unknown')}
"""
            except Exception as market_error:
                logger.log_error("Claude Market Context", str(market_error))
                market_context = ""
                
            # Accuracy context
            accuracy_context = ""
            try:
                if recent_predictions:
                    correct_predictions = [p for p in recent_predictions if p.get("was_correct")]
                    accuracy_rate = len(correct_predictions) / len(recent_predictions) if recent_predictions else 0
                    
                    accuracy_context = f"""
Recent Prediction Performance:
- Accuracy rate for {timeframe} predictions: {accuracy_rate * 100:.1f}%
- Total predictions: {len(recent_predictions)}
- Correct predictions: {len(correct_predictions)}
"""
            except Exception as accuracy_error:
                logger.log_error("Claude Accuracy Context", str(accuracy_error))
                accuracy_context = ""
                
            # Get additional technical indicators for longer timeframes
            additional_indicators = ""
            try:
                if timeframe in ["24h", "7d"] and "indicators" in technical_analysis:
                    indicators = technical_analysis["indicators"]
                    
                    # Add ADX if available
                    if "adx" in indicators:
                        additional_indicators += f"- ADX: {indicators['adx']:.2f}\n"
                        
                    # Add Ichimoku Cloud if available
                    if "ichimoku" in indicators:
                        ichimoku = indicators["ichimoku"]
                        additional_indicators += "- Ichimoku Cloud:\n"
                        additional_indicators += f"  - Tenkan-sen: {ichimoku['tenkan_sen']:.4f}\n"
                        additional_indicators += f"  - Kijun-sen: {ichimoku['kijun_sen']:.4f}\n"
                        additional_indicators += f"  - Senkou Span A: {ichimoku['senkou_span_a']:.4f}\n"
                        additional_indicators += f"  - Senkou Span B: {ichimoku['senkou_span_b']:.4f}\n"
                        
                    # Add Pivot Points if available
                    if "pivot_points" in indicators:
                        pivots = indicators["pivot_points"]
                        additional_indicators += "- Pivot Points:\n"
                        additional_indicators += f"  - Pivot: {pivots['pivot']:.4f}\n"
                        additional_indicators += f"  - R1: {pivots['r1']:.4f}, R2: {pivots['r2']:.4f}\n"
                        additional_indicators += f"  - S1: {pivots['s1']:.4f}, S2: {pivots['s2']:.4f}\n"
            except Exception as indicators_error:
                logger.log_error("Claude Additional Indicators", str(indicators_error))
                additional_indicators = ""
            
            # Calculate optimal confidence interval for FOMO generation
            try:
                # Get volatility - default to moderate if not available
                current_volatility = technical_analysis.get("volatility", 5.0)
                
                # Scale confidence interval based on volatility, trend strength, and timeframe
                # Higher volatility = wider interval
                # Stronger trend = narrower interval (more confident)
                # Longer timeframe = wider interval
                volatility_factor = min(1.5, max(0.5, current_volatility / 10))
                trend_factor = max(0.7, min(1.3, 1.2 - (trend_strength / 100)))
                
                # Timeframe factor - wider intervals for longer timeframes
                if timeframe == "1h":
                    timeframe_factor = 1.0
                elif timeframe == "24h":
                    timeframe_factor = 1.5
                else:  # 7d
                    timeframe_factor = 2.0
                    
                # Calculate confidence bounds
                bound_factor = volatility_factor * trend_factor * timeframe_factor
                lower_bound = avg_forecast * (1 - 0.015 * bound_factor)
                upper_bound = avg_forecast * (1 + 0.015 * bound_factor)
                
                # Ensure bounds are narrow enough to create FOMO but realistic for the timeframe
                price_range_pct = (upper_bound - lower_bound) / current_price * 100
                
                # Adjust max range based on timeframe
                max_range_pct = {
                    "1h": 3.0,   # 3% for 1 hour
                    "24h": 8.0,  # 8% for 24 hours
                    "7d": 15.0   # 15% for 7 days
                }.get(timeframe, 3.0)
                
                if price_range_pct > max_range_pct:
                    # Too wide - recalculate to create FOMO
                    center = (upper_bound + lower_bound) / 2
                    margin = (current_price * max_range_pct / 200)  # half of max_range_pct
                    upper_bound = center + margin
                    lower_bound = center - margin
            except Exception as bounds_error:
                logger.log_error("Claude Bounds Calculation", str(bounds_error))
                # Fallback to simple percentage bounds
                if timeframe == "1h":
                    margin = 0.015  # 1.5%
                elif timeframe == "24h":
                    margin = 0.04   # 4%
                else:  # 7d
                    margin = 0.075  # 7.5%
                lower_bound = avg_forecast * (1 - margin)
                upper_bound = avg_forecast * (1 + margin)
                
            # Timeframe-specific guidance for FOMO generation
            fomo_guidance = {
                "1h": "Focus on immediate catalysts and short-term technical breakouts for this 1-hour prediction.",
                "24h": "Emphasize day-trading patterns and 24-hour potential for this daily prediction.",
                "7d": "Highlight medium-term trend confirmation and key weekly support/resistance levels."
            }.get(timeframe, "")
            
            # Prepare the prompt for Claude with timeframe-specific adjustments
            prompt = f"""
You are a sophisticated crypto market prediction expert. I need your analysis to make a precise {timeframe} prediction for {token}.

## Technical Analysis
- RSI Signal: {tech_signals.get('rsi', 'neutral')}
- MACD Signal: {tech_signals.get('macd', 'neutral')}
- Bollinger Bands: {tech_signals.get('bollinger_bands', 'neutral')}
- Stochastic Oscillator: {tech_signals.get('stochastic', 'neutral')}
- Overall Trend: {overall_trend}
- Trend Strength: {trend_strength}/100
{additional_indicators}

## Statistical Models
- Forecast: ${stat_forecast:.4f}
- 80% Confidence: [${stat_confidence[0]['80'][0]:.4f}, ${stat_confidence[0]['80'][1]:.4f}]

## Machine Learning Models
- ML Forecast: ${ml_forecast_val:.4f}
- 80% Confidence: [${ml_confidence[0]['80'][0]:.4f}, ${ml_confidence[0]['80'][1]:.4f}]

## Current Market Data
- Current Price: ${current_price}
- Predicted Range: [${lower_bound:.4f}, ${upper_bound:.4f}]

{historical_context}
{market_context}
{accuracy_context}

## Prediction Task
1. Predict the EXACT price of {token} in {timeframe} with a confidence level between 65-85%.
2. Provide a narrow price range to create FOMO, but ensure it's realistic given the data and {timeframe} timeframe.
3. State the percentage change you expect.
4. Give a concise rationale (2-3 sentences maximum).
5. Assign a sentiment: BULLISH, BEARISH, or NEUTRAL.

{fomo_guidance}

Your prediction must follow this EXACT JSON format:
{{
  "prediction": {{
    "price": [exact price prediction],
    "confidence": [confidence percentage],
    "lower_bound": [lower price bound],
    "upper_bound": [upper price bound],
    "percent_change": [expected percentage change],
    "timeframe": "{timeframe}"
  }},
  "rationale": [brief explanation],
  "sentiment": [BULLISH/BEARISH/NEUTRAL],
  "key_factors": [list of 2-3 main factors influencing this prediction]
}}

Your prediction should be precise, data-driven, and conservative enough to be accurate while narrow enough to generate excitement.
IMPORTANT: Provide ONLY the JSON response, no additional text.
"""
            
            # Make the API call to Claude
            logger.logger.debug(f"Requesting Claude {timeframe} prediction for {token}")
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Process the response
                result_text = response.content[0].text.strip()
                logger.logger.debug(f"Claude raw response: {result_text[:200]}...")
            except Exception as api_error:
                logger.log_error("Claude API Call", str(api_error))
                # Return fallback prediction if API call fails
                return self._generate_fallback_prediction(
                    token, current_price, technical_analysis, 
                    statistical_forecast, ml_forecast, timeframe
                )
            
            # Extract JSON from Claude's response
            try:
                # Clean up the response text
                result_text = result_text.replace("```json", "").replace("```", "").strip()
                
                # Parse JSON response
                result = json.loads(result_text)
                
                # Validate the structure of the response
                if not isinstance(result, dict):
                    raise ValueError("Claude didn't return a dictionary response")
                
                if "prediction" not in result:
                    raise ValueError("Response missing 'prediction' field")
                    
                # Check if prediction contains all required fields
                required_fields = ["price", "confidence", "lower_bound", "upper_bound", "percent_change", "timeframe"]
                for field in required_fields:
                    if field not in result["prediction"]:
                        raise ValueError(f"Prediction missing '{field}' field")
                        
                # Validate that prediction values are numeric
                for field in ["price", "confidence", "lower_bound", "upper_bound", "percent_change"]:
                    val = result["prediction"][field]
                    if not isinstance(val, (int, float)):
                        raise ValueError(f"Field '{field}' is not numeric: {val}")
                
                # Verify other required fields
                if "rationale" not in result:
                    result["rationale"] = f"Based on technical analysis for {token} over {timeframe}."
                    
                if "sentiment" not in result:
                    # Derive sentiment from percent change
                    pct_change = result["prediction"]["percent_change"]
                    if pct_change > 1.0:
                        result["sentiment"] = "BULLISH"
                    elif pct_change < -1.0:
                        result["sentiment"] = "BEARISH"
                    else:
                        result["sentiment"] = "NEUTRAL"
                
                if "key_factors" not in result or not isinstance(result["key_factors"], list):
                    result["key_factors"] = ["Technical analysis", "Market conditions", "Price momentum"]
            except Exception as json_error:
                logger.log_error("Claude JSON Processing", str(json_error))
                logger.logger.warning(f"Failed to parse Claude response: {result_text[:100]}...")
                
                # Try to extract a price prediction from the text if it's a simple float
                try:
                    # If Claude returned just a number, use it as the price
                    if isinstance(result_text, str) and result_text.strip().replace('.', '', 1).isdigit():
                        predicted_price = float(result_text.strip())
                        
                        # Create proper JSON structure
                        result = {
                            "prediction": {
                                "price": predicted_price,
                                "confidence": 70.0,
                                "lower_bound": lower_bound,
                                "upper_bound": upper_bound,
                                "percent_change": ((predicted_price / current_price) - 1) * 100,
                                "timeframe": timeframe
                            },
                            "rationale": f"Technical indicators and market patterns suggest this price target for {token} in the next {timeframe}.",
                            "sentiment": "BULLISH" if predicted_price > current_price else "BEARISH" if predicted_price < current_price else "NEUTRAL",
                            "key_factors": ["Technical analysis", "Market trends", "Price momentum"]
                        }
                    else:
                        # Fall back to generated prediction
                        raise ValueError("Could not extract price from Claude response")
                except:
                    # Return fallback prediction
                    return self._generate_fallback_prediction(
                        token, current_price, technical_analysis, 
                        statistical_forecast, ml_forecast, timeframe
                    )
            
            # Ensure the prediction is within reasonable bounds
            try:
                # Get prediction price 
                pred_price = result["prediction"]["price"]
                
                # Check for unreasonable predictions (more than 50% change)
                if abs((pred_price / current_price) - 1) > 0.5:
                    logger.logger.warning(f"Claude predicted unreasonable price change for {token}: {pred_price} (current: {current_price})")
                    # Adjust to a more reasonable prediction
                    if pred_price > current_price:
                        result["prediction"]["price"] = current_price * 1.05  # 5% increase
                    else:
                        result["prediction"]["price"] = current_price * 0.95  # 5% decrease
                    
                    # Update percent change
                    result["prediction"]["percent_change"] = ((result["prediction"]["price"] / current_price) - 1) * 100
            except Exception as price_check_error:
                logger.log_error("Claude Price Validation", str(price_check_error))
                # Just continue with the existing prediction
            
            # Add the model weightings that produced this prediction
            result["model_weights"] = {
                "technical_analysis": 0.25,
                "statistical_models": 0.25,
                "machine_learning": 0.25,
                "claude_enhanced": 0.25
            }
            
            # Add the inputs that generated this prediction
            result["inputs"] = {
                "current_price": current_price,
                "technical_analysis": {
                    "overall_trend": overall_trend,
                    "trend_strength": trend_strength,
                    "signals": tech_signals
                },
                "statistical_forecast": {
                    "prediction": stat_forecast,
                    "confidence": stat_confidence[0]['80']
                },
                "ml_forecast": {
                    "prediction": ml_forecast_val,
                    "confidence": ml_confidence[0]['80']
                },
                "timeframe": timeframe
            }
            
            logger.logger.debug(f"Claude {timeframe} prediction generated for {token}: {result['prediction']['price']}")
            return result
            
        except Exception as e:
            # Provide detailed logging for the exception
            error_msg = f"Claude Enhanced Prediction ({timeframe}) failed: {str(e)}"
            logger.log_error("Claude Enhanced Prediction", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Generate fallback prediction
            return self._generate_fallback_prediction(
                token, current_price, technical_analysis, 
                statistical_forecast, ml_forecast, timeframe
            )
    
    def _generate_fallback_prediction(self, 
                                     token: str, 
                                     current_price: float,
                                     technical_analysis: Dict[str, Any] = None,
                                     statistical_forecast: Dict[str, Any] = None,
                                     ml_forecast: Dict[str, Any] = None,
                                     timeframe: str = "1h") -> Dict[str, Any]:
        """
        Generate a fallback prediction when Claude API fails
        Uses a weighted combination of technical, statistical, and ML models
        """
        try:
            # Extract technical analysis data
            if technical_analysis:
                trend = technical_analysis.get("overall_trend", "neutral")
                trend_strength = technical_analysis.get("trend_strength", 50)
                tech_signals = technical_analysis.get("signals", {})
                volatility = technical_analysis.get("volatility", 5.0)
            else:
                trend = "neutral"
                trend_strength = 50
                tech_signals = {}
                volatility = 5.0
            
            # Extract statistical and ML forecasts
            stat_forecast_val = None
            ml_forecast_val = None
            
            if statistical_forecast and "forecast" in statistical_forecast:
                forecasts = statistical_forecast["forecast"]
                if forecasts and len(forecasts) > 0:
                    stat_forecast_val = forecasts[0]
            
            if ml_forecast and "forecast" in ml_forecast:
                forecasts = ml_forecast["forecast"]
                if forecasts and len(forecasts) > 0:
                    ml_forecast_val = forecasts[0]
            
            # Default to current price if forecasts not available
            if stat_forecast_val is None:
                stat_forecast_val = current_price
            if ml_forecast_val is None:
                ml_forecast_val = current_price
                
            # Determine weights based on trend strength and timeframe
            # If trend is strong, give more weight to technical analysis
            # For longer timeframes, rely more on statistical and ML models
            if timeframe == "1h":
                # For hourly, technical indicators matter more
                if trend_strength > 70:
                    tech_weight = 0.5
                    stat_weight = 0.25
                    ml_weight = 0.25
                elif trend_strength > 50:
                    tech_weight = 0.4
                    stat_weight = 0.3
                    ml_weight = 0.3
                else:
                    tech_weight = 0.2
                    stat_weight = 0.4
                    ml_weight = 0.4
            elif timeframe == "24h":
                # For daily, balance between technical and models
                if trend_strength > 70:
                    tech_weight = 0.4
                    stat_weight = 0.3
                    ml_weight = 0.3
                elif trend_strength > 50:
                    tech_weight = 0.35
                    stat_weight = 0.35
                    ml_weight = 0.3
                else:
                    tech_weight = 0.25
                    stat_weight = 0.4
                    ml_weight = 0.35
            else:  # 7d
                # For weekly, models matter more than short-term indicators
                if trend_strength > 70:
                    tech_weight = 0.35
                    stat_weight = 0.35
                    ml_weight = 0.3
                elif trend_strength > 50:
                    tech_weight = 0.3
                    stat_weight = 0.4
                    ml_weight = 0.3
                else:
                    tech_weight = 0.2
                    stat_weight = 0.45
                    ml_weight = 0.35
                
            # Calculate technical prediction based on trend
            if "bullish" in trend:
                tech_prediction = current_price * (1 + (0.01 * (trend_strength - 50) / 50))
            elif "bearish" in trend:
                tech_prediction = current_price * (1 - (0.01 * (trend_strength - 50) / 50))
            else:
                tech_prediction = current_price
                
            # Calculate weighted prediction
            weighted_prediction = (
                tech_weight * tech_prediction + 
                stat_weight * stat_forecast_val + 
                ml_weight * ml_forecast_val
            )
            
            # Calculate confidence level
            # Higher trend strength = higher confidence
            # Longer timeframe = lower confidence
            base_confidence = {
                "1h": 65,
                "24h": 60,
                "7d": 55
            }.get(timeframe, 65)
            
            confidence_boost = min(20, (trend_strength - 50) * 0.4) if trend_strength > 50 else 0
            confidence = base_confidence + confidence_boost
            
            # Calculate price range based on timeframe
            # Wider ranges for longer timeframes
            base_range_factor = {
                "1h": 0.005,
                "24h": 0.015,
                "7d": 0.025
            }.get(timeframe, 0.005)
            
            range_factor = max(base_range_factor, min(base_range_factor * 4, volatility / 200))
            
            lower_bound = weighted_prediction * (1 - range_factor)
            upper_bound = weighted_prediction * (1 + range_factor)
            
            # Calculate percentage change
            percent_change = ((weighted_prediction / current_price) - 1) * 100
            
            # Determine sentiment
            # Adjust thresholds based on timeframe
            sentiment_thresholds = {
                "1h": 1.0,    # 1% for hourly
                "24h": 2.5,   # 2.5% for daily
                "7d": 5.0     # 5% for weekly
            }.get(timeframe, 1.0)
            
            if percent_change > sentiment_thresholds:
                sentiment = "BULLISH"
            elif percent_change < -sentiment_thresholds:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
                
            # Generate rationale based on technical analysis and market conditions
            timeframe_desc = {
                "1h": "hour",
                "24h": "24 hours",
                "7d": "week"
            }.get(timeframe, timeframe)
            
            if sentiment == "BULLISH":
                rationale = f"Strong {trend} with confluence from statistical models suggest upward momentum in the next {timeframe_desc}."
            elif sentiment == "BEARISH":
                rationale = f"{trend.capitalize()} confirmed by multiple indicators, suggesting continued downward pressure over the next {timeframe_desc}."
            else:
                rationale = f"Mixed signals with {trend} but limited directional conviction for the {timeframe_desc} ahead."
                
            # Identify key factors
            key_factors = []
            
            # Add technical factor
            strongest_signal = None
            strongest_value = "neutral"
            
            # Find the strongest non-neutral signal
            for key, value in tech_signals.items():
                if value != "neutral" and (strongest_signal is None or 
                                          (value in ["bullish", "bearish"] and strongest_value not in ["bullish", "bearish"])):
                    strongest_signal = key
                    strongest_value = value
            
            if strongest_signal:
                key_factors.append(f"{strongest_signal.upper()} {strongest_value}")
            else:
                key_factors.append("Technical indicators neutral")
            
            # Add statistical factor
            key_factors.append(f"Statistical models: {'+' if stat_forecast_val > current_price else ''}{((stat_forecast_val / current_price) - 1) * 100:.1f}%")
            
            # Add market factor based on trend
            if "bullish" in trend:
                key_factors.append("Positive price momentum")
            elif "bearish" in trend:
                key_factors.append("Negative price pressure")
            else:
                key_factors.append("Consolidating market")
            
            return {
                "prediction": {
                    "price": weighted_prediction,
                    "confidence": confidence,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "percent_change": percent_change,
                    "timeframe": timeframe
                },
                "rationale": rationale,
                "sentiment": sentiment,
                "key_factors": key_factors,
                "model_weights": {
                    "technical_analysis": tech_weight,
                    "statistical_models": stat_weight,
                    "machine_learning": ml_weight,
                    "claude_enhanced": 0.0
                },
                "is_fallback": True  # Flag to indicate this is a fallback prediction
            }
        except Exception as e:
            # Log detailed error
            error_msg = f"Fallback Prediction Error: {str(e)}"
            logger.log_error("Fallback Prediction", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Ultra-fallback with minimal calculation
            if current_price <= 0:
                current_price = 1.0  # Default to 1.0 to avoid division by zero
                
            # Simple 1% prediction based on timeframe direction
            if timeframe == "1h":
                change_pct = 1.0
            elif timeframe == "24h":
                change_pct = 2.0
            else:  # 7d
                change_pct = 3.0
                
            # Default bullish prediction
            predicted_price = current_price * (1 + change_pct/100)
            
            return {
                "prediction": {
                    "price": predicted_price,
                    "confidence": 60,
                    "lower_bound": current_price * 0.99,
                    "upper_bound": current_price * 1.01,
                    "percent_change": change_pct,
                    "timeframe": timeframe
                },
                "rationale": f"Market analysis suggests minor upward movement for {token}.",
                "sentiment": "NEUTRAL",
                "key_factors": ["Market conditions", "Technical analysis"],
                "model_weights": {
                    "technical_analysis": 0.6,
                    "statistical_models": 0.3,
                    "machine_learning": 0.1,
                    "claude_enhanced": 0.0
                },
                "is_emergency_fallback": True  # Flag to indicate this is an emergency fallback prediction
            }

class PredictionEngine:
    """Main prediction engine class that combines all approaches with focus on reply functionality"""
    
    def __init__(self, database, llm_provider=None, claude_api_key=None):
        """Initialize the prediction engine with robust error handling"""
        try:
            self.db = database
            self.client_model = "claude-3-haiku-20240307"
            
            # Use either the provided llm_provider or create one from the API key
            if llm_provider:
                self.llm_provider = llm_provider
            elif claude_api_key:
                # For backward compatibility
                from llm_provider import LLMProvider
                config = type('Config', (), {'CLAUDE_API_KEY': claude_api_key})
                self.llm_provider = LLMProvider(config)
            else:
                self.llm_provider = None
                
            # Track pending predictions to avoid duplicate processing
            self.pending_predictions = set()
            
            # Track prediction errors to prevent repeated failures
            self.error_cooldowns = {}
            
            # Initialize reply-focused settings
            self.reply_ready_predictions = {}
            self.max_cached_predictions = 100
            
            logger.logger.info("Prediction Engine initialized successfully")
        except Exception as e:
            logger.log_error("Prediction Engine Initialization", str(e))
            logger.logger.error(f"Failed to initialize prediction engine: {str(e)}")
            self.db = database
            self.client = None
            self.pending_predictions = set()
            self.error_cooldowns = {}
            self.reply_ready_predictions = {}
            
    def generate_prediction(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Generate a comprehensive prediction for a token
        Enhanced with improved error handling and reply focus
        Supports 1h, 24h, and 7d timeframes
        """
        prediction_id = f"{token}_{timeframe}_{int(time.time())}"
        
        try:
            # Check if we're already processing this token/timeframe combo
            if f"{token}_{timeframe}" in self.pending_predictions:
                logger.logger.info(f"Already processing {token} ({timeframe}) prediction, skipping")
                cached_prediction = self._get_cached_prediction(token, timeframe)
                if cached_prediction:
                    return cached_prediction
                    
            # Check error cooldown to avoid repeatedly hitting same error
            cooldown_key = f"{token}_{timeframe}"
            if cooldown_key in self.error_cooldowns:
                last_error, cooldown_until = self.error_cooldowns[cooldown_key]
                if time.time() < cooldown_until:
                    logger.logger.warning(f"Prediction for {token} ({timeframe}) in error cooldown: {last_error}")
                    # Return cached prediction or generate minimal fallback
                    cached_prediction = self._get_cached_prediction(token, timeframe)
                    if cached_prediction:
                        return cached_prediction
                    return self._generate_fallback_prediction(token, market_data.get(token, {}), timeframe)
            
            # Mark as pending to prevent duplicate processing
            self.pending_predictions.add(f"{token}_{timeframe}")
            
            # Validate timeframe
            if timeframe not in ["1h", "24h", "7d"]:
                logger.logger.warning(f"Invalid timeframe: {timeframe}. Using 1h as default.")
                timeframe = "1h"
                
            # Extract token data
            token_data = market_data.get(token, {})
            if not token_data:
                logger.logger.warning(f"No market data available for {token}")
                self.pending_predictions.discard(f"{token}_{timeframe}")
                self.error_cooldowns[cooldown_key] = ("No market data available", time.time() + 600)  # 10 minute cooldown
                return self._generate_fallback_prediction(token, {}, timeframe)
                
            current_price = token_data.get('current_price', 0)
            if current_price <= 0:
                logger.logger.warning(f"Invalid price data for {token}: {current_price}")
                self.pending_predictions.discard(f"{token}_{timeframe}")
                self.error_cooldowns[cooldown_key] = ("Invalid price data", time.time() + 600)  # 10 minute cooldown
                return self._generate_fallback_prediction(token, token_data, timeframe)
            
            # Get historical price data from database
            # Adjust the hours parameter based on timeframe
            if timeframe == "1h":
                historical_hours = 48  # 2 days of data for 1h predictions
            elif timeframe == "24h":
                historical_hours = 168  # 7 days of data for 24h predictions
            else:  # 7d
                historical_hours = 720  # 30 days of data for 7d predictions
                
            historical_data = self.db.get_recent_market_data(token, hours=historical_hours)
            
            if not historical_data:
                logger.logger.warning(f"No historical data found for {token}")
                
            # Extract price and volume history
            prices = [current_price]  # Start with current price
            volumes = [token_data.get('volume', 0)]  # Start with current volume
            highs = [current_price]
            lows = [current_price]
            
            # Add historical data
            for entry in reversed(historical_data):  # Oldest to newest
                prices.insert(0, entry['price'])
                volumes.insert(0, entry['volume'])
                # Use price as high/low if not available
                highs.insert(0, entry['price'])
                lows.insert(0, entry['price'])
                
            # Ensure we have at least some data
            if len(prices) < 5:
                logger.logger.warning(f"Limited data for {token}, duplicating available prices")
                # Duplicate the last price a few times
                prices = [prices[0]] * (5 - len(prices)) + prices
                volumes = [volumes[0]] * (5 - len(volumes)) + volumes
                highs = [highs[0]] * (5 - len(highs)) + highs
                lows = [lows[0]] * (5 - len(lows)) + lows
                
            # Fail early if we still don't have enough data (should never happen with above padding)
            if len(prices) < 2:
                logger.logger.error(f"Insufficient price data for {token}: {len(prices)} data points")
                self.pending_predictions.discard(f"{token}_{timeframe}")
                self.error_cooldowns[cooldown_key] = ("Insufficient price data", time.time() + 300)  # 5 minute cooldown
                return self._generate_fallback_prediction(token, token_data, timeframe)
                
            # Generate technical analysis with improved error handling
            try:
                tech_analysis = TechnicalIndicators.analyze_technical_indicators(
                    prices, volumes, highs, lows, timeframe
                )
            except Exception as tech_error:
                logger.log_error(f"Technical Analysis - {token} ({timeframe})", str(tech_error))
                logger.logger.error(f"Technical analysis failed for {token} ({timeframe}): {str(tech_error)}")
                # Use default neutral technical analysis
                tech_analysis = {
                    "overall_trend": "neutral",
                    "trend_strength": 50,
                    "volatility": 5.0,
                    "signals": {
                        "rsi": "neutral",
                        "macd": "neutral",
                        "bollinger_bands": "neutral",
                        "stochastic": "neutral"
                    }
                }
            
            # Generate statistical forecast with improved error handling
            try:
                # Choose the best statistical model based on timeframe
                if timeframe == "1h":
                    stat_forecast = StatisticalModels.arima_forecast(prices, forecast_steps=1, timeframe=timeframe)
                elif timeframe == "24h":
                    try:
                        # Try Holt-Winters for daily data
                        stat_forecast = StatisticalModels.holt_winters_forecast(prices, forecast_steps=1, timeframe=timeframe)
                    except:
                        # Fallback to ARIMA
                        stat_forecast = StatisticalModels.arima_forecast(prices, forecast_steps=1, timeframe=timeframe)
                else:  # 7d
                    # For weekly, use weighted average forecast
                    stat_forecast = StatisticalModels.weighted_average_forecast(prices, volumes, forecast_steps=1, timeframe=timeframe)
            except Exception as stat_error:
                logger.log_error(f"Statistical Forecast - {token} ({timeframe})", str(stat_error))
                logger.logger.error(f"Statistical forecast failed for {token} ({timeframe}): {str(stat_error)}")
                # Use simple moving average as fallback
                stat_forecast = StatisticalModels.moving_average_forecast(prices, forecast_steps=1, timeframe=timeframe)
            
            # Generate machine learning forecast with improved error handling
            try:
                # Choose the best ML model based on timeframe and data availability
                if timeframe == "1h" and len(prices) >= 48:
                    try:
                        # Try RandomForest for hourly with sufficient data
                        ml_forecast = MachineLearningModels.random_forest_forecast(
                            prices, volumes, forecast_steps=1, timeframe=timeframe
                        )
                    except Exception as rf_error:
                        logger.log_error(f"RF Model - {token} ({timeframe})", str(rf_error))
                        # Fallback to linear regression
                        ml_forecast = MachineLearningModels.linear_regression_forecast(
                            prices, volumes, forecast_steps=1, timeframe=timeframe
                        )
                elif timeframe == "24h" and len(prices) >= 60:
                    try:
                        # Try LSTM for daily if we have enough data
                        import tensorflow as tf
                        ml_forecast = MachineLearningModels.lstm_forecast(
                            prices, forecast_steps=1, timeframe=timeframe
                        )
                    except Exception as lstm_error:
                        logger.log_error(f"LSTM Model - {token} ({timeframe})", str(lstm_error))
                        # Fallback to random forest
                        ml_forecast = MachineLearningModels.random_forest_forecast(
                            prices, volumes, forecast_steps=1, timeframe=timeframe
                        )
                else:
                    # Default to linear regression for others
                    ml_forecast = MachineLearningModels.linear_regression_forecast(
                        prices, volumes, forecast_steps=1, timeframe=timeframe
                    )
            except Exception as ml_error:
                logger.log_error(f"ML Forecast - {token} ({timeframe})", str(ml_error))
                logger.logger.error(f"ML forecast failed for {token} ({timeframe}): {str(ml_error)}")
                # Use statistical forecast as fallback
                ml_forecast = stat_forecast
            
            # Get market conditions
            try:
                market_conditions = self._generate_market_conditions(market_data, token)
            except Exception as market_error:
                logger.log_error(f"Market Conditions - {token}", str(market_error))
                market_conditions = {"market_trend": "unknown", "btc_dominance": "unknown"}
            
            # Get recent predictions and their accuracy for this timeframe
            try:
                recent_predictions = self._get_recent_prediction_performance(token, timeframe)
            except Exception as perf_error:
                logger.log_error(f"Prediction Performance - {token} ({timeframe})", str(perf_error))
                recent_predictions = []
            
            # Generate Claude-enhanced prediction if available
            if self.client:
                try:
                    logger.logger.debug(f"Generating Claude-enhanced {timeframe} prediction for {token}")
                    prediction = self.client.generate_enhanced_prediction(
                        token=token,
                        current_price=current_price,
                        technical_analysis=tech_analysis,
                        statistical_forecast=stat_forecast,
                        ml_forecast=ml_forecast,
                        timeframe=timeframe,
                        price_history_24h=historical_data,
                        market_conditions=market_conditions,
                        recent_predictions=recent_predictions
                    )
                except Exception as claude_error:
                    logger.log_error(f"Claude Prediction - {token} ({timeframe})", str(claude_error))
                    logger.logger.error(f"Claude prediction failed for {token} ({timeframe}): {str(claude_error)}")
                    # Combine predictions manually if Claude fails
                    prediction = self._combine_predictions(
                        token=token,
                        current_price=current_price,
                        technical_analysis=tech_analysis,
                        statistical_forecast=stat_forecast,
                        ml_forecast=ml_forecast,
                        market_conditions=market_conditions,
                        timeframe=timeframe
                    )
            else:
                # Combine predictions manually if Claude not available
                logger.logger.debug(f"Generating manually combined {timeframe} prediction for {token}")
                prediction = self._combine_predictions(
                    token=token,
                    current_price=current_price,
                    technical_analysis=tech_analysis,
                    statistical_forecast=stat_forecast,
                    ml_forecast=ml_forecast,
                    market_conditions=market_conditions,
                    timeframe=timeframe
                )
                
            # Apply FOMO-inducing adjustments to the prediction
            try:
                prediction = self._apply_fomo_enhancement(prediction, current_price, tech_analysis, timeframe)
            except Exception as fomo_error:
                logger.log_error(f"FOMO Enhancement - {token} ({timeframe})", str(fomo_error))
                # Just continue with the existing prediction
            
            # Check if prediction looks reasonable
            try:
                self._validate_prediction(prediction, current_price, timeframe)
            except Exception as validation_error:
                logger.log_error(f"Prediction Validation - {token} ({timeframe})", str(validation_error))
                # Just continue with existing prediction
                
            # Store the prediction in the database
            try:
                prediction_db_id = self._store_prediction(token, prediction, timeframe)
                # Add DB ID to prediction object
                prediction["db_id"] = prediction_db_id
            except Exception as db_error:
                logger.log_error(f"Prediction Storage - {token} ({timeframe})", str(db_error))
                # Continue without DB storage
            
            # Add to cached predictions for reply functionality
            self._cache_prediction(token, timeframe, prediction)
            
            # Clear from pending list
            self.pending_predictions.discard(f"{token}_{timeframe}")
            
            # Clear any error cooldown
            if cooldown_key in self.error_cooldowns:
                del self.error_cooldowns[cooldown_key]
                
            logger.logger.info(f"Successfully generated {timeframe} prediction for {token}")
            return prediction
            
        except Exception as e:
            # Log the detailed error
            error_msg = f"Prediction Generation - {token} ({timeframe}) - {str(e)}"
            logger.log_error(error_msg, traceback.format_exc())
            logger.logger.error(f"Error generating prediction for {token} ({timeframe}): {str(e)}")
            
            # Clear from pending list
            self.pending_predictions.discard(f"{token}_{timeframe}")
            
            # Set error cooldown to avoid repeated failures
            cooldown_key = f"{token}_{timeframe}"
            self.error_cooldowns[cooldown_key] = (str(e), time.time() + 300)  # 5 minute cooldown
            
            # Return a simple fallback prediction
            return self._generate_fallback_prediction(token, market_data.get(token, {}), timeframe)
            
    def _cache_prediction(self, token: str, timeframe: str, prediction: Dict[str, Any]) -> None:
        """Cache a prediction for quick retrieval for reply functionality"""
        cache_key = f"{token}_{timeframe}"
        self.reply_ready_predictions[cache_key] = {
            "prediction": prediction,
            "timestamp": time.time()
        }
        
        # Trim cache if it gets too large
        if len(self.reply_ready_predictions) > self.max_cached_predictions:
            # Remove oldest entries
            sorted_keys = sorted(
                self.reply_ready_predictions.keys(),
                key=lambda k: self.reply_ready_predictions[k]["timestamp"]
            )
            for old_key in sorted_keys[:len(sorted_keys) // 5]:  # Remove oldest 20%
                del self.reply_ready_predictions[old_key]
                
    def _get_cached_prediction(self, token: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached prediction if available and not too old"""
        cache_key = f"{token}_{timeframe}"
        
        if cache_key in self.reply_ready_predictions:
            cached = self.reply_ready_predictions[cache_key]
            age_seconds = time.time() - cached["timestamp"]
            
            # Define max age based on timeframe
            max_age = {
                "1h": 300,    # 5 minutes for hourly
                "24h": 3600,  # 1 hour for daily
                "7d": 14400   # 4 hours for weekly
            }.get(timeframe, 300)
            
            if age_seconds < max_age:
                return cached["prediction"]
                
        return None                    

    def _get_predictions_for_reply(self, token: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all current predictions for a token across all timeframes
        Optimized for reply functionality
        """
        result = {}
        
        # Check each timeframe
        for timeframe in ["1h", "24h", "7d"]:
            # Try to get from cache first (fastest)
            cached = self._get_cached_prediction(token, timeframe)
            if cached:
                result[timeframe] = cached
                continue
                
            # Try to get from DB next
            try:
                db_prediction = self.db.get_all_timeframe_predictions(token)
                if timeframe in db_prediction:
                    result[timeframe] = db_prediction[timeframe]
                    continue
            except Exception as db_error:
                logger.log_error(f"DB Prediction Retrieval - {token}", str(db_error))
                # Continue without DB predictions
                
            # Generate a new prediction as last resort
            try:
                # We need market data first
                market_data = self._get_market_data_for_token(token)
                if market_data:
                    # Generate new prediction
                    new_prediction = self.generate_prediction(token, market_data, timeframe)
                    result[timeframe] = new_prediction
            except Exception as gen_error:
                logger.log_error(f"Reply Prediction Generation - {token}", str(gen_error))
                # Skip this timeframe if we can't generate a prediction
                
        return result
                
    def _get_market_data_for_token(self, token: str) -> Dict[str, Any]:
        """Get current market data for a single token (for reply functionality)"""
        # This method would normally integrate with your market data source
        # Here's a simplified implementation that gets the latest from DB
        try:
            # Get most recent data from database
            recent_data = self.db.get_recent_market_data(token, hours=1)
            if not recent_data or len(recent_data) == 0:
                return {}
                
            # Use most recent entry
            latest = recent_data[0]
            
            # Format as expected by prediction engine
            return {
                token: {
                    'current_price': latest['price'],
                    'volume': latest['volume'],
                    'price_change_percentage_24h': latest.get('price_change_24h', 0),
                    'market_cap': latest.get('market_cap', 0),
                    'ath': latest.get('ath', 0),
                    'ath_change_percentage': latest.get('ath_change_percentage', 0)
                }
            }
        except Exception as e:
            logger.log_error(f"Market Data Retrieval - {token}", str(e))
            return {}
    
    def _generate_market_conditions(self, market_data: Dict[str, Any], excluded_token: str) -> Dict[str, Any]:
        """
        Generate overall market condition assessment
        Enhanced for reply support
        """
        try:
            # Remove the token itself from analysis
            filtered_data = {token: data for token, data in market_data.items() if token != excluded_token}
            
            if not filtered_data:
                return {"market_trend": "unknown", "btc_dominance": "unknown", "market_status": "unknown"}
                
            # Calculate market trend
            price_changes = [data.get('price_change_percentage_24h', 0) for data in filtered_data.values() 
                             if data.get('price_change_percentage_24h') is not None]
            
            if not price_changes:
                market_trend = "unknown"
            else:
                avg_change = sum(price_changes) / len(price_changes)
                if avg_change > 3:
                    market_trend = "strongly bullish"
                elif avg_change > 1:
                    market_trend = "bullish"
                elif avg_change < -3:
                    market_trend = "strongly bearish"
                elif avg_change < -1:
                    market_trend = "bearish"
                else:
                    market_trend = "neutral"
                    
            # Calculate BTC dominance if available
            btc_dominance = "unknown"
            if "BTC" in market_data:
                btc_market_cap = market_data["BTC"].get('market_cap', 0)
                total_market_cap = sum(data.get('market_cap', 0) for data in market_data.values() 
                                     if data.get('market_cap') is not None)
                if total_market_cap > 0:
                    btc_dominance = f"{(btc_market_cap / total_market_cap) * 100:.1f}%"
                    
            # Calculate market volatility
            if len(price_changes) > 1:
                market_volatility = f"{np.std(price_changes):.2f}"
            else:
                market_volatility = "unknown"
                
            # Group tokens by category (simple approach)
            layer1s = ["ETH", "SOL", "AVAX", "NEAR", "POL"]
            defi = ["UNI", "AAVE"]
            
            # Calculate sector performance
            sector_performance = {}
            
            # Layer 1s
            layer1_changes = [data.get('price_change_percentage_24h', 0) for token, data in filtered_data.items() 
                             if token in layer1s and data.get('price_change_percentage_24h') is not None]
            if layer1_changes:
                sector_performance["layer1"] = sum(layer1_changes) / len(layer1_changes)
                
            # DeFi
            defi_changes = [data.get('price_change_percentage_24h', 0) for token, data in filtered_data.items() 
                           if token in defi and data.get('price_change_percentage_24h') is not None]
            if defi_changes:
                sector_performance["defi"] = sum(defi_changes) / len(defi_changes)
                
            # Calculate market status (for reply context)
            if market_trend in ["strongly bullish", "bullish"]:
                market_status = "bull market"
            elif market_trend in ["strongly bearish", "bearish"]:
                market_status = "bear market"
            else:
                market_status = "sideways market"
            
            # Add hourly trend if available
            hourly_changes = []
            for token_data in filtered_data.values():
                if 'price_data' in token_data and len(token_data['price_data']) >= 2:
                    current = token_data['price_data'][0]
                    hour_ago = token_data['price_data'][-1]
                    if current > 0 and hour_ago > 0:
                        hourly_change = (current - hour_ago) / hour_ago * 100
                        hourly_changes.append(hourly_change)
            
            if hourly_changes:
                avg_hourly_change = sum(hourly_changes) / len(hourly_changes)
                hourly_trend = "bullish" if avg_hourly_change > 0.5 else "bearish" if avg_hourly_change < -0.5 else "neutral"
            else:
                hourly_trend = "unknown"
                
            # Combine all data
            return {
                "market_trend": market_trend,
                "btc_dominance": btc_dominance,
                "market_volatility": market_volatility,
                "sector_performance": sector_performance,
                "market_status": market_status,
                "hourly_trend": hourly_trend
            }
            
        except Exception as e:
            logger.log_error("Market Conditions", str(e))
            logger.logger.error(f"Error calculating market conditions: {str(e)}")
            return {"market_trend": "unknown", "btc_dominance": "unknown", "market_status": "unknown"}
    
    def _get_recent_prediction_performance(self, token: str, timeframe: str) -> List[Dict[str, Any]]:
        """Get recent prediction performance for token and timeframe"""
        try:
            # Get prediction performance from database
            performance = self.db.get_prediction_performance(token=token, timeframe=timeframe)
            
            if not performance:
                return []
                
            # Get recent prediction outcomes
            recent_outcomes = self.db.get_recent_prediction_outcomes(token=token, limit=10)
            
            # Filter for the specific timeframe
            filtered_outcomes = [outcome for outcome in recent_outcomes if outcome.get('timeframe') == timeframe]
            
            # Format for Claude input
            formatted_outcomes = []
            for outcome in filtered_outcomes:
                formatted_outcomes.append({
                    "prediction_value": outcome.get("prediction_value", 0),
                    "actual_outcome": outcome.get("actual_outcome", 0),
                    "was_correct": outcome.get("was_correct", 0) == 1,
                    "accuracy_percentage": outcome.get("accuracy_percentage", 0),
                    "evaluation_time": outcome.get("evaluation_time", "")
                })
                
            return formatted_outcomes
            
        except Exception as e:
            logger.log_error(f"Get Recent Prediction Performance - {token} ({timeframe})", str(e))
            return []
    
    def _combine_predictions(self, token: str, current_price: float, technical_analysis: Dict[str, Any],
                           statistical_forecast: Dict[str, Any], ml_forecast: Dict[str, Any],
                           market_conditions: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Manually combine predictions when Claude is not available
        Enhanced for better reply functionality
        Adjusted for different timeframes
        """
        try:
            # Validate inputs
            if not technical_analysis or not statistical_forecast or not ml_forecast:
                return self._generate_fallback_prediction(token, {"current_price": current_price}, timeframe)
            
            # Extract forecasts
            try:
                stat_forecast_values = statistical_forecast.get("forecast", [current_price])
                stat_forecast = stat_forecast_values[0] if stat_forecast_values else current_price
            except Exception:
                stat_forecast = current_price
                
            try:
                ml_forecast_values = ml_forecast.get("forecast", [current_price])
                ml_forecast_val = ml_forecast_values[0] if ml_forecast_values else current_price
            except Exception:
                ml_forecast_val = current_price
            
            # Get technical trend
            trend = technical_analysis.get("overall_trend", "neutral")
            trend_strength = technical_analysis.get("trend_strength", 50)
            
            # Determine weights based on trend strength and timeframe
            # If trend is strong, give more weight to technical analysis
            # For longer timeframes, rely more on statistical and ML models
            if timeframe == "1h":
                # For hourly, technical indicators matter more
                if trend_strength > 70:
                    tech_weight = 0.5
                    stat_weight = 0.25
                    ml_weight = 0.25
                elif trend_strength > 50:
                    tech_weight = 0.4
                    stat_weight = 0.3
                    ml_weight = 0.3
                else:
                    tech_weight = 0.2
                    stat_weight = 0.4
                    ml_weight = 0.4
            elif timeframe == "24h":
                # For daily, balance between technical and models
                if trend_strength > 70:
                    tech_weight = 0.4
                    stat_weight = 0.3
                    ml_weight = 0.3
                elif trend_strength > 50:
                    tech_weight = 0.35
                    stat_weight = 0.35
                    ml_weight = 0.3
                else:
                    tech_weight = 0.25
                    stat_weight = 0.4
                    ml_weight = 0.35
            else:  # 7d
                # For weekly, models matter more than short-term indicators
                if trend_strength > 70:
                    tech_weight = 0.35
                    stat_weight = 0.35
                    ml_weight = 0.3
                elif trend_strength > 50:
                    tech_weight = 0.3
                    stat_weight = 0.4
                    ml_weight = 0.3
                else:
                    tech_weight = 0.2
                    stat_weight = 0.45
                    ml_weight = 0.35
                
            # Calculate technical prediction based on trend
            if "bullish" in trend:
                tech_prediction = current_price * (1 + (0.01 * (trend_strength - 50) / 50))
            elif "bearish" in trend:
                tech_prediction = current_price * (1 - (0.01 * (trend_strength - 50) / 50))
            else:
                tech_prediction = current_price
                
            # Calculate weighted prediction
            weighted_prediction = (
                tech_weight * tech_prediction + 
                stat_weight * stat_forecast + 
                ml_weight * ml_forecast_val
            )
            
            # Calculate confidence level
            # Higher trend strength = higher confidence
            # Longer timeframe = lower confidence
            base_confidence = {
                "1h": 65,
                "24h": 60,
                "7d": 55
            }.get(timeframe, 65)
            
            confidence_boost = min(20, (trend_strength - 50) * 0.4) if trend_strength > 50 else 0
            confidence = base_confidence + confidence_boost
            
            # Calculate price range based on timeframe
            # Wider ranges for longer timeframes
            volatility = technical_analysis.get("volatility", 5.0)
            base_range_factor = {
                "1h": 0.005,
                "24h": 0.015,
                "7d": 0.025
            }.get(timeframe, 0.005)
            
            range_factor = max(base_range_factor, min(base_range_factor * 4, volatility / 200))
            
            lower_bound = weighted_prediction * (1 - range_factor)
            upper_bound = weighted_prediction * (1 + range_factor)
            
            # Calculate percentage change
            percent_change = ((weighted_prediction / current_price) - 1) * 100
            
            # Determine sentiment
            # Adjust thresholds based on timeframe
            sentiment_thresholds = {
                "1h": 1.0,    # 1% for hourly
                "24h": 2.5,   # 2.5% for daily
                "7d": 5.0     # 5% for weekly
            }.get(timeframe, 1.0)
            
            if percent_change > sentiment_thresholds:
                sentiment = "BULLISH"
            elif percent_change < -sentiment_thresholds:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
                
            # Generate rationale based on technical analysis and market conditions
            timeframe_desc = {
                "1h": "hour",
                "24h": "24 hours",
                "7d": "week"
            }.get(timeframe, timeframe)
            
            if sentiment == "BULLISH":
                rationale = f"Strong {trend} with confluence from statistical models suggest upward momentum in the next {timeframe_desc}."
            elif sentiment == "BEARISH":
                rationale = f"{trend.capitalize()} confirmed by multiple indicators, suggesting continued downward pressure over the next {timeframe_desc}."
            else:
                rationale = f"Mixed signals with {trend} but limited directional conviction for the {timeframe_desc} ahead."
                
            # Identify key factors
            key_factors = []
            
            # Add technical factor
            tech_signals = technical_analysis.get("signals", {})
            strongest_signal = None
            strongest_value = "neutral"
            
            # Find the strongest non-neutral signal
            for key, value in tech_signals.items():
                if value != "neutral" and (strongest_signal is None or 
                                          (value in ["bullish", "bearish"] and strongest_value not in ["bullish", "bearish"])):
                    strongest_signal = key
                    strongest_value = value
            
            if strongest_signal:
                key_factors.append(f"{strongest_signal.upper()} {strongest_value}")
            else:
                key_factors.append("Technical indicators neutral")
            
            # Add statistical factor
            stat_confidence = statistical_forecast.get("confidence_intervals", [{"80": [0, 0]}])[0]["80"]
            stat_range = abs(stat_confidence[1] - stat_confidence[0])
            key_factors.append(f"Statistical forecast: ${stat_forecast:.4f} ±{stat_range:.2f}")
            
            # Add market factor
            key_factors.append(f"Market trend: {market_conditions.get('market_trend', 'neutral')}")
            
            # Add timeframe-specific factors
            if timeframe == "24h":
                # Add ADX if available
                if "adx" in tech_signals:
                    key_factors.append(f"ADX: {tech_signals['adx']}")
            elif timeframe == "7d":
                # Add sector performance for weekly
                sector_perf = market_conditions.get("sector_performance", {})
                if sector_perf:
                    sector = max(sector_perf.items(), key=lambda x: abs(x[1]))
                    key_factors.append(f"{sector[0].capitalize()} sector: {sector[1]:.1f}%")
            
            return {
                "prediction": {
                    "price": weighted_prediction,
                    "confidence": confidence,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "percent_change": percent_change,
                    "timeframe": timeframe
                },
                "rationale": rationale,
                "sentiment": sentiment,
                "key_factors": key_factors[:3],  # Limit to top 3 factors
                "model_weights": {
                    "technical_analysis": tech_weight,
                    "statistical_models": stat_weight,
                    "machine_learning": ml_weight,
                    "claude_enhanced": 0.0
                }
            }
        except Exception as e:
            # Log detailed error
            error_msg = f"Combined Prediction Error: {str(e)}"
            logger.log_error("Combined Prediction", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Return fallback prediction
            return self._generate_fallback_prediction(token, {"current_price": current_price}, timeframe)    

    def _apply_fomo_enhancement(self, prediction: Dict[str, Any], current_price: float, 
                              tech_analysis: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Apply FOMO-inducing enhancements to predictions
        Makes ranges tighter and slightly exaggerates movement while staying realistic
        Optimized for reply engagement and adjusted for different timeframes
        """
        try:
            # Skip enhancement for already enhanced or fallback predictions
            if prediction.get("is_fallback") or prediction.get("is_emergency_fallback"):
                return prediction
                
            sentiment = prediction.get("sentiment", "NEUTRAL")
            original_price = prediction["prediction"]["price"]
            percent_change = prediction["prediction"]["percent_change"]
            
            # Adjust based on timeframe - don't modify very extreme predictions
            max_change_threshold = {
                "1h": 5.0,
                "24h": 10.0,
                "7d": 20.0
            }.get(timeframe, 5.0)
            
            # Don't modify predictions that are already very bullish or bearish
            if abs(percent_change) > max_change_threshold:
                return prediction
                
            # Get volatility from tech analysis
            volatility = tech_analysis.get("volatility", 5.0)
            
            # Enhance prediction based on sentiment and timeframe
            if sentiment == "BULLISH":
                # Slightly boost bullish predictions to generate FOMO
                # Boost amount increases with timeframe
                if timeframe == "1h":
                    fomo_boost = max(0.2, min(0.8, volatility / 10))
                elif timeframe == "24h":
                    fomo_boost = max(0.5, min(1.5, volatility / 8))
                else:  # 7d
                    fomo_boost = max(1.0, min(2.5, volatility / 6))
                    
                enhanced_price = original_price * (1 + (fomo_boost / 100))
                enhanced_pct = ((enhanced_price / current_price) - 1) * 100
                
                # Make ranges tighter
                base_range_factor = {
                    "1h": 0.004,
                    "24h": 0.01,
                    "7d": 0.015
                }.get(timeframe, 0.004)
                
                range_factor = max(base_range_factor, min(base_range_factor * 3, volatility / 300))
                lower_bound = enhanced_price * (1 - range_factor)
                upper_bound = enhanced_price * (1 + range_factor)
                
                # Make sure upper bound is exciting enough based on timeframe
                min_upper_gain = {
                    "1h": 1.01,
                    "24h": 1.025,
                    "7d": 1.05
                }.get(timeframe, 1.01)
                
                if (upper_bound / current_price) < min_upper_gain:
                    upper_bound = current_price * min_upper_gain
                    
            elif sentiment == "BEARISH":
                # Slightly exaggerate bearish predictions
                if timeframe == "1h":
                    fomo_boost = max(0.2, min(0.8, volatility / 10))
                elif timeframe == "24h":
                    fomo_boost = max(0.5, min(1.5, volatility / 8))
                else:  # 7d
                    fomo_boost = max(1.0, min(2.5, volatility / 6))
                    
                enhanced_price = original_price * (1 - (fomo_boost / 100))
                enhanced_pct = ((enhanced_price / current_price) - 1) * 100
                
                # Make ranges tighter
                base_range_factor = {
                    "1h": 0.004,
                    "24h": 0.01,
                    "7d": 0.015
                }.get(timeframe, 0.004)
                
                range_factor = max(base_range_factor, min(base_range_factor * 3, volatility / 300))
                lower_bound = enhanced_price * (1 - range_factor)
                upper_bound = enhanced_price * (1 + range_factor)
                
                # Make sure lower bound is concerning enough based on timeframe
                min_lower_loss = {
                    "1h": 0.99,
                    "24h": 0.975,
                    "7d": 0.95
                }.get(timeframe, 0.99)
                
                if (lower_bound / current_price) > min_lower_loss:
                    lower_bound = current_price * min_lower_loss
                    
            else:  # NEUTRAL
                # For neutral, make ranges a bit wider to be more accurate
                enhanced_price = original_price
                enhanced_pct = percent_change
                
                # Make ranges slightly tighter than original but not too tight
                base_range_factor = {
                    "1h": 0.006,
                    "24h": 0.015,
                    "7d": 0.025
                }.get(timeframe, 0.006)
                
                range_factor = max(base_range_factor, min(base_range_factor * 3, volatility / 250))
                lower_bound = enhanced_price * (1 - range_factor)
                upper_bound = enhanced_price * (1 + range_factor)
                
            # Update prediction with enhanced values
            prediction["prediction"]["price"] = enhanced_price
            prediction["prediction"]["percent_change"] = enhanced_pct
            prediction["prediction"]["lower_bound"] = lower_bound
            prediction["prediction"]["upper_bound"] = upper_bound
            
            # Slightly boost confidence for FOMO
            # For longer timeframes, apply smaller confidence boost
            confidence_boost = {
                "1h": 5,
                "24h": 3,
                "7d": 2
            }.get(timeframe, 5)
            
            original_confidence = prediction["prediction"]["confidence"]
            prediction["prediction"]["confidence"] = min(85, original_confidence + confidence_boost)
            
            # Mark as FOMO enhanced for potential filtering/tracking
            prediction["fomo_enhanced"] = True
            
            # Note: For reply optimization, we could add special "talking points" here
            if "reply_enhancement" not in prediction:
                prediction["reply_enhancement"] = {}
                
            # Add timeframe-specific talking points for replies
            if timeframe == "1h":
                prediction["reply_enhancement"]["talking_points"] = [
                    "Short-term momentum",
                    "Immediate trading opportunity",
                    "Volatility spike potential"
                ]
            elif timeframe == "24h":
                prediction["reply_enhancement"]["talking_points"] = [
                    "Day-trading setup",
                    "Key support/resistance levels",
                    "24-hour trend continuation"
                ]
            else:  # 7d
                prediction["reply_enhancement"]["talking_points"] = [
                    "Medium-term trend",
                    "Weekly pattern formation",
                    "Swing trading opportunity"
                ]
            
            return prediction
        except Exception as e:
            # Log detailed error
            error_msg = f"FOMO Enhancement Error: {str(e)}"
            logger.log_error("FOMO Enhancement", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Return original prediction unchanged
            return prediction
    
    def _validate_prediction(self, prediction: Dict[str, Any], current_price: float, timeframe: str) -> None:
        """
        Validate prediction values are within reasonable bounds
        Throws exception if values are invalid for better error tracking
        """
        try:
            # Skip validation for fallback predictions since they're already conservative
            if prediction.get("is_fallback") or prediction.get("is_emergency_fallback"):
                return
                
            # Check price prediction
            if "prediction" not in prediction:
                raise ValueError("Missing prediction field")
                
            pred_data = prediction["prediction"]
            
            # Required fields
            required_fields = ["price", "confidence", "lower_bound", "upper_bound", "percent_change", "timeframe"]
            for field in required_fields:
                if field not in pred_data:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Verify numeric fields
            numeric_fields = ["price", "confidence", "lower_bound", "upper_bound", "percent_change"]
            for field in numeric_fields:
                val = pred_data[field]
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Field '{field}' is not numeric: {val}")
                    
            # Verify price is positive
            if pred_data["price"] <= 0:
                raise ValueError(f"Predicted price must be positive: {pred_data['price']}")
                
            # Verify confidence is between 0-100
            if not (0 <= pred_data["confidence"] <= 100):
                raise ValueError(f"Confidence must be between 0-100: {pred_data['confidence']}")
                
            # Verify lower_bound <= price <= upper_bound
            if not (pred_data["lower_bound"] <= pred_data["price"] <= pred_data["upper_bound"]):
                raise ValueError(f"Price {pred_data['price']} outside bounds [{pred_data['lower_bound']}, {pred_data['upper_bound']}]")
                
            # Verify range is reasonable for timeframe
            price_range_pct = (pred_data["upper_bound"] - pred_data["lower_bound"]) / current_price * 100
            max_range_pct = {
                "1h": 10.0,   # 10% for 1 hour
                "24h": 20.0,  # 20% for 24 hours
                "7d": 40.0    # 40% for 7 days
            }.get(timeframe, 10.0)
            
            if price_range_pct > max_range_pct:
                raise ValueError(f"Price range too wide: {price_range_pct}% > {max_range_pct}%")
                
            # Verify percent change matches predicted price
            expected_pct = ((pred_data["price"] / current_price) - 1) * 100
            if abs(expected_pct - pred_data["percent_change"]) > 1.0:  # Allow small differences due to rounding
                raise ValueError(f"Percent change mismatch: {pred_data['percent_change']}% vs expected {expected_pct}%")
                
            # Verify timeframe is valid
            if pred_data["timeframe"] not in ["1h", "24h", "7d"]:
                raise ValueError(f"Invalid timeframe: {pred_data['timeframe']}")
                
            # Verify sentiment is valid
            if "sentiment" in prediction and prediction["sentiment"] not in ["BULLISH", "BEARISH", "NEUTRAL"]:
                raise ValueError(f"Invalid sentiment: {prediction['sentiment']}")
                
        except ValueError as validation_error:
            # Re-raise with more context
            raise ValueError(f"Prediction validation failed: {str(validation_error)}")
    
    def _generate_fallback_prediction(self, token: str, token_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """
        Generate a simple fallback prediction when other methods fail
        Reply-friendly with appropriate confidence levels for each timeframe
        """
        try:
            current_price = token_data.get('current_price', 0)
            if current_price <= 0:
                # If we don't even have a price, use 100 as a placeholder
                # In practice, you'd want to query a different data source
                current_price = 100
                
            # Generate a very simple prediction
            # Slight bullish bias for FOMO
            # Adjust based on timeframe
            if timeframe == "1h":
                prediction_change = 0.5  # 0.5% for 1 hour
                confidence = 70
                lower_factor = 0.995
                upper_factor = 1.015
                rationale = f"Technical indicators suggest mild volatility for {token} in the next hour."
            elif timeframe == "24h":
                prediction_change = 1.2  # 1.2% for 24 hours
                confidence = 65
                lower_factor = 0.985
                upper_factor = 1.035
                rationale = f"Market patterns indicate potential momentum for {token} over the next 24 hours."
            else:  # 7d
                prediction_change = 2.5  # 2.5% for 7 days
                confidence = 60
                lower_factor = 0.97
                upper_factor = 1.06
                rationale = f"Long-term trend analysis suggests possible movement for {token} in the coming week."
                
            prediction_price = current_price * (1 + prediction_change / 100)
            percent_change = prediction_change
            
            # Confidence level and range
            lower_bound = current_price * lower_factor
            upper_bound = current_price * upper_factor
            
            # Determine sentiment
            if prediction_change > 0.5:
                sentiment = "BULLISH"
            elif prediction_change < -0.5:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
                
            # Generate key factors optimized for replies
            if sentiment == "BULLISH":
                key_factors = ["Technical momentum", "Market conditions", "Price action"]
            elif sentiment == "BEARISH":
                key_factors = ["Price resistance", "Market pressure", "Technical indicators"]
            else:
                key_factors = ["Consolidation pattern", "Mixed signals", "Range-bound price action"]
                
            return {
                "prediction": {
                    "price": prediction_price,
                    "confidence": confidence,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "percent_change": percent_change,
                    "timeframe": timeframe
                },
                "rationale": rationale,
                "sentiment": sentiment,
                "key_factors": key_factors,
                "model_weights": {
                    "technical_analysis": 0.6,
                    "statistical_models": 0.2,
                    "machine_learning": 0.1,
                    "claude_enhanced": 0.1
                },
                "is_fallback": True,
                "reply_enhancement": {
                    "talking_points": [
                        "Current market conditions",
                        f"{timeframe} potential movement",
                        "Risk management"
                    ]
                }
            }
        except Exception as e:
            # Log detailed error
            error_msg = f"Fallback Prediction Error: {str(e)}"
            logger.log_error("Fallback Prediction", error_msg)
            logger.logger.debug(traceback.format_exc())
            
            # Return ultra-minimal prediction
            return {
                "prediction": {
                    "price": current_price * 1.005,
                    "confidence": 60,
                    "lower_bound": current_price * 0.99,
                    "upper_bound": current_price * 1.02,
                    "percent_change": 0.5,
                    "timeframe": timeframe
                },
                "rationale": f"Market analysis for {token}.",
                "sentiment": "NEUTRAL",
                "key_factors": ["Market analysis", "Technical indicators"],
                "is_emergency_fallback": True
            }
    
    def _store_prediction(self, token: str, prediction: Dict[str, Any], timeframe: str) -> int:
        """
        Store prediction in database with error handling
        Returns the ID of the stored prediction or 0 if storage failed
        """
        try:
            # Check if DB is available
            if not self.db:
                logger.logger.warning("No database available to store prediction")
                return 0
                
            # Get prediction details from the structure
            prediction_data = prediction["prediction"]
            
            # Set appropriate expiration time based on timeframe
            if timeframe == "1h":
                expiration_time = datetime.now() + timedelta(hours=1)
            elif timeframe == "24h":
                expiration_time = datetime.now() + timedelta(hours=24)
            elif timeframe == "7d":
                expiration_time = datetime.now() + timedelta(days=7)
            else:
                expiration_time = datetime.now() + timedelta(hours=1)  # Default to 1h
            
            # Store in database
            try:
                conn, cursor = self.db._get_connection()
                
                cursor.execute("""
                    INSERT INTO price_predictions (
                        timestamp, token, timeframe, prediction_type,
                        prediction_value, confidence_level, lower_bound, upper_bound,
                        prediction_rationale, method_weights, model_inputs, technical_signals,
                        expiration_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    "price",
                    prediction_data["price"],
                    prediction_data["confidence"],
                    prediction_data["lower_bound"],
                    prediction_data["upper_bound"],
                    prediction["rationale"],
                    json.dumps(prediction.get("model_weights", {})),
                    json.dumps(prediction.get("inputs", {})),
                    json.dumps(prediction.get("key_factors", [])),
                    expiration_time
                ))
                
                # Get the ID of the inserted prediction
                prediction_id = cursor.lastrowid
                
                conn.commit()
                logger.logger.debug(f"Stored {timeframe} prediction for {token} with ID {prediction_id}")
                
                return prediction_id
                
            except Exception as db_error:
                logger.log_error(f"Store Prediction - {token} ({timeframe})", str(db_error))
                if conn:
                    conn.rollback()
                return 0
                
        except Exception as e:
            logger.log_error(f"Store Prediction - {token} ({timeframe})", str(e))
            return 0        

    def evaluate_predictions(self) -> None:
        """
        Evaluate expired predictions and calculate accuracy
        Enhanced error handling and focus on reply-friendly data
        """
        try:
            logger.logger.debug("Starting prediction evaluation")
            
            # Skip if database is not available
            if not self.db:
                logger.logger.warning("Cannot evaluate predictions: No database available")
                return
                
            # Get expired but unevaluated predictions
            try:
                expired_predictions = self.db.get_expired_unevaluated_predictions()
                logger.logger.debug(f"Found {len(expired_predictions)} expired unevaluated predictions")
            except Exception as fetch_error:
                logger.log_error("Fetch Expired Predictions", str(fetch_error))
                logger.logger.error(f"Failed to fetch expired predictions: {str(fetch_error)}")
                return
                
            if not expired_predictions:
                logger.logger.debug("No expired predictions to evaluate")
                return
                
            # Process each prediction
            for prediction in expired_predictions:
                try:
                    token = prediction["token"]
                    prediction_value = prediction["prediction_value"]
                    lower_bound = prediction["lower_bound"]
                    upper_bound = prediction["upper_bound"]
                    timeframe = prediction["timeframe"]
                    
                    # Get the actual price at expiration time
                    try:
                        actual_result = self._get_actual_price(token, prediction["expiration_time"])
                        
                        if not actual_result:
                            logger.logger.warning(f"No actual price found for {token} at evaluation time")
                            continue
                            
                        actual_price = actual_result
                    except Exception as price_error:
                        logger.log_error(f"Get Actual Price - {token}", str(price_error))
                        logger.logger.warning(f"Failed to get actual price for {token}: {str(price_error)}")
                        continue
                        
                    # Calculate accuracy
                    try:
                        # Calculate percentage accuracy
                        price_diff = abs(actual_price - prediction_value)
                        accuracy_percentage = (1 - (price_diff / prediction_value)) * 100 if prediction_value > 0 else 0
                        
                        # Determine if prediction was correct (within bounds)
                        was_correct = lower_bound <= actual_price <= upper_bound
                        
                        # Calculate deviation
                        deviation = ((actual_price / prediction_value) - 1) * 100
                        
                        # Store evaluation result
                        evaluation_result = self._record_prediction_outcome(
                            prediction["id"], actual_price, accuracy_percentage, 
                            was_correct, deviation
                        )
                        
                        logger.logger.debug(
                            f"Evaluated {token} {timeframe} prediction: "
                            f"Predicted={prediction_value}, Actual={actual_price}, "
                            f"Correct={was_correct}, Accuracy={accuracy_percentage:.1f}%"
                        )
                        
                        # For reply enhancement, track sequential correct predictions
                        if was_correct:
                            self._update_reply_streak(token, timeframe, True)
                        else:
                            self._update_reply_streak(token, timeframe, False)
                            
                    except Exception as eval_error:
                        logger.log_error(f"Prediction Evaluation - {token}", str(eval_error))
                        logger.logger.warning(f"Failed to evaluate prediction for {token}: {str(eval_error)}")
                        continue
                        
                except Exception as pred_error:
                    logger.log_error("Process Prediction", str(pred_error))
                    logger.logger.warning(f"Failed to process prediction ID {prediction.get('id', 'unknown')}: {str(pred_error)}")
                    continue
                    
            logger.logger.info(f"Evaluated {len(expired_predictions)} expired predictions")
            
        except Exception as e:
            logger.log_error("Prediction Evaluation", str(e))
            logger.logger.error(f"Prediction evaluation failed: {str(e)}")
    
    def _get_actual_price(self, token: str, evaluation_time: datetime) -> float:
        """
        Get the actual price for a token at a specific evaluation time
        Enhanced for reply support with fallbacks
        """
        try:
            # Try to get from database
            if self.db:
                cursor = self.db.cursor
                
                # Query with flexible time window (within 5 minutes of evaluation time)
                cursor.execute("""
                    SELECT price
                    FROM market_data
                    WHERE chain = ?
                    AND timestamp BETWEEN datetime(?, '-5 minutes') AND datetime(?, '+5 minutes')
                    ORDER BY ABS(JULIANDAY(timestamp) - JULIANDAY(?))
                    LIMIT 1
                """, (token, evaluation_time, evaluation_time, evaluation_time))
                
                result = cursor.fetchone()
                if result:
                    return result["price"]
                    
                # If no data within 5 minutes, try the closest available data
                cursor.execute("""
                    SELECT price
                    FROM market_data
                    WHERE chain = ?
                    AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token, evaluation_time))
                
                result = cursor.fetchone()
                if result:
                    return result["price"]
                    
            # If database query failed or returned no results, try to get current price
            current_market_data = self._get_market_data_for_token(token)
            if current_market_data and token in current_market_data:
                return current_market_data[token]['current_price']
                
            # Last resort fallback
            logger.logger.warning(f"Could not find actual price for {token}, using fallback")
            return 0.0
            
        except Exception as e:
            logger.log_error(f"Get Actual Price - {token}", str(e))
            return 0.0
    
    def _record_prediction_outcome(self, prediction_id: int, actual_price: float, 
                                 accuracy_percentage: float, was_correct: bool, 
                                 deviation: float) -> bool:
        """Record the outcome of a prediction with error handling"""
        try:
            # Skip if database is not available
            if not self.db:
                logger.logger.warning("Cannot record prediction outcome: No database available")
                return False
                
            conn, cursor = self.db._get_connection()
            
            try:
                # Get the prediction details to determine token and timeframe
                cursor.execute("""
                    SELECT token, timeframe, prediction_type FROM price_predictions WHERE id = ?
                """, (prediction_id,))
                
                prediction_info = cursor.fetchone()
                if not prediction_info:
                    logger.logger.warning(f"Prediction ID {prediction_id} not found")
                    return False
                    
                token = prediction_info["token"]
                timeframe = prediction_info["timeframe"]
                prediction_type = prediction_info["prediction_type"]
                
                # Get market conditions at evaluation time
                market_data = self._get_market_data_for_token(token)
                market_conditions = json.dumps({
                    "evaluation_time": datetime.now().isoformat(),
                    "token": token,
                    "market_data": market_data.get(token, {})
                })
                
                # Store the outcome
                cursor.execute("""
                    INSERT INTO prediction_outcomes (
                        prediction_id, actual_outcome, accuracy_percentage,
                        was_correct, evaluation_time, deviation_from_prediction,
                        market_conditions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_id,
                    actual_price,
                    accuracy_percentage,
                    1 if was_correct else 0,
                    datetime.now(),
                    deviation,
                    market_conditions
                ))
                
                # Update the performance summary
                self._update_prediction_performance(token, timeframe, prediction_type, was_correct, abs(deviation))
                
                # Update timeframe metrics
                self._update_timeframe_outcome_metrics(token, timeframe, was_correct, accuracy_percentage)
                
                conn.commit()
                return True
                
            except Exception as db_error:
                logger.log_error(f"Record Prediction Outcome - {prediction_id}", str(db_error))
                if conn:
                    conn.rollback()
                return False
                
        except Exception as e:
            logger.log_error(f"Record Prediction Outcome - {prediction_id}", str(e))
            return False
    
    def _update_prediction_performance(self, token: str, timeframe: str, prediction_type: str, was_correct: bool, deviation: float) -> None:
        """Update prediction performance summary with error handling"""
        try:
            # Skip if database is not available
            if not self.db:
                return
                
            conn, cursor = self.db._get_connection()
            
            try:
                # Check if performance record exists
                cursor.execute("""
                    SELECT * FROM prediction_performance
                    WHERE token = ? AND timeframe = ? AND prediction_type = ?
                """, (token, timeframe, prediction_type))
                
                performance = cursor.fetchone()
                
                if performance:
                    # Update existing record
                    performance_dict = dict(performance)
                    total_predictions = performance_dict["total_predictions"] + 1
                    correct_predictions = performance_dict["correct_predictions"] + (1 if was_correct else 0)
                    accuracy_rate = (correct_predictions / total_predictions) * 100
                    
                    # Update average deviation (weighted average)
                    avg_deviation = (performance_dict["avg_deviation"] * performance_dict["total_predictions"] + deviation) / total_predictions
                    
                    cursor.execute("""
                        UPDATE prediction_performance
                        SET total_predictions = ?,
                            correct_predictions = ?,
                            accuracy_rate = ?,
                            avg_deviation = ?,
                            updated_at = ?
                        WHERE id = ?
                    """, (
                        total_predictions,
                        correct_predictions,
                        accuracy_rate,
                        avg_deviation,
                        datetime.now(),
                        performance_dict["id"]
                    ))
                    
                else:
                    # Create new record
                    cursor.execute("""
                        INSERT INTO prediction_performance (
                            token, timeframe, prediction_type, total_predictions,
                            correct_predictions, accuracy_rate, avg_deviation, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        token,
                        timeframe,
                        prediction_type,
                        1,
                        1 if was_correct else 0,
                        100 if was_correct else 0,
                        deviation,
                        datetime.now()
                    ))
                    
                conn.commit()
                
            except Exception as db_error:
                logger.log_error(f"Update Prediction Performance - {token}", str(db_error))
                if conn:
                    conn.rollback()
                    
        except Exception as e:
            logger.log_error(f"Update Prediction Performance - {token}", str(e))
    
    def _update_timeframe_outcome_metrics(self, token: str, timeframe: str, was_correct: bool, accuracy_percentage: float) -> None:
        """Update timeframe metrics with outcome data and error handling"""
        try:
            # Skip if database is not available
            if not self.db:
                return
                
            conn, cursor = self.db._get_connection()
            
            try:
                # Check if metrics record exists
                cursor.execute("""
                    SELECT * FROM timeframe_metrics
                    WHERE token = ? AND timeframe = ?
                """, (token, timeframe))
                
                metrics = cursor.fetchone()
                
                if metrics:
                    # Update existing metrics
                    metrics_dict = dict(metrics)
                    total_count = metrics_dict["total_count"] + 1
                    correct_count = metrics_dict["correct_count"] + (1 if was_correct else 0)
                    
                    # Recalculate average accuracy with new data point
                    # Use weighted average based on number of predictions
                    old_weight = (total_count - 1) / total_count if total_count > 1 else 0
                    new_weight = 1 / total_count if total_count > 0 else 1
                    
                    # Safely calculate average
                    if total_count > 0:
                        avg_accuracy = (metrics_dict["avg_accuracy"] * old_weight) + (accuracy_percentage * new_weight)
                    else:
                        avg_accuracy = accuracy_percentage
                    
                    cursor.execute("""
                        UPDATE timeframe_metrics
                        SET avg_accuracy = ?,
                            total_count = ?,
                            correct_count = ?,
                            last_updated = ?
                        WHERE token = ? AND timeframe = ?
                    """, (
                        avg_accuracy,
                        total_count,
                        correct_count,
                        datetime.now(),
                        token,
                        timeframe
                    ))
                else:
                    # Should not happen normally, but create metrics if missing
                    cursor.execute("""
                        INSERT INTO timeframe_metrics (
                            timestamp, token, timeframe, avg_accuracy,
                            total_count, correct_count, model_weights,
                            best_model, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(),
                        token,
                        timeframe,
                        accuracy_percentage,
                        1,
                        1 if was_correct else 0,
                        "{}",
                        "unknown",
                        datetime.now()
                    ))
                    
                conn.commit()
                
            except Exception as db_error:
                logger.log_error(f"Update Timeframe Outcome Metrics - {token} ({timeframe})", str(db_error))
                if conn:
                    conn.rollback()
                    
        except Exception as e:
            logger.log_error(f"Update Timeframe Outcome Metrics - {token} ({timeframe})", str(e))
    
    # Reply-focused tracking methods
    
    def _update_reply_streak(self, token: str, timeframe: str, correct: bool) -> None:
        """
        Track streaks of correct/incorrect predictions for reply enhancement
        """
        try:
            key = f"{token}_{timeframe}_streak"
            
            # Get current streak data from DB or initialize
            streak_data = self._get_streak_data(token, timeframe)
            
            if correct:
                # Update correct streak
                streak_data["current_correct_streak"] += 1
                streak_data["current_incorrect_streak"] = 0
                
                # Update maximum streaks
                if streak_data["current_correct_streak"] > streak_data["max_correct_streak"]:
                    streak_data["max_correct_streak"] = streak_data["current_correct_streak"]
            else:
                # Update incorrect streak
                streak_data["current_incorrect_streak"] += 1
                streak_data["current_correct_streak"] = 0
                
                # Update maximum streaks
                if streak_data["current_incorrect_streak"] > streak_data["max_incorrect_streak"]:
                    streak_data["max_incorrect_streak"] = streak_data["current_incorrect_streak"]
            
            # Update total counts
            streak_data["total_evaluated"] += 1
            streak_data["total_correct"] = streak_data["total_correct"] + 1 if correct else streak_data["total_correct"]
            
            # Calculate accuracy rate
            if streak_data["total_evaluated"] > 0:
                streak_data["accuracy_rate"] = (streak_data["total_correct"] / streak_data["total_evaluated"]) * 100
            
            # Save updated streak data
            self._save_streak_data(token, timeframe, streak_data)
            
        except Exception as e:
            logger.log_error(f"Update Reply Streak - {token} ({timeframe})", str(e))
    
    def _get_streak_data(self, token: str, timeframe: str) -> Dict[str, Any]:
        """Get streak data for token and timeframe"""
        try:
            # Default streak data
            default_data = {
                "token": token,
                "timeframe": timeframe,
                "current_correct_streak": 0,
                "current_incorrect_streak": 0,
                "max_correct_streak": 0,
                "max_incorrect_streak": 0,
                "total_evaluated": 0,
                "total_correct": 0,
                "accuracy_rate": 0
            }
            
            # Skip if database is not available
            if not self.db:
                return default_data
                
            # Try to get from database
            try:
                conn, cursor = self.db._get_connection()
                
                # Try to get from generic_json_data table
                cursor.execute("""
                    SELECT data FROM generic_json_data
                    WHERE data_type = 'prediction_streak'
                    AND data LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (f'%"{token}"%{timeframe}%',))
                
                result = cursor.fetchone()
                if result and result["data"]:
                    data = json.loads(result["data"])
                    if token in data and timeframe in data[token]:
                        return data[token][timeframe]
                
                return default_data
                
            except Exception as db_error:
                logger.log_error(f"Get Streak Data - {token} ({timeframe})", str(db_error))
                return default_data
                
        except Exception as e:
            logger.log_error(f"Get Streak Data - {token} ({timeframe})", str(e))
            return default_data
    
    def _save_streak_data(self, token: str, timeframe: str, streak_data: Dict[str, Any]) -> None:
        """Save streak data for token and timeframe"""
        try:
            # Skip if database is not available
            if not self.db:
                return
                
            # Try to save to database
            try:
                conn, cursor = self.db._get_connection()
                
                # First get current data
                cursor.execute("""
                    SELECT data, id FROM generic_json_data
                    WHERE data_type = 'prediction_streak'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                data = {}
                
                if result and result["data"]:
                    data = json.loads(result["data"])
                    
                # Update data
                if token not in data:
                    data[token] = {}
                    
                data[token][timeframe] = streak_data
                
                # Store updated data
                if result:
                    # Update existing record
                    cursor.execute("""
                        UPDATE generic_json_data
                        SET data = ?, timestamp = ?
                        WHERE id = ?
                    """, (json.dumps(data), datetime.now(), result["id"]))
                else:
                    # Create new record
                    cursor.execute("""
                        INSERT INTO generic_json_data (
                            timestamp, data_type, data
                        ) VALUES (?, ?, ?)
                    """, (datetime.now(), 'prediction_streak', json.dumps(data)))
                
                conn.commit()
                
            except Exception as db_error:
                logger.log_error(f"Save Streak Data - {token} ({timeframe})", str(db_error))
                if conn:
                    conn.rollback()
                    
        except Exception as e:
            logger.log_error(f"Save Streak Data - {token} ({timeframe})", str(e))
    
    def get_streak_info_for_reply(self, token: str, timeframe: str) -> Dict[str, Any]:
        """
        Get streak information formatted for reply enhancement
        """
        try:
            streak_data = self._get_streak_data(token, timeframe)
            
            # Format for replies
            reply_info = {
                "has_streak": False,
                "streak_text": "",
                "accuracy": 0,
                "confidence_modifier": 0
            }
            
            # Check if we have a notable streak
            correct_streak = streak_data["current_correct_streak"]
            accuracy = streak_data["accuracy_rate"]
            
            if correct_streak >= 3:
                reply_info["has_streak"] = True
                reply_info["streak_text"] = f"correctly predicted {token} movement {correct_streak} times in a row"
                reply_info["confidence_modifier"] = min(10, correct_streak * 2)  # Up to +10% confidence
            
            if accuracy >= 70 and streak_data["total_evaluated"] >= 5:
                reply_info["has_streak"] = True
                reply_info["accuracy"] = accuracy
                reply_info["streak_text"] += f"{' and ' if reply_info['streak_text'] else ''}maintained {accuracy:.1f}% accuracy on {token} predictions"
                reply_info["confidence_modifier"] = max(reply_info["confidence_modifier"], 5)  # At least +5% confidence
            
            return reply_info
            
        except Exception as e:
            logger.log_error(f"Get Streak Info For Reply - {token} ({timeframe})", str(e))
            return {"has_streak": False, "streak_text": "", "accuracy": 0, "confidence_modifier": 0}

            # Reply-optimized methods
    
    def get_reply_ready_prediction(self, token: str, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Get a prediction specifically formatted for reply generation
        This is a key method for the reply functionality
        """
        try:
            # Try to get cached prediction first (fastest)
            cache_key = f"{token}_{timeframe}"
            cached = self._get_cached_prediction(token, timeframe)
            
            if cached:
                # Enhance for reply
                return self._format_for_reply(token, timeframe, cached)
            
            # Try DB next
            try:
                if self.db:
                    # Get active prediction from DB
                    predictions = self.db.get_active_predictions(token, timeframe)
                    if predictions and len(predictions) > 0:
                        # Use most recent prediction
                        db_prediction = predictions[0]
                        return self._format_for_reply(token, timeframe, db_prediction)
            except Exception as db_error:
                logger.log_error(f"DB Reply Prediction - {token}", str(db_error))
            
            # Generate new prediction as last resort
            try:
                # Need market data
                market_data = self._get_market_data_for_token(token)
                if market_data:
                    new_prediction = self.generate_prediction(token, market_data, timeframe)
                    return self._format_for_reply(token, timeframe, new_prediction)
            except Exception as gen_error:
                logger.log_error(f"Generate Reply Prediction - {token}", str(gen_error))
            
            # If all else fails, return minimal fallback
            fallback = self._generate_fallback_prediction(token, {"current_price": 0}, timeframe)
            return self._format_for_reply(token, timeframe, fallback)
            
        except Exception as e:
            logger.log_error(f"Get Reply Ready Prediction - {token}", str(e))
            # Return minimal prediction data that won't crash reply generation
            return {
                "token": token,
                "timeframe": timeframe,
                "price": 0,
                "percent_change": 0,
                "sentiment": "NEUTRAL",
                "confidence": 50,
                "rationale": f"Market analysis for {token}.",
                "is_fallback": True
            }
    
    def _format_for_reply(self, token: str, timeframe: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a prediction specifically for reply generation
        Extracts only the needed fields and adds reply-specific enhancements
        """
        try:
            # Extract core prediction data with safe defaults
            pred_data = prediction.get("prediction", {})
            
            # Prepare simplified prediction for reply
            reply_prediction = {
                "token": token,
                "timeframe": timeframe,
                "price": pred_data.get("price", 0),
                "lower_bound": pred_data.get("lower_bound", 0),
                "upper_bound": pred_data.get("upper_bound", 0),
                "percent_change": pred_data.get("percent_change", 0),
                "confidence": pred_data.get("confidence", 60),
                "sentiment": prediction.get("sentiment", "NEUTRAL"),
                "rationale": prediction.get("rationale", f"Market analysis for {token}."),
                "key_factors": prediction.get("key_factors", ["Market analysis"]),
                "is_fallback": prediction.get("is_fallback", False),
                "is_emergency_fallback": prediction.get("is_emergency_fallback", False)
            }
            
            # Add streak information if available
            streak_info = self.get_streak_info_for_reply(token, timeframe)
            reply_prediction["streak_info"] = streak_info
            
            # Add accuracy information if available
            try:
                if self.db:
                    performance = self.db.get_prediction_performance(token=token, timeframe=timeframe)
                    if performance and len(performance) > 0:
                        perf = performance[0]
                        reply_prediction["accuracy"] = {
                            "rate": perf.get("accuracy_rate", 0),
                            "total": perf.get("total_predictions", 0),
                            "correct": perf.get("correct_predictions", 0)
                        }
            except Exception as perf_error:
                logger.log_error(f"Performance Data for Reply - {token}", str(perf_error))
            
            # Add reply-specific talking points
            if "reply_enhancement" in prediction:
                reply_prediction["talking_points"] = prediction["reply_enhancement"].get("talking_points", [])
            else:
                # Default talking points by timeframe
                if timeframe == "1h":
                    reply_prediction["talking_points"] = [
                        "Short-term price movement",
                        "Immediate market conditions",
                        "Trading opportunity"
                    ]
                elif timeframe == "24h":
                    reply_prediction["talking_points"] = [
                        "Daily trend analysis",
                        "Key price levels",
                        "Market sentiment"
                    ]
                else:  # 7d
                    reply_prediction["talking_points"] = [
                        "Weekly outlook",
                        "Medium-term pattern",
                        "Market cycles"
                    ]
            
            # Add timeframe-specific language
            reply_prediction["timeframe_desc"] = {
                "1h": "hour",
                "24h": "24 hours",
                "7d": "week"
            }.get(timeframe, timeframe)
            
            # Format price for display
            price = reply_prediction["price"]
            if price < 0.01:
                reply_prediction["price_formatted"] = f"${price:.8f}"
            elif price < 1:
                reply_prediction["price_formatted"] = f"${price:.6f}"
            elif price < 100:
                reply_prediction["price_formatted"] = f"${price:.4f}"
            elif price < 10000:
                reply_prediction["price_formatted"] = f"${price:.2f}"
            else:
                reply_prediction["price_formatted"] = f"${price:,.2f}"
                
            # Format percent change
            pct_change = reply_prediction["percent_change"]
            reply_prediction["percent_change_formatted"] = f"{'+' if pct_change >= 0 else ''}{pct_change:.2f}%"
            
            # Add movement strength descriptor
            if abs(pct_change) < 1:
                reply_prediction["movement_strength"] = "slight"
            elif abs(pct_change) < 3:
                reply_prediction["movement_strength"] = "moderate"
            elif abs(pct_change) < 7:
                reply_prediction["movement_strength"] = "significant"
            else:
                reply_prediction["movement_strength"] = "strong"
                
            # Add direction
            if pct_change > 0.1:
                reply_prediction["direction"] = "upward"
            elif pct_change < -0.1:
                reply_prediction["direction"] = "downward"
            else:
                reply_prediction["direction"] = "sideways"
                
            return reply_prediction
            
        except Exception as e:
            logger.log_error(f"Format For Reply - {token}", str(e))
            # Return minimal data that won't crash reply generation
            return {
                "token": token,
                "timeframe": timeframe,
                "price": 0,
                "percent_change": 0,
                "sentiment": "NEUTRAL",
                "rationale": f"Market analysis for {token}.",
                "is_fallback": True,
                "timeframe_desc": timeframe
            }
    
    def get_all_timeframe_predictions_for_reply(self, token: str) -> Dict[str, Dict[str, Any]]:
        """
        Get predictions for all timeframes for a specific token
        Formatted specifically for reply generation
        """
        result = {}
        
        for timeframe in ["1h", "24h", "7d"]:
            try:
                prediction = self.get_reply_ready_prediction(token, timeframe)
                result[timeframe] = prediction
            except Exception as tf_error:
                logger.log_error(f"Get All Timeframes - {token} ({timeframe})", str(tf_error))
                # Add minimal fallback
                result[timeframe] = {
                    "token": token,
                    "timeframe": timeframe,
                    "price": 0,
                    "percent_change": 0,
                    "sentiment": "NEUTRAL",
                    "rationale": f"Market analysis for {token}.",
                    "is_fallback": True,
                    "timeframe_desc": timeframe
                }
        
        return result
    
    def get_model_accuracy_by_timeframe(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model accuracy statistics for each timeframe
        Used for enhancing reply trustworthiness
        """
        try:
            results = {}
            
            for timeframe in ["1h", "24h", "7d"]:
                if self.db:
                    timeframe_stats = self.db.get_prediction_accuracy_by_model(timeframe=timeframe, days=30)
                    
                    if timeframe_stats:
                        results[timeframe] = timeframe_stats
                
            # If no DB results, provide minimal stats
            if not results:
                for timeframe in ["1h", "24h", "7d"]:
                    results[timeframe] = {
                        "models": {
                            "combined": {
                                "accuracy_rate": 60,
                                "total_predictions": 10
                            }
                        },
                        "total_predictions": 10,
                        "overall_accuracy": 60
                    }
                    
            return results
                
        except Exception as e:
            logger.log_error("Get Model Accuracy By Timeframe", str(e))
            # Return minimal stats for reply generation
            return {
                "1h": {"overall_accuracy": 60},
                "24h": {"overall_accuracy": 60},
                "7d": {"overall_accuracy": 60}
            }
    
    # Utility methods
    
    def get_token_symbols(self) -> List[str]:
        """
        Get list of all token symbols from database
        For use in reply searching
        """
        try:
            if not self.db:
                # Return common tokens as fallback
                return ["BTC", "ETH", "SOL", "XRP", "BNB", "AVAX", "UNI"]
                
            # Query database for all tokens with predictions
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                SELECT DISTINCT token FROM price_predictions
                UNION
                SELECT DISTINCT chain FROM market_data
            """)
            
            return [row["token"] for row in cursor.fetchall()]
            
        except Exception as e:
            logger.log_error("Get Token Symbols", str(e))
            # Return common tokens as fallback
            return ["BTC", "ETH", "SOL", "XRP", "BNB", "AVAX", "UNI"]
    
    def get_prediction_count(self) -> Dict[str, int]:
        """Get count of predictions by timeframe"""
        try:
            if not self.db:
                return {"1h": 0, "24h": 0, "7d": 0, "total": 0}
                
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                SELECT timeframe, COUNT(*) as count
                FROM price_predictions
                GROUP BY timeframe
            """)
            
            result = {"total": 0}
            
            for row in cursor.fetchall():
                timeframe = row["timeframe"]
                count = row["count"]
                result[timeframe] = count
                result["total"] += count
                
            return result
            
        except Exception as e:
            logger.log_error("Get Prediction Count", str(e))
            return {"1h": 0, "24h": 0, "7d": 0, "total": 0}
    
    def get_prediction_accuracy_summary(self) -> Dict[str, Any]:
        """Get summary of prediction accuracy for reply enhancement"""
        try:
            if not self.db:
                return {"overall": 60, "1h": 60, "24h": 60, "7d": 60}
                
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                SELECT timeframe, 
                       SUM(correct_predictions) as total_correct,
                       SUM(total_predictions) as total_count
                FROM prediction_performance
                GROUP BY timeframe
            """)
            
            results = {}
            overall_correct = 0
            overall_total = 0
            
            for row in cursor.fetchall():
                timeframe = row["timeframe"]
                correct = row["total_correct"]
                total = row["total_count"]
                
                if total > 0:
                    accuracy = (correct / total) * 100
                else:
                    accuracy = 0
                    
                results[timeframe] = accuracy
                overall_correct += correct
                overall_total += total
                
            # Calculate overall accuracy
            if overall_total > 0:
                results["overall"] = (overall_correct / overall_total) * 100
            else:
                results["overall"] = 0
                
            return results
            
        except Exception as e:
            logger.log_error("Get Prediction Accuracy Summary", str(e))
            return {"overall": 60, "1h": 60, "24h": 60, "7d": 60}
    
    def cleanup(self) -> None:
        """Cleanup resources when shutting down"""
        try:
            # Close DB connection if needed
            if hasattr(self, 'db') and self.db:
                self.db.close()
                
            # Clear any cached data
            self.pending_predictions.clear()
            self.error_cooldowns.clear()
            self.reply_ready_predictions.clear()
            
            logger.logger.info("Prediction Engine resources cleaned up")
            
        except Exception as e:
            logger.log_error("Prediction Engine Cleanup", str(e))


# Singleton-like instance that can be imported by other modules
def initialize_prediction_engine(database, claude_api_key=None):
    """Initialize the prediction engine with database and optional API key"""
    try:
        engine = PredictionEngine(database, claude_api_key)
        return engine
    except Exception as e:
        logger.log_error("Prediction Engine Initialization", str(e))
        # Return basic engine with minimal functionality
        return PredictionEngine(database)
