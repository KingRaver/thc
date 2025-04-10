#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, TypedDict, Optional
import os
from dotenv import load_dotenv
from utils.logger import logger
from database import CryptoDatabase

class CryptoInfo(TypedDict):
    id: str
    symbol: str
    name: str

class MarketAnalysisConfig(TypedDict):
    correlation_sensitivity: float
    volatility_threshold: float
    volume_significance: int
    historical_periods: List[int]

class TweetConstraints(TypedDict):
    MIN_LENGTH: int
    MAX_LENGTH: int
    HARD_STOP_LENGTH: int

class CoinGeckoParams(TypedDict):
    vs_currency: str
    ids: str
    order: str
    per_page: int
    page: int
    sparkline: bool
    price_change_percentage: str

class PredictionConfig(TypedDict):
    """Configuration for market predictions"""
    enabled_timeframes: List[str]
    confidence_threshold: float
    model_weights: Dict[str, float]
    prediction_frequency: int
    fomo_factor: float
    range_factor: float
    accuracy_threshold: float

class ProviderConfig(TypedDict):
    """LLM provider-specific configuration"""
    anthropic: Dict[str, str]
    openai: Dict[str, str]
    mistral: Dict[str, str]
    groq: Dict[str, str]

@dataclass
class Config:
    def __init__(self) -> None:
        # Get the absolute path to the .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        logger.logger.info(f"Loading .env from: {env_path}")

        # Load environment variables
        load_dotenv(env_path)
        logger.logger.info("Environment variables loaded")
        
        # Initialize LLM provider configuration
        self.LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'anthropic')
        # Use client-agnostic naming for API keys and models
        self.client_API_KEY: str = self._get_provider_api_key()
        self.client_MODEL: str = self._get_provider_model()
        
        # Debug loaded variables
        logger.logger.info(f"CHROME_DRIVER_PATH: {os.getenv('CHROME_DRIVER_PATH')}")
        logger.logger.info(f"LLM Provider: {self.LLM_PROVIDER}")
        logger.logger.info(f"Client API Key Present: {bool(self.client_API_KEY)}")
        logger.logger.info(f"Client Model: {self.client_MODEL}")
        
        # Provider-specific configurations
        self.PROVIDER_CONFIG: ProviderConfig = {
            "anthropic": {
                "api_key": os.getenv('CLAUDE_API_KEY', ''),
                "model": os.getenv('CLAUDE_MODEL', 'claude-3-haiku-20240307')
            },
            "openai": {
                "api_key": os.getenv('OPENAI_API_KEY', ''),
                "model": os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            },
            "mistral": {
                "api_key": os.getenv('MISTRAL_API_KEY', ''),
                "model": os.getenv('MISTRAL_MODEL', 'mistral-medium')
            },
            "groq": {
                "api_key": os.getenv('GROQ_API_KEY', ''),
                "model": os.getenv('GROQ_MODEL', 'llama2-70b-4096')
            }
        }
        
        # Initialize database
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data', 'crypto_history.db')
        self.db = CryptoDatabase(db_path)
        logger.logger.info(f"Database initialized at: {db_path}")
        
        # Analysis Intervals and Thresholds
        self.BASE_INTERVAL: int = 300  # 5 minutes in seconds
        self.PRICE_CHANGE_THRESHOLD: float = 5.0  # 5% change triggers post
        self.VOLUME_CHANGE_THRESHOLD: float = 10.0  # 10% change triggers post
        self.VOLUME_WINDOW_MINUTES: int = 60  # Track volume over 1 hour
        self.VOLUME_TREND_THRESHOLD: float = 15.0  # 15% change over rolling window triggers post
        
        # Google Sheets Configuration (maintained for compatibility)
        self.GOOGLE_SHEETS_PROJECT_ID: str = os.getenv('GOOGLE_SHEETS_PROJECT_ID', '')
        self.GOOGLE_SHEETS_PRIVATE_KEY: str = os.getenv('GOOGLE_SHEETS_PRIVATE_KEY', '')
        self.GOOGLE_SHEETS_CLIENT_EMAIL: str = os.getenv('GOOGLE_SHEETS_CLIENT_EMAIL', '')
        self.GOOGLE_SHEET_ID: str = os.getenv('GOOGLE_SHEET_ID', '')
        self.GOOGLE_SHEETS_RANGE: str = 'Market Analysis!A:F'

        # Twitter Configuration
        self.TWITTER_USERNAME: str = os.getenv('TWITTER_USERNAME', '')
        self.TWITTER_PASSWORD: str = os.getenv('TWITTER_PASSWORD', '')
        self.CHROME_DRIVER_PATH: str = os.getenv('CHROME_DRIVER_PATH', '/usr/local/bin/chromedriver')
        
        # Analysis Configuration
        self.CORRELATION_INTERVAL: int = 5  # minutes (for testing)
        self.MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
        
        # Market Analysis Parameters
        self.MARKET_ANALYSIS_CONFIG: MarketAnalysisConfig = {
            'correlation_sensitivity': 0.7,
            'volatility_threshold': 2.0,
            'volume_significance': 100000,
            'historical_periods': [1, 4, 24]
        }
        
        # Tweet Length Constraints
        self.TWEET_CONSTRAINTS: TweetConstraints = {
            'MIN_LENGTH': 220,
            'MAX_LENGTH': 270,
            'HARD_STOP_LENGTH': 280
        }
        
        # API Endpoints
        self.COINGECKO_BASE_URL: str = "https://api.coingecko.com/api/v3"
        
        # CoinGecko API Request Settings - Ordered by market cap
        self.COINGECKO_PARAMS: CoinGeckoParams = {
            "vs_currency": "usd",
            "ids": "bitcoin,ethereum,binancecoin,solana,ripple,polkadot,avalanche-2,polygon-pos,near,filecoin,uniswap,aave,kaito",
            "order": "market_cap_desc",
            "per_page": 100,  # Ensure we get all tokens
            "page": 1,
            "sparkline": True,
            "price_change_percentage": "1h,24h,7d"
        }
        
        # Tracked Cryptocurrencies - Ordered consistently by market cap
        self.TRACKED_CRYPTO: Dict[str, CryptoInfo] = {
            'bitcoin': {
                'id': 'bitcoin',
                'symbol': 'btc',
                'name': 'Bitcoin'
            },
            'ethereum': {
                'id': 'ethereum', 
                'symbol': 'eth',
                'name': 'Ethereum'
            },
            'binancecoin': {
                'id': 'binancecoin',
                'symbol': 'bnb',
                'name': 'BNB'
            },
            'solana': {
                'id': 'solana',
                'symbol': 'sol',
                'name': 'Solana'
            },
            'ripple': {
                'id': 'ripple',
                'symbol': 'xrp',
                'name': 'XRP'
            },
            'polkadot': {
                'id': 'polkadot',
                'symbol': 'dot',
                'name': 'Polkadot'
            },
            'avalanche-2': {
                'id': 'avalanche-2',
                'symbol': 'avax',
                'name': 'Avalanche'
            },
            'polygon-pos': {
                'id': 'polygon-pos',
                'symbol': 'pol',
                'name': 'Polygon'
            },
            'near': {
                'id': 'near',
                'symbol': 'near',
                'name': 'NEAR Protocol'
            },
            'filecoin': {
                'id': 'filecoin',
                'symbol': 'fil',
                'name': 'Filecoin'
            },
            'uniswap': {
                'id': 'uniswap',
                'symbol': 'uni',
                'name': 'Uniswap'
            },
            'aave': {
                'id': 'aave',
                'symbol': 'aave',
                'name': 'Aave'
            },
            'kaito': {
                'id': 'kaito',
                'symbol': 'kaito',
                'name': 'Kaito'
            }
        }
        
        # Enhanced Client Analysis Prompt Template - Token-agnostic
        self.client_ANALYSIS_PROMPT: str = """Analyze {token} Market Dynamics:

Current Market Data:
{token}:
- Price: ${price:,.2f}
- 24h Change: {change:.2f}%
- Volume: ${volume:,.0f}

Please provide a concise but detailed market analysis:
1. Short-term Movement: 
   - Price action in last few minutes
   - Volume profile significance
   - Immediate support/resistance levels

2. Market Microstructure:
   - Order flow analysis
   - Volume weighted price trends
   - Market depth indicators

3. Cross-Token Dynamics:
   - Correlation changes with other tokens
   - Relative strength shifts
   - Market maker activity signals

Focus on actionable micro-trends and real-time market behavior. Identify minimal but significant price movements.
Keep the analysis technical but concise, emphasizing key shifts in market dynamics."""
        
        # Prediction Configuration
        self.PREDICTION_CONFIG: PredictionConfig = {
            'enabled_timeframes': ['1h', '24h', '7d'],  # Supported prediction timeframes
            'confidence_threshold': 70.0,  # Minimum confidence percentage to display
            'model_weights': {  # Default weights for prediction models
                'technical_analysis': 0.25,
                'statistical_models': 0.25,
                'machine_learning': 0.25,
                'client_enhanced': 0.25  # Changed from claude_enhanced to client_enhanced
            },
            'prediction_frequency': 60,  # Minutes between predictions for same token
            'fomo_factor': 1.2,  # Multiplier for FOMO-inducing predictions (1.0 = neutral)
            'range_factor': 0.015,  # Default price range factor (percentage)
            'accuracy_threshold': 60.0  # Minimum accuracy to highlight in posts
        }

        # Client Prediction Prompt Template
        self.client_PREDICTION_PROMPT: str = """Generate a precise price prediction for {token} in the next {timeframe}.

Technical Analysis Summary:
- Overall Trend: {trend}
- Trend Strength: {trend_strength}/100
- RSI: {rsi:.2f}
- MACD: {macd_signal}
- Bollinger Bands: {bb_signal}
- Volatility: {volatility:.2f}%

Statistical Models:
- ARIMA Forecast: ${arima_price:.4f}
- Confidence Interval: [${arima_lower:.4f}, ${arima_upper:.4f}]

Machine Learning:
- ML Model Forecast: ${ml_price:.4f}
- Confidence Interval: [${ml_lower:.4f}, ${ml_upper:.4f}]

Current Market Data:
- Current Price: ${current_price:.4f}
- 24h Change: {price_change:.2f}%
- Market Sentiment: {market_sentiment}

Provide a precise prediction following this format:
1. Exact price target with confidence level
2. Price range (lower and upper bounds)
3. Expected percentage change
4. Brief rationale (2-3 sentences)
5. Key factors influencing the prediction

Be precise but conversational. Aim for 70-80% confidence level. Create excitement without being unrealistic."""
        
        # Validation
        self._validate_config()

    def _get_provider_api_key(self) -> str:
        """Get the API key for the currently selected provider"""
        if self.LLM_PROVIDER == 'anthropic':
            return os.getenv('CLAUDE_API_KEY', '')
        elif self.LLM_PROVIDER == 'openai':
            return os.getenv('OPENAI_API_KEY', '')
        elif self.LLM_PROVIDER == 'mistral':
            return os.getenv('MISTRAL_API_KEY', '')
        elif self.LLM_PROVIDER == 'groq':
            return os.getenv('GROQ_API_KEY', '')
        return ''

    def _get_provider_model(self) -> str:
        """Get the model for the currently selected provider"""
        if self.LLM_PROVIDER == 'anthropic':
            return os.getenv('CLAUDE_MODEL', 'claude-3-haiku-20240307')
        elif self.LLM_PROVIDER == 'openai':
            return os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        elif self.LLM_PROVIDER == 'mistral':
            return os.getenv('MISTRAL_MODEL', 'mistral-medium')
        elif self.LLM_PROVIDER == 'groq':
            return os.getenv('GROQ_MODEL', 'llama2-70b-4096')
        return ''

    def _validate_config(self) -> None:
        """Validate required configuration settings"""
        required_settings: List[tuple[str, str]] = [
            ('TWITTER_USERNAME', self.TWITTER_USERNAME),
            ('TWITTER_PASSWORD', self.TWITTER_PASSWORD),
            ('CHROME_DRIVER_PATH', self.CHROME_DRIVER_PATH),
            ('Client API Key', self.client_API_KEY),  # Updated from CLAUDE_API_KEY
            ('GOOGLE_SHEETS_PROJECT_ID', self.GOOGLE_SHEETS_PROJECT_ID),
            ('GOOGLE_SHEETS_PRIVATE_KEY', self.GOOGLE_SHEETS_PRIVATE_KEY),
            ('GOOGLE_SHEETS_CLIENT_EMAIL', self.GOOGLE_SHEETS_CLIENT_EMAIL),
            ('GOOGLE_SHEET_ID', self.GOOGLE_SHEET_ID)
        ]
        
        missing_settings: List[str] = []
        for setting_name, setting_value in required_settings:
            if not setting_value or setting_value.strip() == '':
                missing_settings.append(setting_name)
        
        if missing_settings:
            error_msg = f"Missing required configuration: {', '.join(missing_settings)}"
            logger.log_error("Config", error_msg)
            raise ValueError(error_msg)

    def get_coingecko_markets_url(self) -> str:
        """Get CoinGecko markets API endpoint"""
        return f"{self.COINGECKO_BASE_URL}/coins/markets"

    def get_coingecko_params(self, **kwargs) -> Dict:
        """Get CoinGecko API parameters with optional overrides"""
        params = self.COINGECKO_PARAMS.copy()
        params.update(kwargs)
        return params

    def get_provider_config(self) -> Dict[str, str]:
        """Get the configuration for the currently selected LLM provider"""
        return self.PROVIDER_CONFIG.get(self.LLM_PROVIDER, {})

    @property
    def twitter_selectors(self) -> Dict[str, str]:
        """CSS Selectors for Twitter elements"""
        return {
            'username_input': 'input[autocomplete="username"]',
            'password_input': 'input[type="password"]',
            'login_button': '[data-testid="LoginForm_Login_Button"]',
            'tweet_input': '[data-testid="tweetTextarea_0"]',
            'tweet_button': '[data-testid="tweetButton"]'
        }

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'db'):
            self.db.close()

# Create singleton instance
config = Config()
