from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import json
from abc import ABC, abstractmethod

load_dotenv()

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class BaseConfig(ABC):
    @abstractmethod
    def validate(self):
        pass

@dataclass
class MLMIConfig(BaseConfig):
    num_neighbors: int = 200
    momentum_window: int = 20

    def validate(self):
        assert self.num_neighbors > 0, "Number of neighbors must be positive"
        assert self.momentum_window > 0, "Momentum window must be positive"

@dataclass
class QuadraticRegressionConfig(BaseConfig):
    window_size: int = 20
    degree: int = 2

    def validate(self):
        assert self.window_size > 0, "Window size must be positive"
        assert self.degree > 0, "Polynomial degree must be positive"

@dataclass
class FVGConfig(BaseConfig):
    gap_threshold: float = 0.001

    def validate(self):
        assert self.gap_threshold > 0, "Gap threshold must be positive"

@dataclass
class RiskManagementConfig(BaseConfig):
    max_position_size: float = 0.02
    max_risk_per_trade: float = 0.01
    max_open_positions: int = 5
    stop_loss_percent: float = 0.02
    take_profit_percent: float = 0.03

    def validate(self):
        assert 0 < self.max_position_size <= 1, "Max position size must be between 0 and 1"
        assert 0 < self.max_risk_per_trade <= 1, "Max risk per trade must be between 0 and 1"
        assert self.max_open_positions > 0, "Max open positions must be positive"
        assert self.stop_loss_percent > 0, "Stop loss percent must be positive"
        assert self.take_profit_percent > 0, "Take profit percent must be positive"

@dataclass
class TradingConfig(BaseConfig):
    symbols: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])
    trading_start_time: str = '09:00'
    trading_end_time: str = '17:00'

    def validate(self):
        assert len(self.symbols) > 0, "At least one symbol must be specified"
        assert len(self.timeframes) > 0, "At least one timeframe must be specified"
        # Add time format validation if needed

@dataclass
class Config:
    account_id: str = field(default_factory=lambda: os.getenv('IBKR_ACCOUNT_ID', 'default_account_id'))
    api_key: str = field(default_factory=lambda: os.getenv('IBKR_API_KEY', 'default_api_key'))
    api_secret: str = field(default_factory=lambda: os.getenv('IBKR_API_SECRET', 'default_api_secret'))
    db_user: str = field(default_factory=lambda: os.getenv('DB_USER', 'default_user'))
    db_password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', 'default_password'))
    db_name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'trading_bot_db'))
    db_host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_file: str = field(default_factory=lambda: os.getenv('LOG_FILE', 'trading_bot.log'))
    mlmi: MLMIConfig = field(default_factory=MLMIConfig)
    quadratic_regression: QuadraticRegressionConfig = field(default_factory=QuadraticRegressionConfig)
    fvg: FVGConfig = field(default_factory=FVGConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    def __post_init__(self):
        self.validate()

    def validate(self):
        try:
            self.mlmi.validate()
            self.quadratic_regression.validate()
            self.fvg.validate()
            self.risk_management.validate()
            self.trading.validate()
        except AssertionError as e:
            raise ConfigurationError(str(e))

    def update_from_dict(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), BaseConfig):
                    getattr(self, key).update_from_dict(value)
                else:
                    setattr(self, key, value)
        self.validate()

    def save_to_file(self, filename: str = 'config.json'):
        try:
            with open(filename, 'w') as f:
                json.dump(self.__dict__, f, default=lambda o: o.__dict__, indent=4)
        except IOError as e:
            raise ConfigurationError(f"Error saving config to file: {e}")

    @classmethod
    def load_from_file(cls, filename: str = 'config.json'):
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            config = cls()
            config.update_from_dict(config_dict)
            return config
        except (IOError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Error loading config from file: {e}")

class ConfigurationManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.config = Config()
        return cls._instance

    def get_config(self) -> Config:
        return self.config

    def update_config(self, config_dict: Dict[str, Any]):
        self.config.update_from_dict(config_dict)

    def save_config(self, filename: str = 'config.json'):
        self.config.save_to_file(filename)

    def load_config(self, filename: str = 'config.json'):
        self.config = Config.load_from_file(filename)

# Usage
config_manager = ConfigurationManager()

def get_config() -> Config:
    return config_manager.get_config()

def update_config(config_dict: Dict[str, Any]):
    config_manager.update_config(config_dict)