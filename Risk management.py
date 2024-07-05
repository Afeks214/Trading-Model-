# risk_management.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from config import RiskConfig
from indicators.mlmi import MLMI
from indicators.fvg import FairValueGap

@dataclass
class Trade:
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    direction: str  # 'long' or 'short'

class RiskManagement:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.trades_today = 0
        self.active_trades: List[Trade] = []

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        risk_amount = self.config.total_capital * self.config.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        return int(risk_amount / risk_per_share)

    def validate_trade(self, entry_price: float, stop_loss: float, take_profit: float) -> bool:
        if self.trades_today >= self.config.max_trades_per_day:
            return False

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if reward / risk < self.config.min_risk_reward:
            return False

        return True

    def open_trade(self, entry_price: float, stop_loss: float, take_profit: float, direction: str) -> Trade:
        if not self.validate_trade(entry_price, stop_loss, take_profit):
            return None

        position_size = self.calculate_position_size(entry_price, stop_loss)
        trade = Trade(entry_price, stop_loss, take_profit, position_size, direction)
        self.active_trades.append(trade)
        self.trades_today += 1
        return trade

    def update_trailing_stop(self, trade: Trade, current_price: float) -> float:
        if trade.direction == 'long':
            new_stop = max(trade.stop_loss, current_price - (current_price - trade.entry_price) * self.config.trailing_stop_factor)
        else:  # short
            new_stop = min(trade.stop_loss, current_price + (trade.entry_price - current_price) * self.config.trailing_stop_factor)
        
        trade.stop_loss = new_stop
        return new_stop

    def check_exit(self, trade: Trade, current_price: float, mlmi: MLMI) -> Tuple[bool, str]:
        if trade.direction == 'long':
            if current_price <= trade.stop_loss:
                return True, "Stop Loss"
            if current_price >= trade.take_profit:
                return True, "Take Profit"
        else:  # short
            if current_price >= trade.stop_loss:
                return True, "Stop Loss"
            if current_price <= trade.take_profit:
                return True, "Take Profit"
        
        if mlmi.check_divergence(trade.direction, current_price):
            return True, "MLMI Divergence"
        
        return False, ""

    def close_trade(self, trade: Trade) -> None:
        self.active_trades.remove(trade)

    def take_partial_profits(self, trade: Trade, current_price: float) -> None:
        if trade.direction == 'long':
            partial_target = trade.entry_price + (trade.take_profit - trade.entry_price) * self.config.partial_profit_factor
            if current_price >= partial_target:
                trade.position_size //= 2
        else:  # short
            partial_target = trade.entry_price - (trade.entry_price - trade.take_profit) * self.config.partial_profit_factor
            if current_price <= partial_target:
                trade.position_size //= 2

    def reset_daily_trades(self) -> None:
        self.trades_today = 0

    def get_risk_exposure(self) -> float:
        total_risk = sum(trade.position_size * abs(trade.entry_price - trade.stop_loss) 
                         for trade in self.active_trades)
        return total_risk / self.config.total_capital

# config.py

from dataclasses import dataclass

@dataclass
class RiskConfig:
    total_capital: float
    risk_per_trade: float
    max_trades_per_day: int
    min_risk_reward: float
    trailing_stop_factor: float
    partial_profit_factor: float

# main.py

from risk_management import RiskManagement
from config import RiskConfig
from ibkr_interface import IBKRInterface
from strategy import TradingStrategy
from indicators.mlmi import MLMI
from indicators.fvg import FairValueGap
from indicators.quadratic_regression import QuadraticRegression

def main():
    config = RiskConfig(
        total_capital=100000,
        risk_per_trade=0.02,
        max_trades_per_day=5,
        min_risk_reward=2.0,
        trailing_stop_factor=0.5,
        partial_profit_factor=0.5
    )

    risk_manager = RiskManagement(config)
    ibkr = IBKRInterface()
    mlmi = MLMI()
    fvg = FairValueGap()
    qr = QuadraticRegression()
    
    strategy = TradingStrategy(risk_manager, ibkr, mlmi, fvg, qr)
    
    strategy.run()

if __name__ == "__main__":
    main()