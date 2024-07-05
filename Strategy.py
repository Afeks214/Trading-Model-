# strategy.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
from risk_management import RiskManagement, Trade
from ibkr_interface import IBKRInterface
from indicators.mlmi import MLMI
from indicators.fvg import FairValueGap
from indicators.quadratic_regression import QuadraticRegression
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

@dataclass
class StrategyConfig:
    symbols: List[str]
    timeframes: List[str]
    fvg_threshold: float
    mlmi_divergence_threshold: float
    qr_trend_threshold: float
    max_concurrent_trades: int

class TradingStrategy:
    def __init__(self, config: StrategyConfig, risk_manager: RiskManagement, 
                 ibkr: IBKRInterface, mlmi: MLMI, fvg: FairValueGap, qr: QuadraticRegression):
        self.config = config
        self.risk_manager = risk_manager
        self.ibkr = ibkr
        self.mlmi = mlmi
        self.fvg = fvg
        self.qr = qr
        
        self.active_trades: Dict[str, Trade] = {}
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('TradingStrategy')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('trading_strategy.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    async def run(self):
        self.logger.info("Starting trading strategy")
        while True:
            try:
                await self.check_and_update_trades()
                await self.check_for_entry_signals()
                await asyncio.sleep(1)  # Adjust sleep time as needed
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")

    async def check_and_update_trades(self):
        for symbol, trade in list(self.active_trades.items()):
            try:
                current_price = await self.ibkr.get_current_price(symbol)
                
                new_stop = self.risk_manager.update_trailing_stop(trade, current_price)
                await self.ibkr.update_stop_loss(trade, new_stop)
                
                exit_signal, reason = await self.check_exit_conditions(trade, current_price)
                if exit_signal:
                    await self.exit_trade(trade, reason)
                else:
                    self.risk_manager.take_partial_profits(trade, current_price)
            except Exception as e:
                self.logger.error(f"Error updating trade for {symbol}: {str(e)}")

    async def check_for_entry_signals(self):
        if len(self.active_trades) >= self.config.max_concurrent_trades:
            return

        with ThreadPoolExecutor() as executor:
            entry_signals = await asyncio.gather(
                *[self.check_entry_conditions(symbol) for symbol in self.config.symbols]
            )

        for symbol, (entry_signal, direction) in zip(self.config.symbols, entry_signals):
            if entry_signal:
                await self.enter_trade(symbol, direction)

    async def check_entry_conditions(self, symbol) -> Tuple[bool, str]:
        try:
            price_data = await self.ibkr.get_price_data(symbol, self.config.timeframes)
            
            fvg_signal = self.fvg.detect_fvg(price_data, self.config.fvg_threshold)
            mlmi_signal = self.mlmi.check_divergence(price_data, self.config.mlmi_divergence_threshold)
            qr_signal = self.qr.check_trend(price_data, self.config.qr_trend_threshold)

            direction = self.determine_trade_direction(fvg_signal, mlmi_signal, qr_signal)
            
            if direction and await self.confirm_fvg_touch(price_data, fvg_signal, direction):
                return True, direction

            return False, ""
        except Exception as e:
            self.logger.error(f"Error checking entry conditions for {symbol}: {str(e)}")
            return False, ""

    def determine_trade_direction(self, fvg_signal, mlmi_signal, qr_signal) -> str:
        if all(signal == 'bullish' for signal in [fvg_signal, mlmi_signal, qr_signal]):
            return 'long'
        elif all(signal == 'bearish' for signal in [fvg_signal, mlmi_signal, qr_signal]):
            return 'short'
        return ''

    async def confirm_fvg_touch(self, price_data, fvg_signal, direction) -> bool:
        # Implement FVG touch confirmation logic
        pass

    async def enter_trade(self, symbol: str, direction: str):
        try:
            current_price = await self.ibkr.get_current_price(symbol)
            
            stop_loss = self.calculate_stop_loss(current_price, direction)
            take_profit = self.calculate_take_profit(current_price, direction)
            
            trade = self.risk_manager.open_trade(current_price, stop_loss, take_profit, direction)
            if trade:
                order_id = await self.ibkr.place_order(symbol, direction, trade.position_size, stop_loss, take_profit)
                trade.order_id = order_id
                self.active_trades[symbol] = trade
                self.logger.info(f"Entered trade: {trade}")
        except Exception as e:
            self.logger.error(f"Error entering trade for {symbol}: {str(e)}")

    async def check_exit_conditions(self, trade: Trade, current_price: float) -> Tuple[bool, str]:
        if self.check_stop_loss_take_profit(trade, current_price):
            return True, "Stop Loss/Take Profit"

        if await self.mlmi.check_exit_divergence(trade.symbol, trade.direction, current_price):
            return True, "MLMI Divergence"

        if await self.qr.check_trend_change(trade.symbol, trade.direction):
            return True, "Trend Change"

        return False, ""

    def check_stop_loss_take_profit(self, trade: Trade, current_price: float) -> bool:
        return (trade.direction == 'long' and current_price <= trade.stop_loss) or \
               (trade.direction == 'short' and current_price >= trade.stop_loss) or \
               (trade.direction == 'long' and current_price >= trade.take_profit) or \
               (trade.direction == 'short' and current_price <= trade.take_profit)

    async def exit_trade(self, trade: Trade, reason: str):
        try:
            await self.ibkr.close_position(trade.order_id)
            self.risk_manager.close_trade(trade)
            del self.active_trades[trade.symbol]
            self.logger.info(f"Exited trade: {trade}, Reason: {reason}")
        except Exception as e:
            self.logger.error(f"Error exiting trade: {trade}, Error: {str(e)}")

    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
    # Define stop loss percentage (e.g., 2% for this example)
   

def calculate_take_profit(self, entry_price: float, direction: str) -> float: