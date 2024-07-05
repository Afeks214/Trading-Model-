# ibkr_interface.py

import logging
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, BarData
from ibapi.ticktype import TickTypeEnum
from threading import Thread, Lock
import queue
import time
import sqlite3
from datetime import datetime

class IBKRInterface(EWrapper, EClient):
    def __init__(self, db_path='trading_data.db'):
        EClient.__init__(self, self)
        self.data = {}
        self.data_queue = queue.Queue()
        self.lock = Lock()
        self.logger = self.setup_logger()
        self.db_path = db_path
        self.connect_db()
        self.callback_handlers = {}

    def setup_logger(self):
        logger = logging.getLogger('IBKRInterface')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('ibkr_interface.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def connect_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS market_data
                              (symbol TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)''')
        self.conn.commit()

    def connect(self, host='127.0.0.1', port=7497, clientId=1):
        super().connect(host, port, clientId)
        self.run_thread = Thread(target=self.run)
        self.run_thread.start()
        self.logger.info(f"Connected to IBKR: {host}:{port}")

    def disconnect(self):
        self.done = True
        self.conn.close()
        super().disconnect()
        self.run_thread.join()
        self.logger.info("Disconnected from IBKR")

    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        self.logger.error(f"Error {errorCode}: {errorString}")

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        self.logger.info(f"Next valid order ID: {orderId}")

    def historicalData(self, reqId: int, bar: BarData):
        with self.lock:
            if reqId not in self.data:
                self.data[reqId] = []
            self.data[reqId].append(bar)
        self.store_market_data(reqId, bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.data_queue.put(self.data[reqId])
        self.logger.info(f"Historical data received for reqId: {reqId}")

    def store_market_data(self, reqId, bar):
        symbol = self.reqId_to_symbol.get(reqId, "Unknown")
        self.cursor.execute('''INSERT INTO market_data 
                              (symbol, timestamp, open, high, low, close, volume) 
                              VALUES (?, ?, ?, ?, ?, ?, ?)''',
                            (symbol, bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume))
        self.conn.commit()

    def request_market_data(self, contract, duration='1 D', bar_size='1 min'):
        reqId = self.next_valid_order_id
        self.next_valid_order_id += 1
        self.reqId_to_symbol[reqId] = contract.symbol
        self.reqHistoricalData(reqId, contract, '', duration, bar_size, 'TRADES', 1, 1, False, [])
        return reqId

    def place_order(self, contract, order):
        orderId = self.next_valid_order_id
        self.next_valid_order_id += 1
        self.placeOrder(orderId, contract, order)
        self.logger.info(f"Order placed: {order.action} {order.totalQuantity} {contract.symbol}")
        return orderId

    def create_contract(self, symbol, sec_type='STK', exchange='SMART', currency='USD'):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract

    def create_order(self, action, quantity, order_type='MKT', limit_price=None):
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        if limit_price and order_type == 'LMT':
            order.lmtPrice = limit_price
        return order

    def get_account_summary(self):
        self.reqAccountSummary(1, "All", "NetLiquidation,TotalCashValue,AvailableFunds")

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        self.logger.info(f"Account Summary - {account}: {tag} = {value} {currency}")

    def get_portfolio(self):
        self.reqPortfolioUpdate()

    def updatePortfolio(self, contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName):
        self.logger.info(f"Portfolio Update: {contract.symbol} - Position: {position}, Market Value: {marketValue}")

    def get_market_data(self, contract, duration='1 D', bar_size='1 min'):
        reqId = self.request_market_data(contract, duration, bar_size)
        try:
            data = self.data_queue.get(timeout=10)
            return data
        except queue.Empty:
            self.logger.warning(f"Timeout waiting for market data: {contract.symbol}")
            return None

    def register_callback(self, event_type, callback):
        if event_type not in self.callback_handlers:
            self.callback_handlers[event_type] = []
        self.callback_handlers[event_type].append(callback)

    def trigger_callbacks(self, event_type, *args, **kwargs):
        if event_type in self.callback_handlers:
            for callback in self.callback_handlers[event_type]:
                callback(*args, **kwargs)

    # Real-time data handling
    def reqMktData(self, reqId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions):
        super().reqMktData(reqId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions)
        self.reqId_to_symbol[reqId] = contract.symbol

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == TickTypeEnum.LAST:
            symbol = self.reqId_to_symbol.get(reqId, "Unknown")
            self.logger.info(f"Real-time price update: {symbol} - {price}")
            self.trigger_callbacks('tick_price', symbol, price)

    # Paper trading support
    def use_paper_trading(self):
        # Typically, you would change the connection port to the paper trading port
        self.connect(port=7497)

    # Reconnection mechanism
    def check_connection(self):
        if not self.isConnected():
            self.logger.warning("Connection lost. Attempting to reconnect...")
            self.reconnect()

    def reconnect(self):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self.connect()
                self.logger.info("Reconnected successfully")
                return
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(5)
        self.logger.error("Failed to reconnect after maximum attempts")

if __name__ == '__main__':
    ib = IBKRInterface()
    ib.connect()
    time.sleep(1)  # Wait for connection to be established

    # Example usage
    contract = ib.create_contract('AAPL')
    market_data = ib.get_market_data(contract)
    print(f"Market data for AAPL: {market_data}")

    ib.get_account_summary()
    ib.get_portfolio()

    # Place a limit order to buy 100 shares of AAPL at $150
    order = ib.create_order('BUY', 100, 'LMT', 150)
    ib.place_order(contract, order)

    # Register a callback for real-time price updates
    def on_tick_price(symbol, price):
        print(f"New price for {symbol}: {price}")

    ib.register_callback('tick_price', on_tick_price)

    # Request real-time data
    ib.reqMktData(1, contract, "", False, False, [])

    time.sleep(30)  # Wait for callbacks
    ib.disconnect()