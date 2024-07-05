import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

@dataclass
class FVG:
    max: float
    min: float
    is_bull: bool
    t: pd.Timestamp

class FairValueGap:
    def __init__(self, data: pd.DataFrame, threshold_per: float = 0, auto: bool = False,
                 show_last: int = 0, mitigation_levels: bool = False,
                 extend: int = 20, dynamic: bool = False,
                 bull_css: str = '#089981', bear_css: str = '#f23645'):
        """
        Initialize the FairValueGap detector.

        :param data: DataFrame with OHLC data
        :param threshold_per: Threshold percentage for FVG detection
        :param auto: Use automatic threshold calculation
        :param show_last: Number of last FVGs to show in plot
        :param mitigation_levels: Show mitigation levels in plot
        :param extend: Number of periods to extend FVGs
        :param dynamic: Use dynamic FVG calculation
        :param bull_css: Color for bullish FVGs
        :param bear_css: Color for bearish FVGs
        """
        self._validate_input(data)
        self.data = data
        self.threshold_per = threshold_per / 100
        self.auto = auto
        self.show_last = show_last
        self.mitigation_levels = mitigation_levels
        self.extend = extend
        self.dynamic = dynamic
        self.bull_css = bull_css
        self.bear_css = bear_css
        
        self.fvg_records: List[FVG] = []
        self.bull_count = self.bear_count = self.bull_mitigated = self.bear_mitigated = 0
        self.max_bull_fvg = self.min_bull_fvg = self.max_bear_fvg = self.min_bear_fvg = np.nan

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {', '.join(required_columns)}")
        if data.empty:
            raise ValueError("Input data is empty")

    def detect_fvg(self, i: int) -> Tuple[bool, bool, Optional[FVG]]:
        """Detect Fair Value Gaps at a specific index."""
        if self.auto:
            threshold = (self.data['high'].iloc[:i+1] - self.data['low'].iloc[:i+1]).cumsum().iloc[-1] / (i + 1)
        else:
            threshold = self.threshold_per

        bull_fvg = (self.data['low'].iloc[i] > self.data['high'].iloc[i-2]) and \
                   (self.data['close'].iloc[i-1] > self.data['high'].iloc[i-2]) and \
                   ((self.data['low'].iloc[i] - self.data['high'].iloc[i-2]) / self.data['high'].iloc[i-2] > threshold)
        
        bear_fvg = (self.data['high'].iloc[i] < self.data['low'].iloc[i-2]) and \
                   (self.data['close'].iloc[i-1] < self.data['low'].iloc[i-2]) and \
                   ((self.data['low'].iloc[i-2] - self.data['high'].iloc[i]) / self.data['high'].iloc[i] > threshold)
        
        new_fvg = None
        if bull_fvg:
            new_fvg = FVG(self.data['low'].iloc[i], self.data['high'].iloc[i-2], True, self.data.index[i])
        elif bear_fvg:
            new_fvg = FVG(self.data['low'].iloc[i-2], self.data['high'].iloc[i], False, self.data.index[i])
        
        return bull_fvg, bear_fvg, new_fvg

    def detect_touched_fvg(self, current_price: float) -> List[FVG]:
        """Detect FVGs touched by the current price."""
        return [fvg for fvg in self.fvg_records if fvg.min <= current_price <= fvg.max]

    def process_fvgs(self):
        """Process the data to detect and manage FVGs."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.detect_fvg, range(2, len(self.data))))

        for i, (bull_fvg, bear_fvg, new_fvg) in enumerate(results, start=2):
            if new_fvg and (not self.fvg_records or new_fvg.t != self.fvg_records[-1].t):
                if self.dynamic:
                    if new_fvg.is_bull:
                        self.max_bull_fvg, self.min_bull_fvg = new_fvg.max, new_fvg.min
                    else:
                        self.max_bear_fvg, self.min_bear_fvg = new_fvg.max, new_fvg.min
                
                self.fvg_records.insert(0, new_fvg)
                if new_fvg.is_bull:
                    self.bull_count += 1
                else:
                    self.bear_count += 1
            elif self.dynamic:
                current_price = self.data['close'].iloc[i]
                if bull_fvg:
                    self.max_bull_fvg = max(min(current_price, self.max_bull_fvg), self.min_bull_fvg)
                elif bear_fvg:
                    self.min_bear_fvg = min(max(current_price, self.min_bear_fvg), self.max_bear_fvg)
            
            self.check_mitigation(i)
            
            current_price = self.data['close'].iloc[i]
            touched_fvgs = self.detect_touched_fvg(current_price)
            if touched_fvgs:
                print(f"Price {current_price:.2f} touched FVG(s) at time {self.data.index[i]}:")
                for fvg in touched_fvgs:
                    print(f"  {'Bull' if fvg.is_bull else 'Bear'} FVG: {fvg.min:.2f} - {fvg.max:.2f}")

    def check_mitigation(self, i: int):
        """Check and handle FVG mitigation."""
        current_price = self.data['close'].iloc[i]
        self.fvg_records = [fvg for fvg in self.fvg_records if not 
                            ((fvg.is_bull and current_price < fvg.min) or 
                             (not fvg.is_bull and current_price > fvg.max))]
        self.bull_mitigated = self.bull_count - sum(1 for fvg in self.fvg_records if fvg.is_bull)
        self.bear_mitigated = self.bear_count - sum(1 for fvg in self.fvg_records if not fvg.is_bull)

    def plot_fvgs(self):
        """Plot the FVGs and price data."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['close'], label='Close Price')
        
        for fvg in self.fvg_records[:self.show_last]:
            color = self.bull_css if fvg.is_bull else self.bear_css
            plt.axhspan(fvg.min, fvg.max, xmin=self.data.index.get_loc(fvg.t) / len(self.data), 
                        xmax=1, alpha=0.3, color=color)
        
        if self.dynamic:
            plt.fill_between(self.data.index, self.max_bull_fvg, self.min_bull_fvg, color=self.bull_css, alpha=0.3)
            plt.fill_between(self.data.index, self.max_bear_fvg, self.min_bear_fvg, color=self.bear_css, alpha=0.3)
        
        touched_prices = [price for i, price in enumerate(self.data['close']) 
                          if self.detect_touched_fvg(price)]
        plt.scatter(self.data.index[self.data['close'].isin(touched_prices)], 
                    touched_prices, color='red', s=50, zorder=5)
        
        plt.title('Fair Value Gap Indicator')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def run(self):
        """Run the FVG detection and analysis."""
        self.process_fvgs()
        self.plot_fvgs()
        
        print(f"Bull FVGs: {self.bull_count}")
        print(f"Bear FVGs: {self.bear_count}")
        print(f"Bull FVGs Mitigated: {self.bull_mitigated}")
        print(f"Bear FVGs Mitigated: {self.bear_mitigated}")
        print(f"Touched FVGs: {len(self.detect_touched_fvg(self.data['close'].iloc[-1]))}")