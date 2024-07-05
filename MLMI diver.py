import numpy as np
import pandas as pd
from typing import Tuple
from MLMI.py import MLMI

class DivergenceIndicator:
    def __init__(self, window: int = 14):
        self.window = window
        self.mlmi = MLMI()

    @staticmethod
    def find_extrema(series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        highs = series.rolling(window, center=True).apply(lambda x: np.argmax(x) == (window // 2))
        lows = series.rolling(window, center=True).apply(lambda x: np.argmin(x) == (window // 2))
        return highs, lows

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate MLMI
        df = self.mlmi.calculate(df)
        
        # Find extrema for price and MLMI prediction
        price_highs, price_lows = self.find_extrema(df['close'], self.window)
        mlmi_highs, mlmi_lows = self.find_extrema(df['prediction'], self.window)

        # Initialize divergence series
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False

        # Identify bullish divergences (price lower low, indicator higher low)
        for i in range(self.window, len(df) - self.window):
            if price_lows.iloc[i] and mlmi_lows.iloc[i]:
                prev_price_low = df['close'].iloc[i-self.window:i][price_lows.iloc[i-self.window:i]].iloc[-1]
                prev_mlmi_low = df['prediction'].iloc[i-self.window:i][mlmi_lows.iloc[i-self.window:i]].iloc[-1]
                
                if df['close'].iloc[i] < prev_price_low and df['prediction'].iloc[i] > prev_mlmi_low:
                    df.loc[df.index[i], 'bullish_divergence'] = True

        # Identify bearish divergences (price higher high, indicator lower high)
        for i in range(self.window, len(df) - self.window):
            if price_highs.iloc[i] and mlmi_highs.iloc[i]:
                prev_price_high = df['close'].iloc[i-self.window:i][price_highs.iloc[i-self.window:i]].iloc[-1]
                prev_mlmi_high = df['prediction'].iloc[i-self.window:i][mlmi_highs.iloc[i-self.window:i]].iloc[-1]
                
                if df['close'].iloc[i] > prev_price_high and df['prediction'].iloc[i] < prev_mlmi_high:
                    df.loc[df.index[i], 'bearish_divergence'] = True

        return df

    def get_signals(self, df: pd.DataFrame) -> dict:
        """
        Generate trading signals based on the divergence calculations.
        
        :param df: DataFrame with divergence calculations
        :return: Dictionary of signals
        """
        last = df.iloc[-1]
        return {
            'bullish_divergence': last['bullish_divergence'],
            'bearish_divergence': last['bearish_divergence']
        }

# Example usage
if __name__ == "__main__":
    # Assuming df is your DataFrame with a 'close' column
    df = pd.DataFrame({'close': np.random.random(1000) * 100 + 100})  # Replace with your actual data
    divergence_indicator = DivergenceIndicator()
    results = divergence_indicator.calculate(df)
    print(results[['close', 'prediction', 'bullish_divergence', 'bearish_divergence']].tail())
    
    signals = divergence_indicator.get_signals(results)
    print("\nDivergence Signals:", signals)