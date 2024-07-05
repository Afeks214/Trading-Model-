import numpy as np
import pandas as pd
from typing import List, Tuple

class MLMI:
    def __init__(self, num_neighbors: int = 200, momentum_window: int = 20):
        self.num_neighbors = num_neighbors
        self.momentum_window = momentum_window
        self.data = self.Data()

    class Data:
        def __init__(self):
            self.parameter1: List[float] = []
            self.parameter2: List[float] = []
            self.price_array: List[float] = []
            self.result_array: List[float] = []

        def store_previous_trade(self, p1: float, p2: float, close: float):
            if self.parameter1:
                self.parameter1.append(self.parameter1[-1])
                self.parameter2.append(self.parameter2[-1])
                self.price_array.append(self.price_array[-1])
                self.result_array.append(1 if close >= self.price_array[-1] else -1)
            else:
                self.parameter1.append(p1)
                self.parameter2.append(p2)
                self.price_array.append(close)
                self.result_array.append(0)

            self.parameter1[-1] = p1
            self.parameter2[-1] = p2
            self.price_array[-1] = close

        def knn_predict(self, p1: float, p2: float, k: int) -> float:
            if not self.parameter1:
                return 0

            distances = [np.sqrt((p1 - self.parameter1[i])**2 + (p2 - self.parameter2[i])**2) 
                         for i in range(len(self.parameter1))]
            sorted_distances = sorted(distances)
            selected_distances = sorted_distances[:min(k, len(sorted_distances))]
            max_dist = max(selected_distances)
            neighbors = [self.result_array[i] for i in range(len(distances)) if distances[i] <= max_dist]
            return sum(neighbors)

    @staticmethod
    def wma(series: pd.Series, window: int) -> pd.Series:
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

    @staticmethod
    def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['MA_quick'] = self.wma(df['close'], 5)
        df['MA_slow'] = self.wma(df['close'], 20)
        df['rsi_quick'] = self.wma(self.rsi(df['close'], 5), self.momentum_window)
        df['rsi_slow'] = self.wma(self.rsi(df['close'], 20), self.momentum_window)

        predictions = []
        for i in range(len(df)):
            if i > 0 and (self.crossover(df['MA_quick'], df['MA_slow']).iloc[i] or 
                          self.crossunder(df['MA_quick'], df['MA_slow']).iloc[i]):
                self.data.store_previous_trade(df['rsi_slow'].iloc[i], df['rsi_quick'].iloc[i], df['close'].iloc[i])

            prediction = self.data.knn_predict(df['rsi_slow'].iloc[i], df['rsi_quick'].iloc[i], self.num_neighbors)
            predictions.append(prediction)

        df['prediction'] = predictions
        df['prediction_ma'] = self.wma(pd.Series(predictions), 20)

        df['upper'] = df['prediction'].rolling(2000).max()
        df['lower'] = df['prediction'].rolling(2000).min()
        df['upper_'] = df['upper'] - self.wma(df['prediction'].rolling(20).std(), 20)
        df['lower_'] = df['lower'] + self.wma(df['prediction'].rolling(20).std(), 20)

        return df

    def update(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Update the indicator with new data.
        
        :param new_data: New price data
        :return: Updated DataFrame with MLMI calculations
        """
        return self.calculate(new_data)

    def get_signals(self, df: pd.DataFrame) -> dict:
        """
        Generate trading signals based on the MLMI calculations.
        
        :param df: DataFrame with MLMI calculations
        :return: Dictionary of signals
        """
        last = df.iloc[-1]
        return {
            'prediction': last['prediction'],
            'prediction_ma': last['prediction_ma'],
            'is_overbought': last['prediction'] > last['upper_'],
            'is_oversold': last['prediction'] < last['lower_'],
            'cross_above_zero': self.crossover(df['prediction'], pd.Series([0] * len(df))).iloc[-1],
            'cross_below_zero': self.crossunder(df['prediction'], pd.Series([0] * len(df))).iloc[-1],
            'cross_above_ma': self.crossover(df['prediction'], df['prediction_ma']).iloc[-1],
            'cross_below_ma': self.crossunder(df['prediction'], df['prediction_ma']).iloc[-1]
        }

# Example usage
if __name__ == "__main__":
    # Assuming df is your DataFrame with a 'close' column
    df = pd.DataFrame({'close': np.random.random(1000) * 100 + 100})  # Replace with your actual data
    mlmi = MLMI()
    results = mlmi.calculate(df)
    print(results[['close', 'prediction', 'prediction_ma', 'upper_', 'lower_']].tail())
    
    # Example of updating with new data and getting signals
    new_data = pd.DataFrame({'close': [105.0, 106.0, 104.5]})
    updated_results = mlmi.update(new_data)
    signals = mlmi.get_signals(updated_results)
    print("\nSignals:", signals)