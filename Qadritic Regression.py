import numpy as np
import pandas as pd

class NadarayaWatsonRationalQuadratic:
    def __init__(self, lookback_window=8.0, relative_weighting=8.0, start_bar=25, smooth_colors=False, lag=2):
        self.lookback_window = lookback_window
        self.relative_weighting = relative_weighting
        self.start_bar = start_bar
        self.smooth_colors = smooth_colors
        self.lag = lag
        
        self.c_bullish = '#3AFF17'  # Green
        self.c_bearish = '#FD1707'  # Red

    def kernel_regression(self, source, h):
        size = len(source)
        yhat = np.zeros(size)
        
        for i in range(size):
            if i < self.start_bar:
                yhat[i] = np.nan
            else:
                current_weight = 0.0
                cumulative_weight = 0.0
                for j in range(i + 1):
                    y = source[j]
                    w = (1 + (j ** 2 / ((h ** 2) * 2 * self.relative_weighting))) ** -self.relative_weighting
                    current_weight += y * w
                    cumulative_weight += w
                yhat[i] = current_weight / cumulative_weight
        
        return yhat

    def calculate(self, source):
        source = np.array(source)
        size = len(source)
        
        yhat1 = self.kernel_regression(source, self.lookback_window)
        yhat2 = self.kernel_regression(source, self.lookback_window - self.lag)
        
        is_bearish = np.zeros(size, dtype=bool)
        is_bullish = np.zeros(size, dtype=bool)
        is_bearish_change = np.zeros(size, dtype=bool)
        is_bullish_change = np.zeros(size, dtype=bool)
        
        for i in range(2, size):
            is_bearish[i] = yhat1[i-1] > yhat1[i]
            is_bullish[i] = yhat1[i-1] < yhat1[i]
            is_bearish_change[i] = is_bearish[i] and (yhat1[i-2] < yhat1[i-1])
            is_bullish_change[i] = is_bullish[i] and (yhat1[i-2] > yhat1[i-1])
        
        is_bullish_cross = np.zeros(size, dtype=bool)
        is_bearish_cross = np.zeros(size, dtype=bool)
        
        for i in range(1, size):
            is_bullish_cross[i] = yhat2[i] > yhat1[i] and yhat2[i-1] <= yhat1[i-1]
            is_bearish_cross[i] = yhat2[i] < yhat1[i] and yhat2[i-1] >= yhat1[i-1]
        
        is_bullish_smooth = yhat2 > yhat1
        is_bearish_smooth = yhat2 < yhat1
        
        color_by_cross = np.where(is_bullish_smooth, self.c_bullish, self.c_bearish)
        color_by_rate = np.where(is_bullish, self.c_bullish, self.c_bearish)
        plot_color = color_by_cross if self.smooth_colors else color_by_rate
        
        alert_bullish = is_bearish_cross if self.smooth_colors else is_bearish_change
        alert_bearish = is_bullish_cross if self.smooth_colors else is_bullish_change
        
        alert_stream = np.where(alert_bearish, -1, np.where(alert_bullish, 1, 0))
        
        return {
            'yhat1': yhat1,
            'yhat2': yhat2,
            'plot_color': plot_color,
            'alert_bullish': alert_bullish,
            'alert_bearish': alert_bearish,
            'alert_stream': alert_stream
        }

    def update(self, new_data):
        """
        Update the indicator with new data.
        
        :param new_data: New price data (can be a single value or an array)
        :return: Updated calculation results
        """
        if isinstance(new_data, (int, float)):
            new_data = [new_data]
        
        return self.calculate(new_data)

    def get_signals(self, results):
        """
        Generate trading signals based on the indicator results.
        
        :param results: The output from the calculate method
        :return: Dictionary of signals
        """
        last_index = -1
        return {
            'trend': 'bullish' if results['plot_color'][last_index] == self.c_bullish else 'bearish',
            'alert': results['alert_stream'][last_index],
            'estimate': results['yhat1'][last_index]
        }

# Example usage
if __name__ == "__main__":
    # Example data
    close_prices = np.random.randn(100).cumsum() + 100
    
    nw = NadarayaWatsonRationalQuadratic()
    results = nw.calculate(close_prices)
    
    # Convert results to pandas DataFrame for easier viewing
    df = pd.DataFrame({
        'close': close_prices,
        'yhat1': results['yhat1'],
        'yhat2': results['yhat2'],
        'plot_color': results['plot_color'],
        'alert_bullish': results['alert_bullish'],
        'alert_bearish': results['alert_bearish'],
        'alert_stream': results['alert_stream']
    })
    
    print(df)
    
    # Example of updating with new data
    new_price = 105.0
    updated_results = nw.update(new_price)
    signals = nw.get_signals(updated_results)
    print("\nUpdated signals:", signals)