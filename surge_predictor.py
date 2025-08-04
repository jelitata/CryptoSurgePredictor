import ccxt
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

class CryptoSurgePredictor:
    def __init__(self, symbol='ETH/USD', timeframe='4h', lookback_days=10, api_key=None, api_secret=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = ccxt.kraken({'apiKey': api_key, 'secret': api_secret})
        self.scaler = MinMaxScaler()

    def fetch_ohlcv(self):
        """Получение исторических данных с биржи Kraken."""
        since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_sentiment_signals(self):
        """Получение данных настроений с X (заглушка для API X)."""
        # В реальной версии использовать API X для анализа настроений
        return np.random.rand(len(self.fetch_ohlcv())) * 50

    def prepare_data(self, df):
        """Подготовка данных для модели."""
        df['returns'] = df['close'].pct_change()
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['sentiment'] = self.fetch_sentiment_signals()
        features = df[['close', 'volume', 'rsi', 'sentiment']].dropna()

        scaled_data = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(1 if scaled_data[i, 0] > np.percentile(scaled_data[:, 0], 85) else 0)  # Всплеск цены
        return np.array(X), np.array(y)

    def calculate_rsi(self, prices, period=14):
        """Расчет RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def build_model(self):
        """Создание GRU-модели."""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(60, 4)),
            Dropout(0.3),
            GRU(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y):
        """Обучение модели."""
        model = self.build_model()
        model.fit(X, y, epochs=8, batch_size=16, validation_split=0.2, verbose=1)
        return model

    def predict_surge(self, model, X):
        """Прогноз всплесков цен."""
        predictions = model.predict(X)
        return (predictions > 0.5).astype(int)

    def visualize_results(self, df, predictions):
        """Визуализация результатов с Plotly."""
        df = df.iloc[60:].copy()
        df['surge_prediction'] = predictions

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=df['timestamp'][df['surge_prediction'] == 1],
                                 y=df['close'][df['surge_prediction'] == 1],
                                 mode='markers', name='Predicted Surge', marker=dict(size=10, color='red')))
        fig.update_layout(title=f'Price Surges for {self.symbol}', xaxis_title='Time', yaxis_title='Price')
        fig.write_html('data/sample_output/surge_plot.html')
        fig.show()

    def run(self):
        """Основной метод анализа."""
        df = self.fetch_ohlcv()
        X, y = self.prepare_data(df)
        model = self.train_model(X, y)
        predictions = self.predict_surge(model, X)
        self.visualize_results(df, predictions)
        print(f"Price surges predicted: {np.sum(predictions)} out of {len(predictions)} periods.")

if __name__ == "__main__":
    predictor = CryptoSurgePredictor(symbol='ETH/USD', timeframe='4h', lookback_days=10)
    predictor.run()
