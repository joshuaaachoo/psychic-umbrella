"""
Stock Data Fetcher with Technical Indicators
Fetches historical stock data and calculates technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class StockDataFetcher:
    """Fetch stock data and calculate technical indicators"""

    def __init__(self, symbol, period='1y'):
        """
        Initialize stock data fetcher

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period for historical data (e.g., '1mo', '3mo', '6mo', '1y', '2y')
        """
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.news_data = []

    def fetch_data(self):
        """Fetch historical stock data"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)

            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")

            print(f"✓ Fetched {len(self.data)} days of data for {self.symbol}")
            return self.data

        except Exception as e:
            print(f"✗ Error fetching data for {self.symbol}: {e}")
            return None

    def fetch_news(self, max_articles=100):
        """
        Fetch recent news articles for the stock

        Args:
            max_articles: Maximum number of articles to fetch

        Returns:
            List of news articles with sentiment scores
        """
        try:
            ticker = yf.Ticker(self.symbol)
            news = ticker.news

            if not news:
                print(f"⚠ No news found for {self.symbol}")
                return []

            self.news_data = []
            for article in news[:max_articles]:
                # Extract article info
                title = article.get('title', '')
                published = article.get('providerPublishTime', 0)

                # Analyze sentiment
                sentiment_scores = self.sentiment_analyzer.polarity_scores(title)

                self.news_data.append({
                    'title': title,
                    'published': datetime.fromtimestamp(published),
                    'sentiment_compound': sentiment_scores['compound'],
                    'sentiment_pos': sentiment_scores['pos'],
                    'sentiment_neg': sentiment_scores['neg'],
                    'sentiment_neu': sentiment_scores['neu']
                })

            print(f"✓ Fetched {len(self.news_data)} news articles for {self.symbol}")
            return self.news_data

        except Exception as e:
            print(f"⚠ Error fetching news for {self.symbol}: {e}")
            return []

    def calculate_sentiment_features(self):
        """
        Calculate sentiment-based features from news articles
        Adds sentiment features to each day in the historical data
        """
        if self.data is None:
            return None

        if not self.news_data:
            print("Fetching news for sentiment analysis...")
            self.fetch_news()

        # Initialize sentiment columns with neutral values
        self.data['News_Sentiment'] = 0.0
        self.data['News_Positive'] = 0.0
        self.data['News_Negative'] = 0.0
        self.data['News_Volume'] = 0
        self.data['Sentiment_Momentum'] = 0.0

        if not self.news_data:
            print("⚠ No news data available - using neutral sentiment")
            return self.data

        # Create a DataFrame from news for easier processing
        news_df = pd.DataFrame(self.news_data)
        news_df['date'] = pd.to_datetime(news_df['published']).dt.date

        # For each day in our stock data, calculate sentiment metrics
        for idx in self.data.index:
            date = idx.date()

            # Get news from last 7 days (rolling window)
            start_date = date - timedelta(days=7)
            recent_news = news_df[
                (news_df['date'] >= start_date) & (news_df['date'] <= date)
            ]

            if len(recent_news) > 0:
                # Average sentiment
                self.data.loc[idx, 'News_Sentiment'] = recent_news['sentiment_compound'].mean()
                self.data.loc[idx, 'News_Positive'] = recent_news['sentiment_pos'].mean()
                self.data.loc[idx, 'News_Negative'] = recent_news['sentiment_neg'].mean()
                self.data.loc[idx, 'News_Volume'] = len(recent_news)

        # Calculate sentiment momentum (change in sentiment over time)
        self.data['Sentiment_Momentum'] = self.data['News_Sentiment'].diff(5)

        # Fill any remaining NaN values with 0
        sentiment_cols = ['News_Sentiment', 'News_Positive', 'News_Negative',
                         'News_Volume', 'Sentiment_Momentum']
        self.data[sentiment_cols] = self.data[sentiment_cols].fillna(0)

        print("✓ Sentiment features calculated")
        return self.data

    def calculate_sma(self, window=20):
        """Calculate Simple Moving Average"""
        if self.data is None:
            return None
        self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
        return self.data[f'SMA_{window}']

    def calculate_ema(self, window=20):
        """Calculate Exponential Moving Average"""
        if self.data is None:
            return None
        self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        return self.data[f'EMA_{window}']

    def calculate_rsi(self, window=14):
        """
        Calculate Relative Strength Index (RSI)
        RSI > 70: Overbought (potential sell signal)
        RSI < 30: Oversold (potential buy signal)
        """
        if self.data is None:
            return None

        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self.data['RSI']

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        When MACD crosses above signal line: Buy signal
        When MACD crosses below signal line: Sell signal
        """
        if self.data is None:
            return None

        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()

        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal, adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']

        return self.data[['MACD', 'MACD_Signal', 'MACD_Histogram']]

    def calculate_bollinger_bands(self, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        Price near upper band: Potentially overbought
        Price near lower band: Potentially oversold
        """
        if self.data is None:
            return None

        sma = self.data['Close'].rolling(window=window).mean()
        std = self.data['Close'].rolling(window=window).std()

        self.data['BB_Upper'] = sma + (std * num_std)
        self.data['BB_Middle'] = sma
        self.data['BB_Lower'] = sma - (std * num_std)

        return self.data[['BB_Upper', 'BB_Middle', 'BB_Lower']]

    def calculate_volume_indicators(self):
        """Calculate volume-based indicators"""
        if self.data is None:
            return None

        # Volume moving average
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()

        # Volume ratio (current volume / average volume)
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']

        return self.data[['Volume_MA', 'Volume_Ratio']]

    def calculate_momentum(self, window=10):
        """Calculate price momentum"""
        if self.data is None:
            return None

        self.data['Momentum'] = self.data['Close'].diff(window)
        return self.data['Momentum']

    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        if self.data is None:
            self.fetch_data()

        if self.data is None:
            return None

        print("Calculating technical indicators...")

        # Moving averages
        self.calculate_sma(20)
        self.calculate_sma(50)
        self.calculate_ema(12)
        self.calculate_ema(26)

        # Momentum indicators
        self.calculate_rsi(14)
        self.calculate_macd()
        self.calculate_momentum(10)

        # Volatility indicators
        self.calculate_bollinger_bands()

        # Volume indicators
        self.calculate_volume_indicators()

        # Price change percentage
        self.data['Price_Change_Pct'] = self.data['Close'].pct_change() * 100

        # Average True Range (ATR) for volatility
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=14).mean()

        # === Enhanced Direction-Focused Features ===

        # Rate of Change (ROC) - multiple periods
        self.data['ROC_5'] = ((self.data['Close'] - self.data['Close'].shift(5)) /
                              self.data['Close'].shift(5)) * 100
        self.data['ROC_10'] = ((self.data['Close'] - self.data['Close'].shift(10)) /
                               self.data['Close'].shift(10)) * 100

        # Trend strength using moving average crossovers
        self.data['SMA_Cross'] = self.data['SMA_20'] - self.data['SMA_50']
        self.data['EMA_Cross'] = self.data['EMA_12'] - self.data['EMA_26']

        # Price position relative to Bollinger Bands (0-1 scale)
        bb_range = self.data['BB_Upper'] - self.data['BB_Lower']
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / bb_range

        # Directional Movement Index (DMI)
        high_diff = self.data['High'].diff()
        low_diff = -self.data['Low'].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        self.data['Plus_DM'] = pd.Series(plus_dm, index=self.data.index).rolling(window=14).mean()
        self.data['Minus_DM'] = pd.Series(minus_dm, index=self.data.index).rolling(window=14).mean()

        # Directional Indicator
        self.data['DI_Diff'] = self.data['Plus_DM'] - self.data['Minus_DM']

        # Stochastic Oscillator - shows momentum
        low_14 = self.data['Low'].rolling(window=14).min()
        high_14 = self.data['High'].rolling(window=14).max()
        self.data['Stochastic'] = 100 * (self.data['Close'] - low_14) / (high_14 - low_14)

        # Williams %R - another momentum indicator
        self.data['Williams_R'] = -100 * (high_14 - self.data['Close']) / (high_14 - low_14)

        # Commodity Channel Index (CCI) - identifies cyclical trends
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        self.data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

        # Money Flow Index (MFI) - volume-weighted RSI
        typical_price_mfi = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        money_flow = typical_price_mfi * self.data['Volume']

        positive_flow = pd.Series(0.0, index=self.data.index)
        negative_flow = pd.Series(0.0, index=self.data.index)

        for i in range(1, len(self.data)):
            if typical_price_mfi.iloc[i] > typical_price_mfi.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = money_flow.iloc[i]

        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        mfi_ratio = positive_mf / negative_mf
        self.data['MFI'] = 100 - (100 / (1 + mfi_ratio))

        # Price acceleration (second derivative)
        self.data['Price_Acceleration'] = self.data['Close'].diff().diff()

        # Volatility ratio
        self.data['Volatility_Ratio'] = self.data['ATR'] / self.data['Close']

        # === News Sentiment Features ===
        self.calculate_sentiment_features()

        print("✓ All indicators calculated (including enhanced direction features + news sentiment)")

        return self.data

    def get_latest_indicators(self):
        """Get the most recent indicator values"""
        if self.data is None or len(self.data) == 0:
            return None

        latest = self.data.iloc[-1]

        return {
            'symbol': self.symbol,
            'date': latest.name,
            'close_price': latest['Close'],
            'volume': latest['Volume'],
            'rsi': latest.get('RSI', None),
            'macd': latest.get('MACD', None),
            'macd_signal': latest.get('MACD_Signal', None),
            'sma_20': latest.get('SMA_20', None),
            'sma_50': latest.get('SMA_50', None),
            'bb_upper': latest.get('BB_Upper', None),
            'bb_lower': latest.get('BB_Lower', None),
            'volume_ratio': latest.get('Volume_Ratio', None),
            'momentum': latest.get('Momentum', None),
            'atr': latest.get('ATR', None),
            'news_sentiment': latest.get('News_Sentiment', 0),
            'news_volume': latest.get('News_Volume', 0),
            'sentiment_momentum': latest.get('Sentiment_Momentum', 0),
        }

    def get_recent_news(self, num_articles=5):
        """
        Get recent news articles with sentiment

        Args:
            num_articles: Number of recent articles to return

        Returns:
            List of recent news articles with sentiment
        """
        if not self.news_data:
            self.fetch_news()

        # Sort by published date (most recent first)
        sorted_news = sorted(self.news_data, key=lambda x: x['published'], reverse=True)

        return sorted_news[:num_articles]

    def get_summary(self):
        """Get a technical summary of the stock"""
        indicators = self.get_latest_indicators()
        if not indicators:
            return None

        summary = f"\n{'='*60}\n"
        summary += f"Technical Analysis Summary for {self.symbol}\n"
        summary += f"{'='*60}\n"
        summary += f"Date: {indicators['date'].strftime('%Y-%m-%d')}\n"
        summary += f"Price: ${indicators['close_price']:.2f}\n"
        summary += f"\n--- Momentum Indicators ---\n"
        summary += f"RSI (14): {indicators['rsi']:.2f} "

        if indicators['rsi'] > 70:
            summary += "(Overbought ⚠️)\n"
        elif indicators['rsi'] < 30:
            summary += "(Oversold 📈)\n"
        else:
            summary += "(Neutral)\n"

        summary += f"MACD: {indicators['macd']:.2f}\n"
        summary += f"MACD Signal: {indicators['macd_signal']:.2f}\n"

        if indicators['macd'] > indicators['macd_signal']:
            summary += "MACD Status: Bullish 📈\n"
        else:
            summary += "MACD Status: Bearish 📉\n"

        summary += f"\n--- Moving Averages ---\n"
        summary += f"20-day SMA: ${indicators['sma_20']:.2f}\n"
        summary += f"50-day SMA: ${indicators['sma_50']:.2f}\n"

        if indicators['close_price'] > indicators['sma_20'] > indicators['sma_50']:
            summary += "Trend: Strong Uptrend 📈\n"
        elif indicators['close_price'] < indicators['sma_20'] < indicators['sma_50']:
            summary += "Trend: Strong Downtrend 📉\n"
        else:
            summary += "Trend: Mixed\n"

        summary += f"\n--- Volatility ---\n"
        summary += f"Bollinger Upper: ${indicators['bb_upper']:.2f}\n"
        summary += f"Bollinger Lower: ${indicators['bb_lower']:.2f}\n"
        summary += f"ATR: {indicators['atr']:.2f}\n"

        summary += f"\n--- Volume ---\n"
        summary += f"Volume Ratio: {indicators['volume_ratio']:.2f}x average\n"

        summary += f"\n--- News Sentiment ---\n"
        sentiment = indicators['news_sentiment']
        summary += f"Sentiment Score: {sentiment:.3f} "

        if sentiment > 0.3:
            summary += "(Very Positive 🚀)\n"
        elif sentiment > 0.1:
            summary += "(Positive 📈)\n"
        elif sentiment < -0.3:
            summary += "(Very Negative 📉)\n"
        elif sentiment < -0.1:
            summary += "(Negative ⚠️)\n"
        else:
            summary += "(Neutral ➡️)\n"

        summary += f"News Volume (7d): {int(indicators['news_volume'])} articles\n"
        summary += f"Sentiment Trend: {indicators['sentiment_momentum']:+.3f}\n"

        # Show recent headlines
        if self.news_data:
            recent = self.get_recent_news(3)
            summary += f"\nRecent Headlines:\n"
            for i, article in enumerate(recent, 1):
                sentiment_label = "📈" if article['sentiment_compound'] > 0 else "📉" if article['sentiment_compound'] < 0 else "➡️"
                summary += f"  {i}. {sentiment_label} {article['title'][:60]}...\n"

        summary += f"{'='*60}\n"

        return summary


if __name__ == "__main__":
    # Example usage
    fetcher = StockDataFetcher('AAPL', period='6mo')
    fetcher.calculate_all_indicators()
    print(fetcher.get_summary())
