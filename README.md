# Stock Price Predictor

A machine learning-based stock price prediction system using LSTM neural networks and technical indicators.

## Features

- **Technical Analysis**: Calculate 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **LSTM Neural Network**: Deep learning model for time series prediction
- **Multiple Modes**: Train models, quick analysis, or compare stocks
- **Visualization**: Training history and prediction plots
- **Model Persistence**: Save and load trained models

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train and Predict

Train a model and predict next day's price:

```bash
python main.py --mode train --ticker AAPL --epochs 50
```

### 2. Quick Analysis

Get technical analysis without training:

```bash
python main.py --mode analyze --ticker AAPL --period 6mo
```

### 3. Compare Stocks

Compare multiple stocks:

```bash
python main.py --mode compare --tickers AAPL GOOGL MSFT TSLA
```

## Usage Examples

### Basic Prediction

```python
from stock_predictor import StockPredictor

# Initialize
predictor = StockPredictor('AAPL', sequence_length=60)

# Prepare data
predictor.prepare_data(period='2y')

# Train
predictor.train(epochs=50, batch_size=32)

# Evaluate
metrics = predictor.evaluate()

# Predict next day
next_price = predictor.predict_next_day()
print(f"Predicted price: ${next_price:.2f}")

# Save model
predictor.save_model('aapl_model.pth')
```

### Technical Analysis Only

```python
from stock_data import StockDataFetcher

# Fetch data
fetcher = StockDataFetcher('AAPL', period='1y')
fetcher.fetch_data()
fetcher.calculate_all_indicators()

# Get summary
print(fetcher.get_summary())

# Get latest indicators
latest = fetcher.get_latest_indicators()
```

## Command Line Arguments

```
--mode           : Operation mode (train, analyze, compare)
--ticker         : Stock ticker symbol (default: AAPL)
--tickers        : Multiple tickers for comparison
--period         : Historical data period (1mo, 6mo, 1y, 2y, etc.)
--epochs         : Number of training epochs (default: 50)
--sequence-length: LSTM sequence length (default: 60)
```

## Model Architecture

The predictor uses a 2-layer LSTM network with:
- Input: 20 technical indicators
- Hidden layers: 128 units each
- Dropout: 0.2 for regularization
- Output: Single price prediction
- Optimizer: Adam with learning rate scheduling

## Technical Indicators

The system calculates the following indicators:

- **Moving Averages**: SMA(20, 50), EMA(12, 26)
- **Momentum**: RSI, MACD, Price Momentum
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume MA, Volume Ratio

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Direction Accuracy**: Percentage of correct up/down predictions

## Output Files

After training, the following files are generated:

- `{TICKER}_model.pth`: Saved model checkpoint
- `{TICKER}_training_history.png`: Training/validation loss plot
- `{TICKER}_predictions.png`: Predictions vs actual prices

## Examples

### Train TSLA with custom parameters

```bash
python main.py --mode train --ticker TSLA --period 1y --epochs 100 --sequence-length 90
```

### Analyze GOOGL

```bash
python main.py --mode analyze --ticker GOOGL
```

### Compare Tech Stocks

```bash
python main.py --mode compare --tickers AAPL MSFT GOOGL META AMZN
```

## Project Structure

```
stock-predictor/
├── stock_data.py       # Data fetching and technical indicators
├── stock_predictor.py  # LSTM model and prediction logic
├── main.py            # Command-line interface
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib

## Notes

- This is for educational purposes only
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- The model's predictions should not be used as sole basis for trading decisions

## Future Enhancements

- [ ] Sentiment analysis integration
- [ ] Real-time predictions
- [ ] Portfolio optimization
- [ ] Web dashboard
- [ ] More advanced models (Transformers, GRU)
- [ ] Automated backtesting

## License

MIT License
