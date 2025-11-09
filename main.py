"""
Main entry point for stock prediction system
Provides an easy-to-use interface for training and predicting stock prices
"""

import argparse
from stock_predictor import StockPredictor
from stock_data import StockDataFetcher


def train_and_predict(ticker, period='2y', epochs=50, sequence_length=60):
    """
    Train model and make predictions

    Args:
        ticker: Stock ticker symbol
        period: Historical data period
        epochs: Number of training epochs
        sequence_length: Sequence length for LSTM
    """
    print(f"\n{'='*70}")
    print(f"Stock Price Predictor - {ticker}")
    print(f"{'='*70}\n")

    # Initialize predictor
    predictor = StockPredictor(ticker, sequence_length=sequence_length)

    # Prepare data
    print("Step 1: Fetching and preparing data...")
    predictor.prepare_data(period=period)

    # Display current technical analysis
    print("\nCurrent Technical Analysis:")
    print(predictor.data_fetcher.get_summary())

    # Train model
    print(f"\nStep 2: Training LSTM model ({epochs} epochs)...")
    predictor.train(epochs=epochs, batch_size=32, learning_rate=0.001, verbose=True)

    # Evaluate
    print("\nStep 3: Evaluating model performance...")
    metrics = predictor.evaluate()

    # Predict next day
    print("\nStep 4: Making prediction...")
    next_day_price, direction, confidence = predictor.predict_next_day()
    current_price = predictor.df['Close'].iloc[-1]
    price_change = next_day_price - current_price
    percent_change = (price_change / current_price) * 100

    # Final summary
    print(f"\n{'='*70}")
    print(f"PREDICTION SUMMARY FOR {ticker}")
    print(f"{'='*70}")
    print(f"\nCurrent Status:")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Predicted Next Day: ${next_day_price:.2f}")
    print(f"  Expected Change: ${price_change:.2f} ({percent_change:+.2f}%)")
    print(f"  Predicted Direction: {direction} (Confidence: {confidence:.1%})")

    if direction == "UP":
        print(f"  Signal: 🟢 BUY (Bullish)")
    else:
        print(f"  Signal: 🔴 SELL (Bearish)")

    print(f"\nModel Performance:")
    print(f"  RMSE: ${metrics['rmse']:.2f}")
    print(f"  MAE: ${metrics['mae']:.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
    print(f"  Price-based Dir Accuracy: {metrics['price_direction_accuracy']:.2f}%")

    print(f"\n{'='*70}\n")

    # Plot results
    print("Generating visualizations...")
    predictor.plot_training_history()
    predictor.plot_predictions(num_days=100)

    # Save model
    print(f"\nSaving model...")
    predictor.save_model()

    print(f"\nAll done! ✅")

    return predictor


def quick_analysis(ticker, period='6mo'):
    """
    Quick technical analysis without training

    Args:
        ticker: Stock ticker symbol
        period: Historical data period
    """
    print(f"\n{'='*70}")
    print(f"Quick Technical Analysis - {ticker}")
    print(f"{'='*70}\n")

    # Fetch data
    fetcher = StockDataFetcher(symbol=ticker, period=period)
    fetcher.fetch_data()
    fetcher.calculate_all_indicators()

    # Display summary
    print(fetcher.get_summary())

    # Display recent prices
    print("\nRecent Price History (Last 10 Days):")
    print(f"{'='*70}")

    recent = fetcher.data[['Close', 'Volume', 'RSI', 'MACD']].tail(10)
    print(recent.to_string())

    print(f"\n{'='*70}\n")


def compare_stocks(tickers, period='1y'):
    """
    Compare multiple stocks

    Args:
        tickers: List of stock ticker symbols
        period: Historical data period
    """
    print(f"\n{'='*70}")
    print(f"Comparing Stocks: {', '.join(tickers)}")
    print(f"{'='*70}\n")

    results = []

    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        fetcher = StockDataFetcher(symbol=ticker, period=period)
        fetcher.fetch_data()
        fetcher.calculate_all_indicators()

        latest = fetcher.get_latest_indicators()

        results.append({
            'Ticker': ticker,
            'Price': latest['Close'],
            'RSI': latest.get('RSI', 0),
            'MACD': latest.get('MACD', 0),
            'Volume_Ratio': latest.get('Volume_Ratio', 0)
        })

    # Display comparison
    print(f"\n{'='*70}")
    print(f"Comparison Summary:")
    print(f"{'='*70}\n")

    print(f"{'Ticker':<10} {'Price':<12} {'RSI':<10} {'MACD':<12} {'Vol Ratio':<10}")
    print(f"{'-'*70}")

    for r in results:
        print(f"{r['Ticker']:<10} ${r['Price']:<11.2f} {r['RSI']:<10.2f} {r['MACD']:<12.4f} {r['Volume_Ratio']:<10.2f}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Stock Price Predictor')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'analyze', 'compare'],
                        help='Operation mode')

    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol')

    parser.add_argument('--tickers', type=str, nargs='+',
                        help='Multiple tickers for comparison mode')

    parser.add_argument('--period', type=str, default='2y',
                        help='Historical data period (e.g., 1mo, 6mo, 1y, 2y)')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')

    parser.add_argument('--sequence-length', type=int, default=60,
                        help='Sequence length for LSTM')

    args = parser.parse_args()

    if args.mode == 'train':
        train_and_predict(
            ticker=args.ticker,
            period=args.period,
            epochs=args.epochs,
            sequence_length=args.sequence_length
        )

    elif args.mode == 'analyze':
        quick_analysis(ticker=args.ticker, period=args.period)

    elif args.mode == 'compare':
        tickers = args.tickers if args.tickers else ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        compare_stocks(tickers=tickers, period=args.period)


if __name__ == "__main__":
    # Example runs if no arguments provided
    import sys

    if len(sys.argv) == 1:
        print("Running example predictions...")
        print("\nExample 1: Train and predict for AAPL")
        train_and_predict('AAPL', period='1y', epochs=30)
    else:
        main()
