"""
Quick test of news sentiment analysis feature
"""
from stock_data import StockDataFetcher

print("="*70)
print("Testing News Sentiment Analysis")
print("="*70)

# Test with AAPL
ticker = "AAPL"
print(f"\nFetching data and news for {ticker}...")

fetcher = StockDataFetcher(symbol=ticker, period='3mo')
fetcher.fetch_data()
fetcher.calculate_all_indicators()

print("\n" + "="*70)
print("Recent News with Sentiment Scores")
print("="*70)

recent_news = fetcher.get_recent_news(5)
if recent_news:
    for i, article in enumerate(recent_news, 1):
        sentiment = article['sentiment_compound']
        sentiment_label = "📈 POSITIVE" if sentiment > 0.1 else "📉 NEGATIVE" if sentiment < -0.1 else "➡️ NEUTRAL"
        print(f"\n{i}. {sentiment_label} (Score: {sentiment:+.3f})")
        print(f"   {article['title']}")
        print(f"   Published: {article['published'].strftime('%Y-%m-%d %H:%M')}")
else:
    print("No news data available")

print("\n" + "="*70)
print("Technical Summary (with Sentiment)")
print("="*70)
print(fetcher.get_summary())

print("\n✅ News sentiment analysis is working!")
