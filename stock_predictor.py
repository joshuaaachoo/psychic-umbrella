"""
Stock Price Predictor using Machine Learning
Implements LSTM neural network for time series prediction with technical indicators
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from stock_data import StockDataFetcher


class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences with direction labels"""

    def __init__(self, sequences, targets, directions):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.directions = torch.LongTensor(directions)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.directions[idx]


class LSTMPredictor(nn.Module):
    """LSTM Neural Network for stock price prediction with multi-task learning"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Shared fully connected layer
        self.fc_shared = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Price regression head
        self.fc_price = nn.Linear(64, 1)

        # Direction classification head (binary: up or down)
        self.fc_direction = nn.Linear(64, 2)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Shared layer
        shared = self.fc_shared(last_output)
        shared = self.relu(shared)
        shared = self.dropout(shared)

        # Price prediction
        price_out = self.fc_price(shared)

        # Direction prediction (logits for binary classification)
        direction_out = self.fc_direction(shared)

        return price_out, direction_out


class StockPredictor:
    """
    Main class for stock price prediction
    """

    def __init__(self, ticker, sequence_length=60, test_size=0.2):
        """
        Args:
            ticker: Stock ticker symbol
            sequence_length: Number of days to use for prediction
            test_size: Proportion of data to use for testing
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.test_size = test_size

        self.model = None
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()  # Separate scaler for target prices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_columns = None
        self.data_fetcher = None
        self.df = None

    def prepare_data(self, period='2y'):
        """
        Fetch and prepare data for training

        Args:
            period: Time period for historical data
        """
        print(f"Fetching data for {self.ticker}...")

        # Fetch stock data with indicators
        self.data_fetcher = StockDataFetcher(symbol=self.ticker, period=period)
        self.df = self.data_fetcher.fetch_data()
        self.df = self.data_fetcher.calculate_all_indicators()

        # Drop NaN values
        self.df = self.df.dropna()

        # Select features for prediction (including new direction-focused features + sentiment)
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'ATR', 'Volume_MA', 'Volume_Ratio', 'Momentum',
            # New direction-focused features
            'ROC_5', 'ROC_10', 'SMA_Cross', 'EMA_Cross',
            'BB_Position', 'Plus_DM', 'Minus_DM', 'DI_Diff',
            'Stochastic', 'Williams_R', 'CCI', 'MFI',
            'Price_Acceleration', 'Volatility_Ratio',
            # News sentiment features
            'News_Sentiment', 'News_Positive', 'News_Negative',
            'News_Volume', 'Sentiment_Momentum'
        ]

        print(f"Data prepared: {len(self.df)} samples with {len(self.feature_columns)} features")

        return self.df

    def create_sequences(self, data, target_col='Close'):
        """
        Create sequences for LSTM training with direction labels

        Args:
            data: DataFrame with features
            target_col: Column to predict

        Returns:
            X, y, directions: Sequences, price targets, and direction labels
        """
        sequences = []
        targets = []
        directions = []

        # Get feature data
        feature_data = data[self.feature_columns].values
        target_data = data[target_col].values.reshape(-1, 1)

        # Normalize features AND targets
        scaled_features = self.scaler.fit_transform(feature_data)
        scaled_targets = self.target_scaler.fit_transform(target_data)

        # Get raw prices for direction calculation
        raw_prices = data[target_col].values

        # Create sequences
        for i in range(len(scaled_features) - self.sequence_length):
            seq = scaled_features[i:i + self.sequence_length]
            target = scaled_targets[i + self.sequence_length]

            # Calculate direction: 1 if price goes up, 0 if down
            current_price = raw_prices[i + self.sequence_length - 1]
            next_price = raw_prices[i + self.sequence_length]
            direction = 1 if next_price > current_price else 0

            sequences.append(seq)
            targets.append(target[0])  # Extract scalar from array
            directions.append(direction)

        return np.array(sequences), np.array(targets), np.array(directions)

    def build_model(self, input_size):
        """Build the LSTM model"""
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        ).to(self.device)

        print(f"Model built and moved to {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        return self.model

    def train(self, epochs=50, batch_size=32, learning_rate=0.001, direction_weight=2.0, verbose=True):
        """
        Train the model with multi-task learning (price + direction)

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            direction_weight: Weight for direction loss (higher = more emphasis on direction)
            verbose: Print training progress
        """
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Create sequences with direction labels
        X, y, directions = self.create_sequences(self.df)

        # Split data
        X_train, X_test, y_train, y_test, dir_train, dir_test = train_test_split(
            X, y, directions, test_size=self.test_size, shuffle=False
        )

        # Create datasets
        train_dataset = StockDataset(X_train, y_train, dir_train)
        test_dataset = StockDataset(X_test, y_test, dir_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Build model if not exists
        if self.model is None:
            self.build_model(input_size=X.shape[2])

        # Loss functions
        price_criterion = nn.MSELoss()
        direction_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        train_losses = []
        test_losses = []
        train_dir_accs = []
        test_dir_accs = []

        print(f"\nTraining on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")
        print(f"Multi-task learning: Price prediction + Direction classification")
        print(f"Direction weight: {direction_weight}x")
        print(f"{'='*60}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_dir_correct = 0
            train_total = 0

            for sequences, targets, dir_labels in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                dir_labels = dir_labels.to(self.device)

                # Forward pass
                price_outputs, direction_outputs = self.model(sequences)

                # Calculate losses
                price_loss = price_criterion(price_outputs.squeeze(), targets)
                direction_loss = direction_criterion(direction_outputs, dir_labels)

                # Combined loss with weighting
                loss = price_loss + direction_weight * direction_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calculate direction accuracy
                _, predicted_dirs = torch.max(direction_outputs, 1)
                train_dir_correct += (predicted_dirs == dir_labels).sum().item()
                train_total += dir_labels.size(0)

            train_loss /= len(train_loader)
            train_dir_acc = 100 * train_dir_correct / train_total
            train_losses.append(train_loss)
            train_dir_accs.append(train_dir_acc)

            # Validation
            self.model.eval()
            test_loss = 0.0
            test_dir_correct = 0
            test_total = 0

            with torch.no_grad():
                for sequences, targets, dir_labels in test_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    dir_labels = dir_labels.to(self.device)

                    price_outputs, direction_outputs = self.model(sequences)

                    price_loss = price_criterion(price_outputs.squeeze(), targets)
                    direction_loss = direction_criterion(direction_outputs, dir_labels)
                    loss = price_loss + direction_weight * direction_loss

                    test_loss += loss.item()

                    # Calculate direction accuracy
                    _, predicted_dirs = torch.max(direction_outputs, 1)
                    test_dir_correct += (predicted_dirs == dir_labels).sum().item()
                    test_total += dir_labels.size(0)

            test_loss /= len(test_loader)
            test_dir_acc = 100 * test_dir_correct / test_total
            test_losses.append(test_loss)
            test_dir_accs.append(test_dir_acc)

            # Learning rate scheduling
            scheduler.step(test_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}/{test_loss:.4f} - "
                      f"Dir Acc: {train_dir_acc:.1f}%/{test_dir_acc:.1f}%")

        print(f"{'='*60}")
        print(f"Training completed!")
        print(f"Final Test Direction Accuracy: {test_dir_acc:.2f}%")

        # Store losses and accuracies for plotting
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.train_dir_accs = train_dir_accs
        self.test_dir_accs = test_dir_accs
        self.X_test = X_test
        self.y_test = y_test
        self.dir_test = dir_test

        return train_losses, test_losses

    def predict(self, sequence):
        """
        Make prediction on a sequence

        Args:
            sequence: Input sequence (should be normalized)

        Returns:
            Tuple of (predicted_price, predicted_direction, direction_confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            price_pred, direction_pred = self.model(sequence_tensor)

        # Denormalize the price prediction
        prediction_scaled = price_pred.cpu().numpy().reshape(-1, 1)
        prediction_actual = self.target_scaler.inverse_transform(prediction_scaled)

        # Get direction prediction
        direction_probs = torch.softmax(direction_pred, dim=1)
        direction_class = torch.argmax(direction_probs, dim=1).cpu().numpy()[0]
        direction_confidence = direction_probs[0][direction_class].cpu().numpy()

        return prediction_actual[0][0], direction_class, direction_confidence

    def predict_next_day(self):
        """
        Predict next day's closing price and direction

        Returns:
            Tuple of (predicted_price, predicted_direction, direction_confidence)
            where predicted_direction is "UP" or "DOWN"
        """
        if self.df is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Get last sequence
        feature_data = self.df[self.feature_columns].values
        scaled_features = self.scaler.transform(feature_data)
        last_sequence = scaled_features[-self.sequence_length:]

        # Predict
        price, direction, confidence = self.predict(last_sequence)
        direction_label = "UP" if direction == 1 else "DOWN"

        return price, direction_label, confidence

    def evaluate(self):
        """Evaluate model performance with improved direction accuracy"""
        if self.model is None or not hasattr(self, 'X_test'):
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        predictions_scaled = []
        direction_predictions = []

        with torch.no_grad():
            for i in range(len(self.X_test)):
                seq = torch.FloatTensor(self.X_test[i]).unsqueeze(0).to(self.device)
                price_pred, dir_pred = self.model(seq)
                predictions_scaled.append(price_pred.cpu().numpy()[0][0])

                # Get direction prediction
                _, predicted_dir = torch.max(dir_pred, 1)
                direction_predictions.append(predicted_dir.cpu().numpy()[0])

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        actual_scaled = self.y_test.reshape(-1, 1)

        # Denormalize both predictions and actual values
        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        actual = self.target_scaler.inverse_transform(actual_scaled).flatten()

        # Calculate price metrics
        mse = np.mean((predictions - actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        # Direction accuracy (from classification head)
        direction_predictions = np.array(direction_predictions)
        actual_directions = self.dir_test
        direction_accuracy = np.mean(direction_predictions == actual_directions) * 100

        # Additional metric: Price-based direction accuracy
        # This compares if predicted price change matches actual price change direction
        price_direction_actual = np.diff(actual) > 0
        price_direction_pred = np.diff(predictions) > 0
        price_direction_accuracy = np.mean(price_direction_actual == price_direction_pred) * 100

        print(f"\n{'='*60}")
        print(f"Model Evaluation Metrics")
        print(f"{'='*60}")
        print(f"Price Prediction:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"\nDirection Prediction:")
        print(f"  Classification Accuracy: {direction_accuracy:.2f}%")
        print(f"  Price-based Accuracy: {price_direction_accuracy:.2f}%")
        print(f"{'='*60}\n")

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'price_direction_accuracy': price_direction_accuracy,
            'predictions': predictions,
            'actual': actual
        }

    def plot_training_history(self):
        """Plot training and validation loss and direction accuracy"""
        if not hasattr(self, 'train_losses'):
            raise ValueError("No training history available")

        fig = plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Combined Loss')
        plt.title('Training History - Loss')
        plt.legend()
        plt.grid(True)

        # Loss plot (log scale)
        plt.subplot(1, 3, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Combined Loss')
        plt.yscale('log')
        plt.title('Training History - Loss (Log Scale)')
        plt.legend()
        plt.grid(True)

        # Direction accuracy plot
        plt.subplot(1, 3, 3)
        if hasattr(self, 'train_dir_accs'):
            plt.plot(self.train_dir_accs, label='Train Dir Acc')
            plt.plot(self.test_dir_accs, label='Test Dir Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Direction Accuracy (%)')
            plt.title('Direction Classification Accuracy')
            plt.legend()
            plt.grid(True)
            plt.ylim([0, 100])

        plt.tight_layout()
        plt.savefig(f'{self.ticker}_training_history.png', dpi=150, bbox_inches='tight')
        print(f"Training history saved to {self.ticker}_training_history.png")
        plt.show()

    def plot_predictions(self, num_days=100):
        """Plot predictions vs actual prices"""
        metrics = self.evaluate()
        predictions = metrics['predictions']
        actual = metrics['actual']

        # Plot last num_days
        predictions = predictions[-num_days:]
        actual = actual[-num_days:]

        plt.figure(figsize=(14, 6))
        plt.plot(actual, label='Actual Price', linewidth=2)
        plt.plot(predictions, label='Predicted Price', linewidth=2, alpha=0.7)
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.title(f'{self.ticker} - Price Predictions vs Actual (Last {num_days} days)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_predictions.png', dpi=150, bbox_inches='tight')
        print(f"Predictions plot saved to {self.ticker}_predictions.png")
        plt.show()

    def save_model(self, filepath=None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")

        if filepath is None:
            filepath = f'{self.ticker}_model.pth'

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }, filepath)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.scaler = checkpoint['scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.feature_columns = checkpoint['feature_columns']
        self.sequence_length = checkpoint['sequence_length']

        # Build and load model
        input_size = len(self.feature_columns)
        self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"

    print(f"Stock Price Predictor for {ticker}")
    print(f"{'='*60}\n")

    # Initialize predictor
    predictor = StockPredictor(ticker, sequence_length=60)

    # Prepare data
    predictor.prepare_data(period='2y')

    # Train model
    predictor.train(epochs=50, batch_size=32, learning_rate=0.001)

    # Evaluate
    predictor.evaluate()

    # Predict next day
    next_day_price, direction, confidence = predictor.predict_next_day()
    current_price = predictor.df['Close'].iloc[-1]

    print(f"\nPrediction Summary:")
    print(f"{'='*60}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Next Day Price: ${next_day_price:.2f}")
    print(f"Expected Change: ${next_day_price - current_price:.2f} ({((next_day_price - current_price) / current_price * 100):.2f}%)")
    print(f"Predicted Direction: {direction} (Confidence: {confidence:.1%})")
    print(f"{'='*60}\n")

    # Plot results
    predictor.plot_training_history()
    predictor.plot_predictions(num_days=100)

    # Save model
    predictor.save_model()
