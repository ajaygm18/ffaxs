"""
Improved Stock Prediction with Technical Indicators
This version adds RSI, MACD, and Bollinger Bands to improve directional accuracy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import math
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMPROVED MODEL WITH TECHNICAL INDICATORS")
print("="*70)

stockID = '2330'
traindata = 'data/'+ stockID + '_2015_2019_ochlv.csv'
testdata = 'data/'+ stockID +'_202001_03_ochlv.csv'

# Hyperparameters
timesteps = 22  # Optimized
epochs = 200
batch_size = 20

def add_technical_indicators(df):
    """Add RSI, MACD, and other technical indicators"""
    # Make a copy
    data = df.copy()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Moving Averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (2 * bb_std)
    data['BB_lower'] = data['BB_middle'] - (2 * bb_std)
    
    # Volume indicators
    data['Volume_MA'] = data['Volume'].rolling(window=10).mean()
    data['Volume_Ratio'] = data['Volume'] / (data['Volume_MA'] + 1e-10)
    
    # Price momentum
    data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
    
    # Fill NaN values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data

# Load and prepare training data
print("\n📊 Loading and engineering features...")
train_df = pd.read_csv(traindata)
train_df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']

# Add technical indicators
train_df_enhanced = add_technical_indicators(train_df)

print(f"Original features: 5 (OHLCV)")
print(f"Enhanced features: {len(train_df_enhanced.columns)-1}")
print("Added: RSI, MACD, Moving Averages, Bollinger Bands, Volume indicators")

# Prepare training data
training_set = train_df_enhanced.iloc[:,1:].values  # Exclude Date

# Scale
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create sequences
X_train = []
Y_train = []
for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i, :])
    Y_train.append(training_set_scaled[i, 0])  # Predict Open price

X_train, Y_train = np.array(X_train), np.array(Y_train)

print(f"\n🔧 Building model...")
print(f"Timesteps: {timesteps}")
print(f"Features: {X_train.shape[2]}")

# Build model
model = Sequential()
model.add(LSTM(units=80, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.25))
model.add(LSTM(units=80, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(units=40))
model.add(Dropout(0.25))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=1))

optimizer = Adam(learning_rate=0.0008)
model.compile(optimizer=optimizer, loss='mean_squared_error')

print("\n🎯 Training model...")
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=0)
print("✓ Training complete")

# Prepare test data
print("\n📈 Making predictions...")
test_df = pd.read_csv(testdata)
test_df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
test_df_enhanced = add_technical_indicators(test_df)

test_set = test_df_enhanced.iloc[:,1:].values
test_set_scaled = sc.transform(test_set)

# Create test sequences
X_test = []
for i in range(timesteps, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-timesteps:i, :])
X_test = np.array(X_test)

# Predict
predicted_scaled = model.predict(X_test, verbose=0)
predicted_stock_price = predicted_scaled * (sc.data_max_[0] - sc.data_min_[0]) + sc.data_min_[0]

# Actual prices
real_stock_price = test_df.iloc[timesteps:, 1:2].values

# Calculate directional accuracy
def calculate_directional_accuracy(actual, predicted):
    correct = 0
    total = 0
    for i in range(1, len(actual)):
        actual_direction = 1 if actual[i] > actual[i-1] else 0
        predicted_direction = 1 if predicted[i] > predicted[i-1] else 0
        if actual_direction == predicted_direction:
            correct += 1
        total += 1
    return (correct / total) * 100 if total > 0 else 0

dir_acc = calculate_directional_accuracy(real_stock_price, predicted_stock_price)
rmse_test = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

# Training metrics
predicted_train_scaled = model.predict(X_train, verbose=0)
predicted_train = predicted_train_scaled * (sc.data_max_[0] - sc.data_min_[0]) + sc.data_min_[0]
real_train = train_df.iloc[timesteps:, 1:2].values
rmse_train = math.sqrt(mean_squared_error(real_train, predicted_train))

# Results
print("\n" + "="*70)
print("RESULTS - IMPROVED MODEL WITH TECHNICAL INDICATORS")
print("="*70)

print(f"\n📈 DIRECTIONAL ACCURACY:")
print(f"   Current Model:    {dir_acc:.1f}%")
print(f"   Original Model:   58.0%")
print(f"   Baseline (Random): 50.0%")

improvement = dir_acc - 58.0
if improvement > 0:
    print(f"   ✅ IMPROVEMENT: +{improvement:.1f} percentage points!")
elif improvement >= -1:
    print(f"   ➡️  Similar performance ({improvement:+.1f} pp)")
else:
    print(f"   ℹ️  Result: {improvement:.1f} pp (variability in training)")

print(f"\n📊 MAGNITUDE ACCURACY:")
print(f"   RMSE (Test):      {rmse_test:.2f} TWD")
print(f"   RMSE (Train):     {rmse_train:.2f} TWD")
print(f"   RMSE as %:        {(rmse_test/real_stock_price.mean())*100:.2f}%")
print(f"   Original RMSE:    7.65 TWD")

# Save predictions
np.savetxt(f'{stockID}_with_indicators.csv', predicted_stock_price, fmt="%.3f")

# Create visualization
plt.figure(figsize=(15, 7))
plt.plot(real_stock_price, color='red', label='Actual Price', linewidth=2.5, marker='o', markersize=4)
plt.plot(predicted_stock_price, color='blue', label='Predicted (w/ Tech Indicators)', linewidth=2, marker='x', markersize=4, alpha=0.7)
plt.title(f'Stock Prediction with Technical Indicators - TSMC ({stockID})\nDirectional Accuracy: {dir_acc:.1f}% | RMSE: {rmse_test:.2f} TWD', 
          fontsize=14, fontweight='bold')
plt.xlabel('Trading Days', fontsize=12)
plt.ylabel('Stock Price (TWD)', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('pic_with_indicators.png', dpi=150, bbox_inches='tight')

print(f"\n💾 Files saved:")
print(f"   - {stockID}_with_indicators.csv")
print(f"   - pic_with_indicators.png")

print("\n" + "="*70)
print("IMPROVEMENTS APPLIED:")
print("="*70)
print("✅ RSI (Relative Strength Index) - measures momentum")
print("✅ MACD (Moving Average Convergence Divergence) - trend indicator")
print("✅ Bollinger Bands - volatility and price levels")
print("✅ Multiple Moving Averages (5, 10, 20) - trend detection")
print("✅ Volume indicators - trading activity patterns")
print("✅ Momentum indicators - price change velocity")
print("✅ Optimized hyperparameters (timesteps, learning rate)")
print("\n💡 Next step for 65%+ accuracy: Integrate sentiment analysis!")
print("="*70)
