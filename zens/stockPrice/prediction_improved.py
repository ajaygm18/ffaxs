import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math

print("=" * 60)
print("IMPROVED STOCK PREDICTION MODEL - Enhanced Directional Accuracy")
print("=" * 60)

stockID = '2330'
traindata = 'data/'+ stockID + '_2015_2019_ochlv.csv'
testdata = 'data/'+ stockID +'_202001_03_ochlv.csv'
print(f'\nTraining data: {traindata}')
print(f'Test data: {testdata}')

# Enhanced hyperparameters
dataNum = 5
timesteps = 30  # Increased from 20 to capture longer patterns
epochNum = 300  # Increased for better learning
batch_size = 16  # Reduced for better generalization

print(f'\nHyperparameters:')
print(f'  - Timesteps: {timesteps} (increased from 20)')
print(f'  - Epochs: {epochNum} (increased from 200)')
print(f'  - Batch size: {batch_size} (reduced from 32)')

# Load and prepare training data
dataset_train = pd.read_csv(traindata)
training_set = dataset_train.iloc[:,1:dataNum+1].values

# Feature Engineering: Add technical indicators
def add_technical_indicators(df_prices):
    """Add technical indicators to improve directional accuracy"""
    df = pd.DataFrame(df_prices, columns=['Open', 'Close', 'High', 'Low', 'Volume'])
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Rate of change
    df['ROC_5'] = df['Close'].pct_change(periods=5)
    df['ROC_10'] = df['Close'].pct_change(periods=10)
    
    # Volatility (standard deviation)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Volume rate of change
    df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
    
    # High-Low range
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Replace NaN with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df.values

print('\nAdding technical indicators for better direction prediction...')
training_set_enhanced = add_technical_indicators(training_set)
enhanced_features = training_set_enhanced.shape[1]
print(f'  - Original features: {dataNum}')
print(f'  - Enhanced features: {enhanced_features}')

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set_enhanced)

# Prepare training sequences
X_train = []
Y_train = []
for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i, :])    
    Y_train.append(training_set_scaled[i, 0])  # Predict Open price
X_train, Y_train = np.array(X_train), np.array(Y_train)

print(f'\nTraining set shape: {X_train.shape}')

# Build Enhanced LSTM Model
print('\nBuilding enhanced LSTM model...')
regressor = Sequential()

# Bidirectional LSTM layers for better pattern recognition
regressor.add(Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], enhanced_features))))
regressor.add(Dropout(0.2))

regressor.add(Bidirectional(LSTM(units=64, return_sequences=True)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=32, return_sequences=False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=16, activation='relu'))
regressor.add(Dense(units=1))

# Custom optimizer with learning rate
optimizer = Adam(learning_rate=0.001)
regressor.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

print('\nModel Architecture:')
regressor.summary()

# Callbacks for better training
early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)

print('\nTraining enhanced model...')
print('(Using early stopping and learning rate reduction)')
history = regressor.fit(
    X_train, Y_train, 
    batch_size=batch_size, 
    epochs=epochNum,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Prepare test data
test_set = pd.read_csv(testdata)
real_stock_price_original = test_set.iloc[:,1:dataNum+1].values

# Add technical indicators to test set
test_set_enhanced = add_technical_indicators(real_stock_price_original)
inputs = sc.transform(test_set_enhanced)

# Prepare test sequences
inputs_test = []
for i in range(timesteps, len(inputs)):
    inputs_test.append(inputs[i-timesteps:i, :])
inputs_test = np.array(inputs_test)

# Make predictions
predicted_stock_price_scaled = regressor.predict(inputs_test)

# Inverse transform
predicted_stock_price = predicted_stock_price_scaled * (sc.data_max_[0] - sc.data_min_[0]) + sc.data_min_[0]
real_stock_price = test_set.iloc[timesteps:,1:2].values

# Calculate metrics
rmseTest = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

# Calculate Directional Accuracy
def calculate_directional_accuracy(actual, predicted):
    """Calculate how often we predict the correct direction"""
    correct = 0
    total = 0
    
    for i in range(1, len(actual)):
        actual_direction = 1 if actual[i] > actual[i-1] else -1
        predicted_direction = 1 if predicted[i] > predicted[i-1] else -1
        
        if actual_direction == predicted_direction:
            correct += 1
        total += 1
    
    return (correct / total) * 100 if total > 0 else 0

directional_accuracy = calculate_directional_accuracy(real_stock_price, predicted_stock_price)

# Training set predictions
predicted_stock_price_train_scaled = regressor.predict(X_train)
predicted_stock_price_train = predicted_stock_price_train_scaled * (sc.data_max_[0] - sc.data_min_[0]) + sc.data_min_[0]
real_stock_price_train = dataset_train.iloc[timesteps:,1:2].values
rmseTrain = math.sqrt(mean_squared_error(real_stock_price_train, predicted_stock_price_train))

# Print Results
print("\n" + "=" * 60)
print("IMPROVED MODEL RESULTS")
print("=" * 60)
print(f"\n📊 Magnitude Accuracy:")
print(f"   RMSE (Test):  {rmseTest:.2f} TWD")
print(f"   RMSE (Train): {rmseTrain:.2f} TWD")
print(f"   RMSE as %:    {(rmseTest/real_stock_price.mean())*100:.2f}%")

print(f"\n📈 Directional Accuracy:")
print(f"   Test Set:     {directional_accuracy:.1f}%")
print(f"   Target:       60-65%+ (Good)")
print(f"   Baseline:     50% (Random)")
print(f"   Previous:     58% (Original model)")

improvement = directional_accuracy - 58.0
if improvement > 0:
    print(f"   ✅ Improvement: +{improvement:.1f} percentage points")
else:
    print(f"   ⚠️  Change: {improvement:.1f} percentage points")

# Save outputs
np.savetxt(stockID + '_improved.csv', predicted_stock_price, fmt="%.3f", delimiter=",")

# Plot test predictions
plt.figure(figsize=(12, 6))
plt.plot(real_stock_price, color='red', label='Actual Price', linewidth=2)
plt.plot(predicted_stock_price, color='blue', label='Predicted Price', linewidth=2, alpha=0.7)
plt.title(f'IMPROVED Stock Price Prediction - TSMC ({stockID})\nDirectional Accuracy: {directional_accuracy:.1f}%', fontsize=14, fontweight='bold')
plt.xlabel('Time (Days)', fontsize=12)
plt.ylabel('Stock Price (TWD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pic1_improved.png', dpi=150)
print(f"\n💾 Test predictions saved to: pic1_improved.png")

# Plot training predictions
plt.figure(figsize=(14, 6))
plt.plot(real_stock_price_train, color='red', label='Actual Price', linewidth=1.5, alpha=0.8)
plt.plot(predicted_stock_price_train, color='blue', label='Predicted Price', linewidth=1.5, alpha=0.6)
plt.title(f'Training Set Prediction - TSMC ({stockID})', fontsize=14, fontweight='bold')
plt.xlabel('Time (Days)', fontsize=12)
plt.ylabel('Stock Price (TWD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pic2_improved.png', dpi=150)
print(f"💾 Training predictions saved to: pic2_improved.png")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Model Loss Over Time', fontsize=12, fontweight='bold')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.title('Mean Absolute Error', fontsize=12, fontweight='bold')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print(f"💾 Training history saved to: training_history.png")

print("\n" + "=" * 60)
print("SUMMARY OF IMPROVEMENTS")
print("=" * 60)
print("✅ Added technical indicators (MA, ROC, Volatility, Momentum)")
print("✅ Increased timesteps from 20 to 30")
print("✅ Enhanced model with Bidirectional LSTM layers")
print("✅ Added dropout layers for regularization")
print("✅ Implemented early stopping and learning rate reduction")
print("✅ Increased model capacity (64→64→32 units)")
print("=" * 60)
