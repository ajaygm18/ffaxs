import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import math

print("=" * 70)
print("ENSEMBLE APPROACH: Multiple Models for Better Direction")
print("=" * 70)

stockID = '2330'
traindata = 'data/'+ stockID + '_2015_2019_ochlv.csv'
testdata = 'data/'+ stockID +'_202001_03_ochlv.csv'

dataset_train = pd.read_csv(traindata)
test_set = pd.read_csv(testdata)

# Feature engineering
def engineer_features(data):
    df = pd.DataFrame(data, columns=['Open', 'Close', 'High', 'Low', 'Volume'])
    
    # Technical indicators
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(10).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df.values

# Prepare data
training_set = dataset_train.iloc[:,1:6].values
training_set = engineer_features(training_set)

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Multiple timesteps for ensemble
timesteps_list = [15, 20, 25]
models = []

print(f'\nTraining {len(timesteps_list)} models with different timesteps...')

for idx, timesteps in enumerate(timesteps_list):
    print(f'\nModel {idx+1}: timesteps={timesteps}')
    
    X_train = []
    Y_train = []
    for i in range(timesteps, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-timesteps:i, :])
        Y_train.append(training_set_scaled[i, 0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    
    # Build model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(timesteps, training_set.shape[1])))
    model.add(Dropout(0.25))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    model.fit(X_train, Y_train, batch_size=16, epochs=150, verbose=0)
    models.append((model, timesteps))
    print(f'  ✓ Model {idx+1} trained')

print('\n✓ All ensemble models trained')

# Test data
test_set_values = test_set.iloc[:,1:6].values
test_set_enhanced = engineer_features(test_set_values)
inputs_scaled = sc.transform(test_set_enhanced)

# Get predictions from all models
all_predictions = []

for model, timesteps in models:
    inputs_test = []
    for i in range(timesteps, len(inputs_scaled)):
        inputs_test.append(inputs_scaled[i-timesteps:i, :])
    inputs_test = np.array(inputs_test)
    
    pred = model.predict(inputs_test, verbose=0)
    pred = pred * (sc.data_max_[0] - sc.data_min_[0]) + sc.data_min_[0]
    
    # Pad to match longest sequence
    max_len = len(inputs_scaled) - min(timesteps_list)
    padding = max_len - len(pred)
    if padding > 0:
        pred = np.pad(pred, ((padding, 0), (0, 0)), mode='edge')
    
    all_predictions.append(pred)

# Ensemble: Average predictions
ensemble_prediction = np.mean(all_predictions, axis=0)

# Get actual prices (aligned with longest timestep)
max_timestep = max(timesteps_list)
real_stock_price = test_set.iloc[max_timestep:,1:2].values

# Calculate metrics
def calc_directional_accuracy(actual, predicted):
    correct = 0
    for i in range(1, len(actual)):
        actual_dir = 1 if actual[i] > actual[i-1] else 0
        pred_dir = 1 if predicted[i] > predicted[i-1] else 0
        if actual_dir == pred_dir:
            correct += 1
    return (correct / (len(actual) - 1)) * 100

dir_acc = calc_directional_accuracy(real_stock_price, ensemble_prediction)
rmse = math.sqrt(mean_squared_error(real_stock_price, ensemble_prediction))

print("\n" + "=" * 70)
print("ENSEMBLE MODEL RESULTS")
print("=" * 70)

print(f"\n📈 DIRECTIONAL ACCURACY:")
print(f"   Ensemble:     {dir_acc:.1f}%")
print(f"   Baseline:     50.0%")
print(f"   Original:     58.0%")

improvement = dir_acc - 58.0
if improvement > 0:
    print(f"   ✅ IMPROVEMENT: +{improvement:.1f} percentage points!")
elif improvement >= -2:
    print(f"   ➡️  Similar to original ({improvement:+.1f} pp)")
else:
    print(f"   ⚠️  Lower than original ({improvement:.1f} pp)")

print(f"\n📊 Magnitude:")
print(f"   RMSE: {rmse:.2f} TWD ({(rmse/real_stock_price.mean())*100:.2f}%)")

# Save
np.savetxt(stockID + '_ensemble.csv', ensemble_prediction, fmt="%.3f", delimiter=",")

# Plot
plt.figure(figsize=(14, 7))
plt.plot(real_stock_price, color='red', label='Actual', linewidth=2.5, marker='o', markersize=5)
plt.plot(ensemble_prediction, color='blue', label='Ensemble Prediction', linewidth=2, marker='x', markersize=5, alpha=0.7)
plt.title(f'Ensemble Model - TSMC ({stockID})\nDirectional Accuracy: {dir_acc:.1f}% | RMSE: {rmse:.2f}',
          fontsize=15, fontweight='bold')
plt.xlabel('Days', fontsize=13)
plt.ylabel('Price (TWD)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('pic1_ensemble.png', dpi=150)

print(f"\n💾 Saved: {stockID}_ensemble.csv, pic1_ensemble.png")
print("\n" + "=" * 70)
