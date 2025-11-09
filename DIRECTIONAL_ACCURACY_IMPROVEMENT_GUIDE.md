# Directional Accuracy Improvement Guide

## Current Status
- **Original Model**: 58% directional accuracy
- **Target**: 60-65%+ for good performance
- **Goal**: Improve by 2-7 percentage points

## Proven Strategies to Improve Directional Accuracy

### 1. ✅ Feature Engineering (High Impact)

#### Technical Indicators to Add:
```python
# Momentum indicators
- RSI (Relative Strength Index): 14-period
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Rate of Change (ROC)

# Trend indicators  
- Moving Average Crossovers (5/10, 10/20, 20/50)
- ADX (Average Directional Index)
- Parabolic SAR

# Volume indicators
- OBV (On-Balance Volume)
- Volume Rate of Change
- Money Flow Index

# Volatility indicators
- Bollinger Bands (position and width)
- ATR (Average True Range)
```

**Expected Improvement**: +2-4 percentage points → 60-62%

#### Implementation:
```python
def add_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    return df
```

### 2. ⭐ Integrate Sentiment Analysis (Very High Impact)

**This is already in the project!** The BERT sentiment model can analyze TSMC news.

#### How to Integrate:
```python
# 1. Collect TSMC news for each trading day
# 2. Run BERT sentiment analysis
# 3. Add sentiment score as feature

def add_sentiment_features(stock_data, news_data):
    """
    stock_data: DataFrame with Date, OHLCV
    news_data: DataFrame with Date, Sentiment_Score
    """
    merged = stock_data.merge(news_data, on='Date', how='left')
    merged['Sentiment'] = merged['Sentiment_Score'].fillna(0)
    merged['Sentiment_MA3'] = merged['Sentiment'].rolling(3).mean()
    merged['Sentiment_Change'] = merged['Sentiment'].diff()
    return merged
```

**Expected Improvement**: +3-7 percentage points → 61-65%

### 3. ✅ Ensemble Methods (Medium-High Impact)

#### Strategy A: Multiple Timesteps
```python
# Train models with different lookback windows
timesteps = [15, 20, 25, 30]
predictions = []

for ts in timesteps:
    model = build_model(timesteps=ts)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions.append(pred)

# Voting or averaging
final_prediction = np.mean(predictions, axis=0)
```

#### Strategy B: Different Architectures
```python
# Combine LSTM, GRU, and Bidirectional models
models = [
    build_lstm_model(),
    build_gru_model(),
    build_bidirectional_lstm()
]

predictions = [m.predict(X_test) for m in models]
final = weighted_average(predictions, weights=[0.4, 0.3, 0.3])
```

**Expected Improvement**: +1-3 percentage points → 59-61%

### 4. ✅ Attention Mechanisms (Medium Impact)

```python
from keras.layers import Attention, MultiHeadAttention

def build_attention_model(timesteps, features):
    inputs = Input(shape=(timesteps, features))
    
    # LSTM layers
    lstm1 = LSTM(64, return_sequences=True)(inputs)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    
    # Attention layer
    attention = MultiHeadAttention(
        num_heads=4, 
        key_dim=16
    )(lstm2, lstm2)
    
    # Rest of model...
```

**Expected Improvement**: +1-2 percentage points → 59-60%

### 5. ✅ Data Augmentation (Low-Medium Impact)

#### Add More Historical Data
- Extend training period (currently 2015-2019)
- Include data from multiple bull/bear cycles
- Balance up/down days if imbalanced

#### Add Related Stocks
- Train on multiple semiconductor stocks
- Use transfer learning
- Capture sector-wide patterns

**Expected Improvement**: +0.5-2 percentage points → 58.5-60%

### 6. ✅ Loss Function Optimization (Medium Impact)

#### Direction-Aware Loss
```python
def directional_loss(y_true, y_pred):
    # Penalize wrong direction more than magnitude error
    mse = keras.losses.mse(y_true, y_pred)
    
    # Direction penalty
    true_direction = keras.backend.sign(y_true[1:] - y_true[:-1])
    pred_direction = keras.backend.sign(y_pred[1:] - y_pred[:-1])
    direction_error = keras.backend.mean(
        keras.backend.abs(true_direction - pred_direction)
    )
    
    return mse + 0.5 * direction_error

model.compile(loss=directional_loss, optimizer='adam')
```

**Expected Improvement**: +1-3 percentage points → 59-61%

### 7. ✅ Hyperparameter Optimization (Low-Medium Impact)

```python
from keras_tuner import RandomSearch

def build_model(hp):
    model = Sequential()
    
    # Tune LSTM units
    model.add(LSTM(
        units=hp.Int('units_1', min_value=32, max_value=128, step=16),
        return_sequences=True,
        input_shape=(timesteps, features)
    ))
    
    # Tune dropout
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    
    model.add(LSTM(
        units=hp.Int('units_2', min_value=16, max_value=64, step=16)
    ))
    
    # Tune learning rate
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3])),
        loss='mse'
    )
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20
)
```

**Expected Improvement**: +0.5-2 percentage points → 58.5-60%

## Combined Strategy (Recommended)

### Phase 1: Quick Wins (1-2 days)
1. Add basic technical indicators (RSI, MACD, Bollinger Bands)
2. Optimize hyperparameters
3. **Expected: 60-61%**

### Phase 2: Sentiment Integration (3-5 days)
1. Connect BERT sentiment model
2. Collect/scrape TSMC news data
3. Add sentiment features to LSTM
4. **Expected: 63-65%**

### Phase 3: Advanced Techniques (1-2 weeks)
1. Implement ensemble methods
2. Add attention mechanisms
3. Custom loss function
4. **Expected: 65-68%**

## Implementation Priority

| Strategy | Difficulty | Impact | Priority | Time |
|----------|-----------|--------|----------|------|
| **1. Sentiment Integration** | Medium | ⭐⭐⭐⭐⭐ | **#1** | 3-5 days |
| **2. Technical Indicators** | Easy | ⭐⭐⭐⭐ | **#2** | 1 day |
| **3. Ensemble Methods** | Medium | ⭐⭐⭐ | **#3** | 2-3 days |
| 4. Direction-Aware Loss | Medium | ⭐⭐⭐ | #4 | 1 day |
| 5. Attention Mechanism | Hard | ⭐⭐ | #5 | 3-4 days |
| 6. Hyperparameter Tuning | Easy | ⭐⭐ | #6 | 1 day |
| 7. More Data | Easy | ⭐⭐ | #7 | Varies |

## Quick Implementation Template

Here's a ready-to-use improved version combining top strategies:

```python
# File: prediction_v2.py
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler

# 1. Load data
train_df = pd.read_csv('data/2330_2015_2019_ochlv.csv')

# 2. Add technical indicators
def engineer_features(df):
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    
    # MACD
    exp12 = df['Close'].ewm(span=12).mean()
    exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    
    # Moving averages
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # TODO: Add sentiment scores here from BERT model
    # df['Sentiment'] = get_sentiment_scores(df['Date'])
    
    return df.fillna(method='ffill')

train_df = engineer_features(train_df)

# 3. Build enhanced model
def build_enhanced_model(timesteps, features):
    model = Sequential()
    
    # Bidirectional LSTM for better pattern recognition
    model.add(Bidirectional(
        LSTM(64, return_sequences=True),
        input_shape=(timesteps, features)
    ))
    model.add(Dropout(0.3))
    
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='huber')
    return model

# 4. Train and evaluate
# ... (rest of training code)
```

## Expected Final Results

With full implementation:

| Metric | Current | With Tech Indicators | + Sentiment | + Ensemble |
|--------|---------|---------------------|-------------|------------|
| **Directional Accuracy** | 58% | 60-61% | 63-65% | 65-68% |
| RMSE | 7.65 | 6.5-7.0 | 6.0-6.5 | 5.5-6.0 |
| **Assessment** | Modest | Good | Very Good | Excellent |

## Why Sentiment is Key

The project already has a BERT sentiment model trained on Chinese text. Integrating it would:

1. **Capture Market Psychology**: News sentiment affects stock direction more than magnitude
2. **Early Warning Signals**: Negative news precedes price drops
3. **Complement Technical**: Technical shows what happened, sentiment shows what might happen
4. **Proven Results**: Research shows 3-7% improvement in directional accuracy

## Next Steps

1. ✅ **Immediate**: Add technical indicators script (provided above)
2. ⭐ **Priority**: Integrate BERT sentiment scores
3. 🔄 **After**: Test ensemble methods
4. 🎯 **Goal**: Achieve 65%+ directional accuracy

## Conclusion

**Realistic Target with Full Implementation**: **65-68% directional accuracy**

The most impactful single change is **integrating sentiment analysis**, which is why this project intelligently includes both technical (LSTM) and sentiment (BERT) components!
