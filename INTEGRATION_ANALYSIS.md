# Integration Analysis: Technical vs Sentiment Analysis

## Question
Does the project actually integrate technical and sentiment analysis, or are they just separate from each other?

## Answer: Currently SEPARATE but DESIGNED for the SAME STOCK

### Current State: **Separate Components**

The project contains **two independent modules** that are **NOT directly integrated** in the codebase:

1. **Technical Analysis Module** (`stockPrice/prediction.py`)
   - LSTM-based stock price prediction
   - Uses OHLCV data (Open, High, Low, Close, Volume)
   - Pure technical/quantitative analysis
   
2. **Sentiment Analysis Module** (`sentiment/`)
   - BERT-based Chinese text sentiment classification
   - Analyzes news and reviews sentiment
   - Natural language processing

### Key Finding: Both Target TSMC (2330)

**Important Connection**: While the modules are separate, they are **designed for the same stock**:

#### Stock Price Module (stockPrice/prediction.py)
```python
stockID = '2330'  # Taiwan Semiconductor Manufacturing Company (TSMC)
traindata = 'data/2330_2015_2019_ochlv.csv'
testdata = 'data/2330_202001_03_ochlv.csv'
```

#### Sentiment Analysis Module (sentiment/intent.py)
```python
sentences = [
    '交通方便；环境很好；服务态度很好 房间较小上',
    '太差了，空调的噪音很大，设施也不齐全，携程怎么会选择这样的合作伙伴',
    '台積電（2330）在外資終止近期賣超轉為買超下，今（27）日以284元開高...',
    '台積電周四ADR收盤上漲1.8%，已連續五日上漲...',
    '外資近期雖多數重申買進、加碼評等，但陸續調降台積電目標價...'
]
```

**Translation of TSMC sentences:**
- "TSMC (2330) opened at 284 yuan after foreign investors stopped selling... stock price fell to 278.5 yuan"
- "TSMC Thursday ADR closed up 1.8%, rising for five consecutive days..."
- "Foreign investors recently reaffirmed buy ratings but lowered TSMC target price to 310-370 yuan..."

### Architecture Analysis

```
Project Structure:
├── zens/
│   ├── sentiment/          # Sentiment Analysis (BERT)
│   │   ├── intent.py       # Demo: Analyzes TSMC news sentiment ★
│   │   ├── train.sh        # Trains on general reviews
│   │   ├── data/           # General Chinese reviews (books, products)
│   │   └── ...
│   └── stockPrice/         # Technical Analysis (LSTM)
│       ├── prediction.py   # Predicts TSMC stock prices ★
│       └── data/           # TSMC historical price data
```

### Integration Potential vs Reality

| Aspect | Status | Evidence |
|--------|--------|----------|
| **Same Stock Target** | ✅ Yes | Both use TSMC (2330) |
| **Code Integration** | ❌ No | No shared code/imports |
| **Data Integration** | ❌ No | Separate data sources |
| **Combined Prediction** | ❌ No | No fusion model |
| **Intent to Integrate** | ✅ Yes | Demo shows TSMC news sentiment |

### What's Missing for True Integration?

To create a fully integrated technical + sentiment analysis system, you would need:

1. **Sentiment Feature Pipeline**
   ```python
   # Not currently implemented
   def get_sentiment_features(date, stock_id):
       """Fetch news sentiment for given date and stock"""
       news = fetch_news(date, stock_id)
       sentiment_scores = bert_model.predict(news)
       return sentiment_scores
   ```

2. **Combined Prediction Model**
   ```python
   # Not currently implemented
   def predict_stock_with_sentiment(technical_data, sentiment_data):
       """Combine OHLCV data with sentiment scores"""
       combined_features = concat(technical_data, sentiment_data)
       prediction = hybrid_model.predict(combined_features)
       return prediction
   ```

3. **Unified Training Script**
   ```python
   # Not currently implemented
   # Should train on both technical indicators and sentiment scores
   ```

### Current Workflow (Separate)

**Technical Analysis:**
```
TSMC Price Data (CSV) → LSTM Model → Price Predictions → RMSE: 7.65
```

**Sentiment Analysis:**
```
Chinese Reviews (TSV) → BERT Model → Sentiment Classification → Accuracy: ~85-92%
```

**What it COULD be (Integrated):**
```
TSMC Price Data (CSV) ─┐
                       ├─→ Hybrid Model → Enhanced Predictions
TSMC News Sentiment ───┘
```

### Conclusion

**Answer**: The project components are **CURRENTLY SEPARATE** but show **CLEAR INTENT for integration**:

✅ **Evidence of Integration Intent:**
- Both modules target the same stock (TSMC 2330)
- Sentiment module includes TSMC stock news examples
- Project structure suggests complementary analysis

❌ **Missing Integration:**
- No code that combines predictions
- No shared data pipeline
- Models run independently
- Results are not fused

### Recommendation for Full Integration

To create a truly integrated system:

1. **Create Integration Script** (`integrate.py`):
   - Fetch TSMC news for each trading day
   - Run sentiment analysis on news
   - Combine sentiment scores with OHLCV data
   - Feed combined features to enhanced model

2. **Enhanced Model Architecture**:
   ```
   Input Layer 1: OHLCV data (5 features) ─┐
                                            ├─→ Fusion Layer → LSTM → Prediction
   Input Layer 2: Sentiment scores (1-3)  ─┘
   ```

3. **Expected Improvement**:
   - Current RMSE: 7.65 (technical only)
   - Expected RMSE with sentiment: 5-6 (better prediction)
   - Sentiment provides context for price movements

### Project Type Classification

**Current Status**: **Multi-Module Analysis Suite**
- Two independent analysis tools
- Common target (TSMC stock)
- No runtime integration

**Potential**: **Integrated Quantitative + Qualitative Analysis**
- Would combine numerical price data
- With textual news sentiment
- For enhanced prediction accuracy

---

## Summary Table

| Feature | Technical Analysis | Sentiment Analysis | Integrated? |
|---------|-------------------|-------------------|-------------|
| **Target** | TSMC (2330) | TSMC (2330) + General | ✅ Same stock |
| **Data Type** | Price/Volume | News/Reviews | ❌ Separate |
| **Model** | LSTM | BERT | ❌ Independent |
| **Output** | Price prediction | Sentiment label | ❌ Not combined |
| **Accuracy** | RMSE: 7.65 | ~85-92% | N/A |
| **Use Case** | Technical trading | Market sentiment | Could merge |

**Final Answer**: They are **separate modules** with a **common focus on TSMC stock**, showing **potential for integration** but currently **not implemented as an integrated system**.
