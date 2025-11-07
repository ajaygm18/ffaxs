# GPU Setup Instructions - Run FFAXS Project on Your Device

This guide will help you run the complete FFAXS project (TSMC stock prediction + Chinese sentiment analysis) on your GPU-enabled device.

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (recommended: GTX 1060 or better)
- **CUDA**: Version 10.0 or 10.1 (for TensorFlow 1.15)
- **cuDNN**: Version 7.6 (compatible with CUDA 10.x)
- **RAM**: At least 16GB (BERT training uses ~9GB)
- **Disk Space**: ~5GB free (for BERT model and data)
- **OS**: Linux (Ubuntu 18.04/20.04), Windows 10/11, or macOS

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ajaygm18/ffaxs.git
cd ffaxs
git checkout copilot/disable-workflow-timeouts
```

### 2. Install Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Download and install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create Python 3.7 environment
conda create -n tf1env python=3.7 -y
conda activate tf1env
```

#### Option B: Using virtualenv

```bash
# Install Python 3.7 if not available
sudo apt-get install python3.7 python3.7-venv

# Create virtual environment
python3.7 -m venv tf1env
source tf1env/bin/activate  # On Windows: tf1env\Scripts\activate
```

### 3. Install TensorFlow GPU

```bash
# For GPU support (CUDA 10.0)
pip install tensorflow-gpu==1.15.0

# Or CPU-only version (slower)
# pip install tensorflow==1.15.0

# Install other dependencies
pip install pandas protobuf==3.20.3
```

### 4. Verify GPU Setup

```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.test.is_gpu_available())"
```

**Expected output:**
```
GPU Available: True
```

If False, check your CUDA/cuDNN installation.

### 5. Download BERT Pre-trained Model

```bash
cd zens/sentiment

# Download Chinese BERT model (393MB)
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

# Extract
unzip chinese_L-12_H-768_A-12.zip

# Remove zip file
rm chinese_L-12_H-768_A-12.zip

# Verify files
ls chinese_L-12_H-768_A-12/
# Should show: bert_config.json, bert_model.ckpt.*, vocab.txt

cd ../..
```

**Alternative (Windows PowerShell):**
```powershell
cd zens\sentiment
Invoke-WebRequest -Uri "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip" -OutFile "chinese_L-12_H-768_A-12.zip"
Expand-Archive -Path "chinese_L-12_H-768_A-12.zip" -DestinationPath "."
Remove-Item "chinese_L-12_H-768_A-12.zip"
cd ..\..
```

### 6. Install Stock Prediction Dependencies

```bash
# For stock prediction (Keras/LSTM)
pip install keras scikit-learn matplotlib
```

## Running the Projects

### A. Stock Price Prediction (LSTM)

This is faster and should complete in 2-5 minutes with GPU.

```bash
cd zens/stockPrice

# Run prediction
python prediction.py

# Output files will be generated:
# - 2330.csv (predicted prices)
# - pic1.png (test set visualization)
# - pic2.png (training set visualization)
```

**Expected output:**
```
traindata = data/2330_2015_2019_ochlv.csv
testdata = data/2330_202001_03_ochlv.csv
...
Epoch 200/200
38/38 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 1.6505e-04
RMSE_test = 7.649986350165048
RMSE_train = 2.318478729789448
```

### B. Sentiment Analysis (BERT)

This will take longer (~1-2 hours with GPU, ~10-15 hours with CPU for 3 epochs).

#### Training

```bash
cd zens/sentiment

# Option 1: Use the provided training script (3 epochs)
bash train.sh

# Option 2: Custom training with fewer epochs (faster testing)
python run_classifier.py \
  --data_dir=data \
  --task_name=sim \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --output_dir=tmp/sim_model \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=300 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0
```

**Training output (you'll see):**
```
INFO:tensorflow:global_step/sec: 0.5-2.0  (with GPU)
INFO:tensorflow:examples/sec: 8-32 (with GPU)
...
INFO:tensorflow:Saving checkpoints for 820 into tmp/sim_model/model.ckpt.
```

#### Testing/Prediction

```bash
# After training completes, run prediction
bash predict.sh

# This will generate: tmp/output/test_results.tsv

# Calculate accuracy
python test.py
```

**Expected output:**
```
acc is: 0.85-0.92  (85-92% accuracy)
```

#### Demo with TSMC Stock News

```bash
# Run sentiment analysis on TSMC stock news samples
python intent.py
```

This will analyze sentiment of TSMC (2330) stock news articles.

## Performance Comparison

| Hardware | Training Speed | Time for 1 Epoch | Time for 3 Epochs |
|----------|----------------|------------------|-------------------|
| **CPU Only** | ~0.05 steps/sec | ~4.5 hours | ~13.5 hours |
| **GPU (GTX 1060)** | ~0.5 steps/sec | ~27 minutes | ~1.5 hours |
| **GPU (RTX 3080)** | ~2.0 steps/sec | ~7 minutes | ~21 minutes |

## Monitoring Training Progress

### On Linux/Mac:
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training log (in another terminal)
tail -f tmp/sim_model/train.log

# Or for the script
tail -f nohup.out
```

### On Windows:
```powershell
# Monitor GPU usage
nvidia-smi -l 1

# View log file
Get-Content tmp/sim_model/train.log -Wait
```

## Troubleshooting

### GPU Not Detected

**Check CUDA version:**
```bash
nvcc --version
```

**Check cuDNN:**
```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

**Verify TensorFlow GPU:**
```python
import tensorflow as tf
print(tf.test.gpu_device_name())
# Should show: /device:GPU:0
```

### Out of Memory Error

Reduce batch size in train.sh:
```bash
--train_batch_size=8  # Instead of 16
```

Or reduce sequence length:
```bash
--max_seq_length=128  # Instead of 300
```

### CUDA Version Mismatch

For different CUDA versions, install compatible TensorFlow:
- **CUDA 10.0**: `pip install tensorflow-gpu==1.15.0`
- **CUDA 10.1**: `pip install tensorflow-gpu==1.15.0` (same)
- **CUDA 11.x**: Use TensorFlow 2.x (requires code modifications)

## Output Files Location

After successful runs, you'll have:

```
zens/stockPrice/
├── 2330.csv              # Predicted stock prices
├── pic1.png              # Test predictions chart
└── pic2.png              # Training predictions chart

zens/sentiment/
├── tmp/
│   ├── sim_model/        # Trained BERT model checkpoints
│   │   ├── model.ckpt-820.*
│   │   └── checkpoint
│   └── output/
│       └── test_results.tsv  # Prediction results
```

## Verification

### Stock Prediction Verification

Check the RMSE values:
```bash
cd zens/stockPrice
python prediction.py | grep RMSE
```

Expected:
- RMSE_test: 6-9 (lower is better)
- RMSE_train: 2-3 (lower is better)

### Sentiment Analysis Verification

Check the accuracy:
```bash
cd zens/sentiment
python test.py
```

Expected: 85-92% accuracy on Chinese sentiment classification

## Tips for Faster Training

1. **Use GPU**: Speeds up training 10-40x
2. **Reduce epochs**: Use `--num_train_epochs=1.0` for quick testing
3. **Reduce sequence length**: Use `--max_seq_length=128` if sentences are shorter
4. **Increase batch size** (if you have enough GPU memory): `--train_batch_size=32`
5. **Use mixed precision** (TensorFlow 2.x feature, not available in 1.15)

## Data Verification

All data is real (no synthetic data):

### Stock Data
```bash
head -5 zens/stockPrice/data/2330_2015_2019_ochlv.csv
```

Real TSMC historical prices from Taiwan Stock Exchange.

### Sentiment Data
```bash
head -5 zens/sentiment/data/train.tsv
```

Real Chinese book/product reviews.

## Next Steps

After successful runs:

1. View the charts: `pic1.png` and `pic2.png`
2. Check prediction accuracy in output files
3. Experiment with hyperparameters for better results
4. Integrate sentiment scores into stock prediction (advanced)

## Support

If you encounter issues:

1. Check GPU is properly detected: `nvidia-smi`
2. Verify CUDA/cuDNN compatibility
3. Check TensorFlow version: `pip show tensorflow-gpu`
4. Review error logs in `tmp/sim_model/` directory
5. Ensure all data files are present in `data/` directories

## Summary Commands (Quick Start)

```bash
# Setup
git clone https://github.com/ajaygm18/ffaxs.git
cd ffaxs
git checkout copilot/disable-workflow-timeouts
conda create -n tf1env python=3.7 -y
conda activate tf1env
pip install tensorflow-gpu==1.15.0 pandas protobuf==3.20.3 keras scikit-learn matplotlib

# Download BERT model
cd zens/sentiment
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip && rm chinese_L-12_H-768_A-12.zip
cd ../..

# Run stock prediction (fast)
cd zens/stockPrice && python prediction.py

# Run sentiment analysis (slower)
cd ../sentiment && bash train.sh
```

## Estimated Completion Times

**With GPU (GTX 1060 or better):**
- Stock prediction: 2-5 minutes ✓
- Sentiment training (1 epoch): 25-30 minutes ✓
- Sentiment training (3 epochs): 1-2 hours ✓
- Total: ~1.5-2.5 hours

**With CPU only:**
- Stock prediction: 5-10 minutes ✓
- Sentiment training (1 epoch): 4-5 hours ⚠️
- Sentiment training (3 epochs): 12-15 hours ⚠️
- Total: ~12-15 hours

Enjoy running the full FFAXS project on your GPU device! 🚀
