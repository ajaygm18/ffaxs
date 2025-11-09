# BERT Model Setup Instructions

This project requires the Chinese BERT pre-trained model which is too large to be stored in GitHub (393MB).

## Download BERT Model

The project uses the **BERT-Base, Chinese** model from Google. Follow these steps to set it up:

### Automatic Download (Recommended)

```bash
cd zens/sentiment
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
rm chinese_L-12_H-768_A-12.zip
```

### Manual Download

1. Download the model from: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
2. Extract the zip file to `zens/sentiment/chinese_L-12_H-768_A-12/`
3. Verify the following files exist:
   - `bert_config.json`
   - `bert_model.ckpt.data-00000-of-00001`
   - `bert_model.ckpt.index`
   - `bert_model.ckpt.meta`
   - `vocab.txt`

## Model Details

- **Model Name**: BERT-Base, Chinese
- **Architecture**: 12-layer, 768-hidden, 12-heads
- **Total Parameters**: 110M
- **File Size**: ~393MB
- **Source**: Google Research

## Important Notes

- The model directory `chinese_L-12_H-768_A-12/` is excluded from git via `.gitignore`
- You must download this model before running any sentiment analysis scripts
- The model is only needed for the sentiment analysis project under `zens/sentiment/`

## Alternative Storage Options

If you need to share the model with your team:
1. Store it on cloud storage (Google Drive, Dropbox, AWS S3, etc.)
2. Use Git LFS (Large File Storage) if your repository supports it
3. Set up a shared network drive or internal artifact repository
