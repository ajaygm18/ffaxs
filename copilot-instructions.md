## Summary

This repo contains two small projects: a BERT-based sentiment/classification toolkit under `sentiment/` and an LSTM stock-prediction demo under `stockPrice/`.

These instructions give an AI coding agent the precise, actionable knowledge needed to be productive: where core logic lives, how training is run, important assumptions, and file-level examples to reference.

## Big picture (what to know first)

- `sentiment/` is adapted from Google's BERT finetuning code (TensorFlow 1.x, Estimator/TPUEstimator). Key files: `run_classifier.py` (entrypoint), `modeling.py`, `tokenization.py`, `optimization.py`, `create_pretraining_data.py`, and `run_pretraining.py`.
- Data for `sentiment` is TSV-based in `sentiment/data/` (files: `train.tsv`, `dev.tsv`, `test.tsv`). The pipeline converts examples to TFRecord via `file_based_convert_examples_to_features` before training.
- The project expects a pre-trained BERT checkpoint directory (example used: `chinese_L-12_H-768_A-12/`) containing `vocab.txt`, `bert_config.json`, and `bert_model.ckpt`.
- `stockPrice/prediction.py` is a standalone Keras LSTM script for stock-price demo — unrelated to the `sentiment/` training pipeline.

## Important technical constraints & conventions

- This codebase targets TensorFlow 1.x (see `requirements.txt`: `tensorflow >= 1.11.0`). Do NOT port to TF2 automatically. Any upgrade is a separate, explicit task.
- CLI flags use `tf.flags`/`flags.DEFINE_*` in `run_classifier.py` (expect many required flags: `--data_dir`, `--task_name`, `--vocab_file`, `--bert_config_file`, `--output_dir`). Follow that pattern when adding new command-line options.
- Tokenization uses `tokenization.FullTokenizer(vocab_file, do_lower_case)` — most text handling flows through `tokenization.py` (Unicode normalization, wordpiece logic). When modifying IO or token handling, edit this file first.
- Sequence length defaults are large (`--max_seq_length=300` in `run_classifier.py` and `train.sh`). Check `bert_config.json` for `max_position_embeddings` to avoid incompatibility.
- Training produces TFRecords and writes checkpoints to `--output_dir`. Many helper functions use `tf.gfile` and `tf.contrib` — keep code style consistent with the existing TF1.x APIs.

## Developer workflows (how to run / debug)

- Quick train (Linux/macOS example from `sentiment/train.sh`):
  - Python 3 recommended (scripts use `python3` in the shell script).
  - Example run:
    `python3 run_classifier.py --data_dir=data --task_name=sim --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --output_dir=tmp/sim_model --do_train=true --do_eval=true --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=300 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=3.0`
  - On Windows PowerShell, replace `python3` with `python` if Python 3 is on `python`.
- To run prediction/inference: use `--do_predict` and provide `--output_dir` and `--data_dir` (test tsv). Output is written to `test_results.tsv`.
- For tokenization/compatibility issues, check `tokenization.validate_case_matches_checkpoint()` which validates `--do_lower_case` vs checkpoint name.

## Project-specific patterns to follow

- Reuse the existing `DataProcessor` subclasses in `run_classifier.py` for new tasks: add a new processor class and register it in `processors` mapping; follow `SimProcessor` and `MyTaskProcessor` examples.
- Prefer TFRecord I/O utilities already present (do not invent new record formats). Use `file_based_convert_examples_to_features` and `file_based_input_fn_builder` when making dataset changes.
- When adding CLI flags, use the `flags.DEFINE_*` pattern at top of `run_classifier.py` and mark required flags at the bottom with `flags.mark_flag_as_required`.

## Integration points & external dependencies

- External: TensorFlow 1.x and pandas (see `sentiment/requirements.txt`). BERT pre-trained checkpoints are external artifacts expected in directories like `chinese_L-12_H-768_A-12/`.
- Cross-file contracts: token ids flow from `tokenization.FullTokenizer` -> `convert_single_example` -> TFRecord writer -> Estimator input_fn. If you change tokenization output IDs or special tokens, update all callers.

## Quick examples to copy-paste

- Create TFRecords for training (internal call): `file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)` (see `run_classifier.py`).
- Build model: `model_fn = model_fn_builder(bert_config=..., num_labels=..., init_checkpoint=..., ...); estimator = tf.contrib.tpu.TPUEstimator(..., model_fn=model_fn, ...)`.

## What not to change without test/permission

- Do not migrate code to TF2 or change high-level Estimator/TPU usage without explicit task request and tests — the repo is TF1.x specific.
- Avoid changing tokenization primitives or vocabulary format without updating `vocab.txt` and re-running pretraining/feature creation.

## Files to inspect first when modifying behavior

- `sentiment/run_classifier.py` — main entrypoint and task definitions
- `sentiment/modeling.py` — core BERT model implementation
- `sentiment/tokenization.py` — tokenizer and unicode handling
- `sentiment/optimization.py` — optimizer schedule used during training
- `sentiment/train.sh` — canonical flag set for training

If anything is unclear or you'd like the instructions to be expanded with run/debug commands for Windows, a short checklist for contributing, or small automated tests, tell me which section to expand and I'll iterate.
