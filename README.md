# SeACDroid

SeACDroid is an Android malware detection framework that models local API contexts around security-relevant system API calls. It extracts API context subgraphs from statically analyzed APKs, encodes each subgraph with SBERT API semantics and a GAT encoder, aggregates context embeddings with gated-attention multiple-instance learning, and uses selected high-attention contexts for LLM-based explanation.

This package contains the source code, configuration templates, and dataset manifest files needed to reproduce SeACDroid preprocessing, training, evaluation, and explanation workflows. It excludes trained checkpoints, raw APK files, generated experiment outputs, and auxiliary analysis scripts.

## Structure

```text
SeACDroid_release/
  seacdroid/
    preprocessing/
      extract_static_ir.py        # APK -> static-analysis text representation
      build_context_features.py   # text representation -> context-subgraph PKL features
    training/
      train_same_year.py     # same-year train/validation/test split
      train_robustness.py    # prepared robustness-evaluation training set
      common.py              # shared training utilities
    evaluation/
      evaluate_detector.py   # checkpoint evaluation
    llm/
      context.py             # recover readable contexts from static-analysis text
      manifest.py            # Manifest metadata extraction
      prompts.py             # focused explanation prompt formatting
      explain_sample.py      # single-sample LLM explanation
      batch_explain.py       # batch LLM explanation
    data.py                  # feature loading and APK-bag batching
    models.py                # model definition
  data/
    security_relevant_apis.txt
    ordinary_apis_list.txt
    official_packages.txt
    callbacks/AndroidCallbacks.txt
    sha256/
  configs/default.yaml
  requirements.txt
```

## Installation

Install PyTorch and PyTorch Geometric with the wheel versions matching your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The experiments in the paper used PyTorch 2.7.1 with CUDA 11.8 on an RTX 4090 workstation. Exact CUDA wheel commands may differ by machine.

## Data Layout

After preprocessing, arrange feature files as:

```text
feature_root/
  2011/
    benign/*.pkl
    malware/*.pkl
  2012/
    benign/*.pkl
    malware/*.pkl
  1112/
    benign/*.pkl
    malware/*.pkl
```

Each feature file represents one APK as a bag of context subgraphs. Raw APKs are not included. The SHA256 lists under `data/sha256/` document dataset membership. Prepared robustness-evaluation directories such as `1112`, `111213`, and `11121314` should physically contain their own `benign/` and `malware/` feature files; the training scripts do not reconstruct them by filtering files from individual year directories.

## Preprocessing

Extract the static-analysis text representation from one APK:

```bash
python -m seacdroid.preprocessing.extract_static_ir \
  /path/to/app.apk \
  /path/to/static_ir_root/2021/benign/<sha256>.txt
```

This command also writes a Manifest metadata sidecar next to the text file, for example
`/path/to/static_ir_root/2021/benign/<sha256>.manifest.json`. Pass `--manifest_output` only if you want to place that JSON elsewhere.

Convert extracted text files into SeACDroid feature PKLs:

```bash
python -m seacdroid.preprocessing.build_context_features \
  --input_root /path/to/static_ir_root \
  --output_root /path/to/feature_root \
  --datasets 2019 2020 2021 \
  --security_relevant_api_list data/security_relevant_apis.txt \
  --pkg_list data/official_packages.txt \
  --model_path all-MiniLM-L6-v2 \
  --workers 8
```

For a prepared robustness-evaluation training set, use the same physical dataset name in the text and feature roots:

```text
static_ir_root/
  1112/
    benign/*.txt
    malware/*.txt
feature_root/
  1112/
    benign/*.pkl
    malware/*.pkl
```

## Training

Train and evaluate a same-year split:

```bash
python -m seacdroid.training.train_same_year \
  --data_dir /path/to/feature_root \
  --year 2021 \
  --output_dir outputs/same_year \
  --epochs 20 \
  --batch_size 32
```

Same-year training writes a metrics report, but does not save a model checkpoint by default.
The `--split_seed` option controls only the stratified data split. The same-year script first holds out `--test_ratio` and then takes `--val_ratio` from the remaining non-test split; with the defaults, this corresponds to approximately 72% train, 8% validation, and 20% held-out test before small-sample rounding.
Training uses undirected graph edges by default. Pass `--directed` only when reproducing a directed-edge variant.

Train on a prepared robustness-evaluation dataset directory:

```bash
python -m seacdroid.training.train_robustness \
  --data_dir /path/to/feature_root \
  --train_set 1112 \
  --output_dir outputs/checkpoints \
  --epochs 20 \
  --batch_size 32
```

Robustness training saves per-epoch checkpoints as `<run_name>_epochXX.pth` by default and also saves a validation-selected checkpoint as `<run_name>_best_validation.pth`. Use `--run_name` only when you want to override the output filename prefix.

## Evaluation

Evaluate a validation-selected checkpoint on later years:

```bash
python -m seacdroid.evaluation.evaluate_detector \
  --data_dir /path/to/feature_root \
  --checkpoint outputs/checkpoints/robustness_1112_best_validation.pth \
  --test_years 2013 2014 2015 \
  --output_json outputs/robustness_1112_eval.json \
  --prediction_dir outputs/predictions
```

Evaluate all saved robustness checkpoints:

```bash
python -m seacdroid.evaluation.evaluate_detector \
  --data_dir /path/to/feature_root \
  --checkpoint_dir outputs/checkpoints \
  --checkpoint_glob "robustness_1112_epoch*.pth" \
  --test_years 2013 2014 2015 \
  --output_json outputs/robustness_1112_epochs_eval.json
```

The evaluator reports accuracy, F1-score, precision, and malware recall.

## LLM Explanation

The LLM explanation workflow follows the detector attention weights: run the checkpoint on an APK feature file, select the top-K high-attention context subgraphs, recover readable context from the matching static-analysis text file using the same context-subgraph matching logic as feature construction, add Manifest metadata from the static-analysis sidecar JSON, and send the prompt to an OpenAI-compatible LLM endpoint. By default, the sidecar path is inferred as `<static-analysis-text>.manifest.json`; pass `--manifest_json` only if that JSON was written elsewhere.

Generate the prompt and report for one sample without calling an LLM:

```bash
python -m seacdroid.llm.explain_sample \
  --checkpoint outputs/checkpoints/robustness_1112_epoch20.pth \
  --feature_file /path/to/feature_root/2018/malware/<sha256>.pkl \
  --static_ir_file /path/to/static_ir_root/2018/malware/<sha256>.txt \
  --output outputs/explanations/sample_report.json \
  --dry_run
```

Call DeepSeek by setting an environment variable:

```bash
export DEEPSEEK_API_KEY=<your_key>
python -m seacdroid.llm.explain_sample \
  --checkpoint outputs/checkpoints/robustness_1112_epoch20.pth \
  --feature_file /path/to/feature_root/2018/malware/<sha256>.pkl \
  --static_ir_file /path/to/static_ir_root/2018/malware/<sha256>.txt \
  --output outputs/explanations/sample_report.json
```

Run batch explanation over mirrored feature/text directories:

```bash
python -m seacdroid.llm.batch_explain \
  --checkpoint outputs/checkpoints/robustness_1112_epoch20.pth \
  --feature_dir /path/to/feature_root/2018 \
  --static_ir_dir /path/to/static_ir_root/2018 \
  --only_predicted_malware \
  --limit 500 \
  --output outputs/explanations/batch_results.json
```

The default endpoint is `https://api.deepseek.com` with model `deepseek-chat`. The package does not include API keys.

## Reproducibility Notes

- The main model follows the implementation used in the paper evaluation: 3 GAT layers, 4 attention heads, 128 hidden dimensions, 64-dimensional context embeddings, gated-attention MIL, and a two-layer MLP classifier.
- The 2011-2021 dataset sizes, year/class SHA256 manifests, and prepared robustness-evaluation training-set manifests are documented under `data/`.
- Trained models, generated results, raw APKs, and generated feature PKLs should stay outside the submitted source archive unless the venue explicitly asks for them.
