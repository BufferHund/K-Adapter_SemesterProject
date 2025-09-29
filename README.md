# K-Adapter Experiments and Results (Reorganized)

This repository contains our experimental study of K-Adapter on entity typing and related tasks, along with extensive ablations and efficiency analysis. We reorganized all runnable scripts under `Experiments/` to make reproduction straightforward, and we fixed several critical issues to support robust ablations.

For the original K-Adapter paper, see: https://arxiv.org/abs/2002.01808

## Environment
- Python 3.6
- PyTorch 1.3.1
- tensorboardX
- transformers

Create environment and install dependencies:
```bash
conda create -n kadapter python=3.6
conda activate kadapter
pip install -r requirements.txt
```

## Key Code Fixes Applied
We applied the following fixes to enable dynamic and robust experiments (details documented in `ABLATION_STUDY_GUIDE.md`):
- Dynamic adapter internals derived from `adapter_size` (hidden size, intermediate size, attention heads) to avoid shape/head divisibility issues when `adapter_size != 768`.
- Robust "concat" fusion in `examples/run_finetune_openentity_adapter.py` that works for single or multiple adapters.
- Correct saving of adapter models in `fac-adapter.py` to avoid saving tuples.

## 4 Experiments and Results

We structure the evaluation in three parts: fine-tuning vs. zero-shot; ablations on adapter architecture; and efficiency plus pre-training data impact.

### 4.1 Fine-tuning vs. Zero-shot Transfer

#### 4.1.1 Fine-tuning Performance on OpenEntity
Our reproduction on OpenEntity shows that concatenation tends to outperform addition under matched hyperparameters, but none of the adapter settings surpass full fine-tuning of RoBERTa-large. Results summarized below:

| Setting | Fusion | LR | Test F1 (ours) | Test F1 (paper) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (full fine-tuning) | N/A | 2e-5 | ~76.2% | (n/a) | Reference baseline |
| fac-adapter | add | 5e-6 | ~76.2% | ~77.5% | Matches baseline in our runs |
| lin-adapter | add | 5e-6 | ~75.6% | ~76.9% | Slightly below baseline |
| fac+lin | add | 5e-6 | ~75.4% | (concat ~77.6%) | Add underperforms |
| fac+lin | concat | 5e-6 | ~75.8% | ~77.6% | Best adapter fusion for us |
| fac+lin | concat | 1e-4 | ~68.8% | ~77.6% | Illustrates LR sensitivity |

Run scripts:
```bash
# Fine-tune on OpenEntity (adapter tuning)
bash Experiments/4.1/OpenEntity_Finetune/run_finetune_openentity_adapter.sh

# Full fine-tuning baseline (no adapters)
bash Experiments/4.1/OpenEntity_Finetune/run_finetune_full.sh
```

#### 4.1.2 Zero-shot Transferability
Pre-trained adapters without any task-specific fine-tuning do not help zero-shot transfer on OpenEntity and only provide negligible gains on FIGER where absolute scores are near random.

OpenEntity (Zero-shot):

| Configuration | Fusion | Test F1 (Micro) |
| :--- | :--- | :--- |
| Baseline (RoBERTa-large) | N/A | ~24.0% |
| + fac-adapter | add | ~15.3% |
| + lin-adapter | add | ~14.1% |
| + fac+lin | concat | ~19.4% |
| + fac+lin | add | ~11.7% |

FIGER (Zero-shot):

| Configuration | Fusion | Test F1 (Micro) |
| :--- | :--- | :--- |
| Baseline (RoBERTa-large) | N/A | ~2.5% |
| + fac-adapter | add | ~2.7% |
| + lin-adapter | add | ~2.6% |
| + fac+lin | add | ~2.8% |
| + fac+lin | concat | ~2.6% |

Overall, K-Adapter’s factual and linguistic adapters need downstream fine-tuning to deliver value; their out-of-the-box zero-shot transfer is limited and can even be harmful on OpenEntity.

Run scripts:
```bash
# Fine-tune on FIGER (for comparison; zero-shot analysis discussed in text)
bash Experiments/4.1/FIGER_Finetune/run_finetune_figer_adapter.sh
```

Additional downstream:
```bash
# TACRED relation classification (fine-tuning)
bash Experiments/4.1/TACRED_Finetune/run_finetune_tacred_adapter.sh

# CosmosQA (multiple-choice QA)
bash Experiments/4.1/CosmosQA_Finetune/run_finetune_cosmosqa_adapter.sh
```

### 4.2 Ablation Studies on Adapter Architecture

#### 4.2.1 Impact of Adapter Size
Varying the bottleneck `adapter_size` from 16 to 768 yields remarkably stable performance on OpenEntity (F1-A around 0.686–0.690), demonstrating high parameter efficiency. Small adapters are sufficient.

Run scripts:
```bash
# Pretrain + finetune sweeps for adapter size
bash Experiments/4.2/Adapter_Size/run_ablation_study_size.sh

# Or run finetune per size directly
bash Experiments/4.2/Adapter_Size/run_finetune_oe_size16.sh
bash Experiments/4.2/Adapter_Size/run_finetune_oe_size64.sh
bash Experiments/4.2/Adapter_Size/run_finetune_oe_size256.sh
bash Experiments/4.2/Adapter_Size/run_finetune_oe_size768.sh
```

#### 4.2.2 Impact of Insertion Position
We tested early (0–2), middle (10–12), late (21–23), and dispersed (0,11,22). Middle-layer insertion performs best (Micro F1: 0.706). Late-layer insertion significantly degrades performance.

Run scripts:
```bash
# Pretrain variants for insertion positions / numbers
bash Experiments/4.2/Insertion_Position/run_ablation_study_position.sh
bash Experiments/4.2/Insertion_Position/run_ablation_study_number.sh

# Finetune per setting
bash Experiments/4.2/Insertion_Position/run_finetune_pos_early.sh
bash Experiments/4.2/Insertion_Position/run_finetune_pos_middle.sh
bash Experiments/4.2/Insertion_Position/run_finetune_pos_late.sh
# Optional: run all positions convenience script
bash Experiments/4.2/Insertion_Position/run_finetune_position_all.sh
```

#### 4.2.3 Impact of Internal Complexity
Varying the number of internal transformer layers inside the adapter (1, 2, 4) shows minimal effect. The simplest 1-layer adapter performs on par with deeper variants.

Run scripts:
```bash
# Pretrain variants for different internal depths
bash Experiments/4.2/Internal_Complexity/run_ablation_study_layers.sh

# Finetune per depth
bash Experiments/4.2/Internal_Complexity/run_finetune_layers_1.sh
bash Experiments/4.2/Internal_Complexity/run_finetune_layers_4.sh
```

### 4.3 Efficiency and Pre-training Data Analysis

#### 4.3.1 Parameter and Inference Efficiency
Adapter-tuning trains far fewer parameters than full fine-tuning but introduces predictable inference overhead (higher latency and memory). See `benchmark_inference.py` for measurements across batch sizes and adapter combinations.

Run benchmark:
```bash
python benchmark_inference.py \
  --model roberta-large \
  --batch_sizes 1 8 16 32 \
  --adapters none,factual,factual+linguistic
```

#### 4.3.2 Impact of Pre-training Data
Adapters pre-trained on large, general-purpose knowledge (e.g., T-REx) transfer much better than those trained on smaller, task-specific data (e.g., TACRED). On OpenEntity, the T-REx adapter attains F1-A ≈ 0.762, whereas the TACRED-pretrained adapter reaches ≈ 0.397.

Run scripts:
```bash
# Prepare TACRED pretraining data and train a TACRED adapter
python Experiments/4.3/Pretraining_Data_Impact/preprocess_tacred.py \
  --input_dir ./data/tacred \
  --output_dir ./data/tacred_pretrain

bash Experiments/4.3/Pretraining_Data_Impact/run_pretrain_on_tacred.sh

# Evaluate the TACRED-pretrained adapter on OpenEntity
bash Experiments/4.3/Pretraining_Data_Impact/run_finetune_with_tacred_adapter.sh

# Pretrain factual/linguistic adapters on default corpora (for comparison)
bash Experiments/4.3/Pretraining_Data_Impact/run_pretrain_fac-adapter.sh
bash Experiments/4.3/Pretraining_Data_Impact/run_pretrain_lin-adapter.sh
```

## Data Notes
- T-REx processing helpers:
  - `scripts/clean_T_REx.py`
  - `scripts/create_subdataset-relation-classification.ipynb`
- OpenEntity/FIGER/TACRED formatting follows standard conventions described in our experiments. For TACRED pretraining, see `Experiments/4.3/Pretraining_Data_Impact/preprocess_tacred.py`.

## Repro Tips
- Adjust GPU devices via `CUDA_VISIBLE_DEVICES` in scripts when needed.
- Hyperparameters for each experiment are specified inside the corresponding scripts under `Experiments/`.
- Ensure adapter checkpoints exist or are trained before running fine-tuning scripts that expect them.

## Visualization and Data Processing Scripts

### Visualization Scripts
Plot experimental results from `outputs_light/` directory:
```bash
python scripts/plot_exp1_adapter_size.py      # Adapter size ablation
python scripts/plot_exp2_adapter_position.py  # Insertion position ablation  
python scripts/plot_exp3_adapter_layers.py    # Internal complexity ablation
python scripts/plot_exp4_efficiency.py        # Parameter efficiency analysis
python scripts/plot_exp4_inference_from_doc.py # Inference benchmarking
```
Outputs: `figures/` directory with PNG plots and CSV data files.

### Data Processing Scripts
Preprocess datasets for adapter pretraining:
```bash
# T-REx dataset (factual adapter pretraining)
python scripts/clean_T_REx.py --input_dir ./data/t_rex_raw --output_dir ./data/t_rex_cleaned
jupyter notebook scripts/create_subdataset-relation-classification.ipynb

# TACRED dataset (task-specific pretraining)
python Experiments/4.3/Pretraining_Data_Impact/preprocess_tacred.py \
  --input_dir ./data/tacred --output_dir ./data/tacred_pretrain

# SciERC dataset (scientific relation extraction)
python preprocess_scierc.py
```

## References
- K-Adapter paper: https://arxiv.org/abs/2002.01808

