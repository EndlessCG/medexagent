# MedExAgent

This is the official repository for "MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments". 

Paper: https://arxiv.org/abs/2605.07058

## Citation

```bibtex
@misc{gao2026medexagenttrainingllmagents,
      title={MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments}, 
      author={Yicheng Gao and Xiaolin Zhou and Yahan Li and Yue Zhao and Ruishan Liu},
      year={2026},
      eprint={2605.07058},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.07058}, 
}
```

## Setup

First, install dependencies in a virtual environment (Python 3.11 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install weave # optional, for logging
```

Also, set API credentials for OpenAI-compatible endpoints when using API-based data generation, judging, patient simulation, or model inference:

```bash
export OPENAI_API_KEY=...
export WANDB_API_KEY=... # optional, for wandb logging
```

Most commands accept `--base-url` or `--api-base-url`, so local vLLM/SGLang/OpenAI-compatible servers can be used instead of `https://api.openai.com/v1`.

## Model Checkpoint

The released 8B model checkpoint is hosted on Hugging Face:

- https://huggingface.co/medagent/MedExAgent-8B

The model repository is gated for research-use acknowledgement. It is not intended for clinical use.

After access is granted, you can chat with the model locally:

```bash
python chat.py --hf-model medagent/MedExAgent-8B
```


## Data Generation

The unified pipeline has three stages:

1. `extract`: read PMC-Patients JSON and use an LLM to extract structured patient rows;
2. `filter`: remove unsuitable rows and classify free-text exam names into tool-call strings;
3. `generate`: create multi-turn doctor-patient diagnostic conversations.

Run stages individually:

```bash
python -m data.pipeline extract \
  --input data/ehr/PMC-Patients-V2.json \
  --output data/ehr/pmc_patients_extracted.csv \
  --base-url http://localhost:8000/v1 \
  --model DEFAULT \
  --total-size 100

python -m data.pipeline filter \
  --input data/ehr/pmc_patients_extracted.csv \
  --output data/ehr/pmc_patients_post_processed.csv \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini

python -m data.pipeline generate \
  --input data/ehr/pmc_patients_post_processed.csv \
  --output data/conversations/conversations.jsonl \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --patient-noise-level 0.3 \
  --exam-noise-level 0.1 \
  --noise-seed 42
```

Or run all stages:

```bash
python -m data.pipeline all \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --total-size 100
```

Checkpoint files are written under `data/ehr/` or `data/conversations/` by default, allowing interrupted runs to resume.

## Evaluation

Evaluate a local checkpoint:

```bash
python -m evaluation \
  --model /path/to/checkpoint \
  --data ddxplus=data/conversations/processed/ddxplus_test_conversations_clean.jsonl \
         pmc=data/conversations/processed/pmc_test_conversations_clean.jsonl \
  --eval-types diagnosis conversation \
  --judge-model gpt-4.1-mini \
  --patient-model gpt-4.1-mini \
  --output eval_results.json
```

Evaluate an OpenAI-compatible API model:

```bash
python -m evaluation \
  --api-model my-served-model \
  --api-base-url http://localhost:8000/v1 \
  --data test=data/conversations/processed/pmc_test_conversations_clean.jsonl \
  --eval-types diagnosis conversation \
  --no-wandb
```

## Supervised Fine-Tuning

Edit `config/sft_config.yaml` first:

- set `model_name` to the base model or local checkpoint;
- set `data_path` to an SFT JSONL conversation file;
- set `output_dir` to a writable checkpoint directory;
- adjust batch size, gradient accumulation, LoRA, and evaluation paths for your hardware.

Run:

```bash
python -m train.sft.run --config config/sft_config.yaml
```

Common overrides:

```bash
python -m train.sft.run \
  --config config/sft_config.yaml \
  --model_name OpenMeditron/Meditron3-8B \
  --data_path data/conversations/processed/sft_mix.jsonl \
  --output_dir checkpoints/sft \
  --run_name medagent-sft-run
```

SFT masks loss to assistant tokens only. Tool calls are serialized into `<tool_call>` blocks for tokenizer compatibility.

## Reinforcement Learning

RL uses `verl` with a custom multi-turn medical agent loop. Edit `config/rl_config.yaml` first:

- set `model_path` and `tokenizer_path` to an SFT checkpoint or compatible model;
- set `data_path` to the RL CSV;
- set `output_dir` to a writable checkpoint directory;
- tune GPU, batch, rollout, and reward settings for your cluster.

Prepare only the verl parquet dataset:

```bash
python -m train.rl.run --config config/rl_config.yaml --prepare_data_only
```

Train:

```bash
python -m train.rl.run --config config/rl_config.yaml
```

Resume:

```bash
python -m train.rl.run \
  --config config/rl_config.yaml \
  --resume_from_checkpoint latest
```

The RL launcher patches installed `verl` modules at runtime to surface MedExAgent metrics, save top-k validation checkpoints, and throttle rollout tracing. Use a disposable or project-specific Python environment if you do not want these local package edits to affect other projects.

## Noise Settings

Both data generation and RL dataset preparation support controlled noise:

- `patient_noise_level`: fraction/probability of patient-side noise such as incomplete or misleading symptom reporting;
- `exam_noise_level`: fraction/probability of noisy exam findings;
- `noise_seed`: reproducible noise assignment seed.

These settings are useful for training and evaluating robustness in noisy clinical environments.

## Notes

- This repository is for research use, not clinical deployment.
