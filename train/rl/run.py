"""CLI entry point for Stage 2: DAPO-based RL training.

This script sets up the verl DAPO trainer with:
- Custom dataset from dataset.py
- Custom reward from reward.py
- Multi-turn rollout using patient_agent + tool_agent
- SFT checkpoint as initial policy
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .dataset import build_multi_eval_parquet, build_verl_dataset, load_patient_profiles
from .patient_agent import RuleBasedPatientAgent
from .reward import compute_score, extract_diagnosis
from .tool_agent import ToolAgent


def _patch_compute_data_metrics():
    """Patch verl's compute_data_metrics to aggregate medagent/* metrics from non_tensor_batch.

    The agent loop stores per-sample custom metrics in extra_fields (which flow to
    non_tensor_batch), because AgentLoopMetrics is a fixed Pydantic model that drops
    unknown keys.  This patch adds a block at the end of compute_data_metrics that
    picks up any key starting with "medagent/" and reports mean/max/min so they
    appear in wandb.  It also picks up one sample conversation (medagent_conv/sample)
    and logs it as a wandb.Html panel.

    The patch is written to the installed verl source file so it's visible in
    Ray actor processes (runtime monkey-patching doesn't survive across processes).
    """
    import re
    import verl.trainer.ppo.metric_utils as mu

    source_path = mu.__file__
    with open(source_path) as f:
        source = f.read()

    begin_marker = "# --- medagent custom metrics ---"
    end_marker = "# --- end medagent custom metrics ---"
    version_marker = "# medagent_patch_version: 2"

    snippet = '''\
    # --- medagent custom metrics ---
    # medagent_patch_version: 2
    _ds_col = batch.non_tensor_batch.get("data_source", None)
    for _key in list(batch.non_tensor_batch.keys()):
        if _key.startswith("medagent_conv/"):
            for _v in batch.non_tensor_batch[_key]:
                if _v:
                    try:
                        import wandb as _wandb
                        if _wandb.run is not None:
                            _wandb.log({"train/sample_conversation": _wandb.Html(
                                "<pre style='white-space:pre-wrap'>" + str(_v) + "</pre>"
                            )})
                    except Exception:
                        pass
                    break
            continue
        if not _key.startswith("medagent/"):
            continue
        _vals = batch.non_tensor_batch[_key]
        try:
            _arr = np.array([float(_v) for _v in _vals if _v is not None])
        except (TypeError, ValueError):
            continue
        if len(_arr) > 0:
            metrics[f"{_key}/mean"] = float(_arr.mean())
            metrics[f"{_key}/max"] = float(_arr.max())
            metrics[f"{_key}/min"] = float(_arr.min())
        # Per-dataset aggregation: group values by data_source column.
        if _ds_col is not None and len(_ds_col) == len(_vals):
            _by_ds = {}
            for _ds, _v in zip(_ds_col, _vals):
                if _v is None or not _ds:
                    continue
                try:
                    _by_ds.setdefault(str(_ds), []).append(float(_v))
                except (TypeError, ValueError):
                    continue
            _suffix = _key[len("medagent/"):]
            for _ds, _dvals in _by_ds.items():
                if not _dvals:
                    continue
                _darr = np.array(_dvals)
                metrics[f"medagent/{_ds}/{_suffix}/mean"] = float(_darr.mean())
                metrics[f"medagent/{_ds}/{_suffix}/max"] = float(_darr.max())
                metrics[f"medagent/{_ds}/{_suffix}/min"] = float(_darr.min())
    # --- end medagent custom metrics ---

'''

    if version_marker in source:
        return  # already patched with current version

    # If an older version of the patch is installed, strip it out before re-injecting.
    if begin_marker in source and end_marker in source:
        old_pattern = re.compile(
            r"\n?[ \t]*" + re.escape(begin_marker) + r".*?" + re.escape(end_marker) + r"\s*\n",
            re.DOTALL,
        )
        new_source, n = old_pattern.subn("\n", source)
        if n > 0:
            source = new_source
            print(f"Removed previous medagent patch from {source_path}; re-applying current version.")

    # Find the last "return metrics" inside compute_data_metrics (before the next
    # top-level function).  Use a regex so we're resilient to whitespace variations.
    pattern = re.compile(
        r'(\n    return metrics\s*\n+)(def |class )',
        re.MULTILINE,
    )
    match = pattern.search(source)
    if not match:
        print(f"WARNING: Could not find injection point in {source_path}; medagent metrics will not appear in wandb.")
        return

    insert_pos = match.start(1)
    patched = source[:insert_pos] + "\n" + snippet + match.group(1) + match.group(2) + source[match.end():]
    with open(source_path, "w") as f:
        f.write(patched)
    print(f"Patched {source_path} to aggregate medagent/* custom metrics.")


def _patch_best_checkpoint_saving():
    """Patch verl's RayPPOTrainer.fit to save top-k checkpoints based on validation metrics.

    After each validation step, computes the average of configured metrics
    (default: reward, r_diagnosis, r_tool_match, similarity). If the score is
    in the top-k, saves a checkpoint and prunes the worst to maintain exactly k.
    State is persisted to best_checkpoints.json for resume support.
    """
    import verl.trainer.ppo.ray_trainer as rt

    source_path = rt.__file__
    with open(source_path) as f:
        source = f.read()

    marker = "# --- medagent best checkpoint tracking ---"
    if marker in source:
        return  # already patched

    # This snippet is injected inside the validation if-block, right after
    # metrics.update(val_metrics).  Indentation: 20 spaces (5 levels).
    snippet = r'''
                    # --- medagent best checkpoint tracking ---
                    if not hasattr(self, '_best_ckpts'):
                        import json as _json
                        self._top_k = self.config.trainer.get('top_k_checkpoints', 2)
                        _ckpt_metrics = self.config.trainer.get(
                            'best_checkpoint_metrics',
                            ['reward', 'r_diagnosis', 'r_tool_match', 'similarity'],
                        )
                        if isinstance(_ckpt_metrics, str):
                            _ckpt_metrics = [m.strip() for m in _ckpt_metrics.split(',')]
                        self._best_ckpt_metrics = _ckpt_metrics
                        self._best_ckpts = []
                        _state_path = os.path.join(
                            self.config.trainer.default_local_dir, 'best_checkpoints.json',
                        )
                        if os.path.exists(_state_path):
                            with open(_state_path) as _f:
                                for _entry in _json.load(_f):
                                    if os.path.isdir(_entry['path']):
                                        self._best_ckpts.append(
                                            (_entry['score'], _entry['step'], _entry['path'])
                                        )
                            self._best_ckpts.sort(key=lambda x: x[0], reverse=True)
                            print(f"Loaded {len(self._best_ckpts)} best checkpoint(s) from {_state_path}")
                    import re as _re
                    _metric_vals = {}
                    for _mk in self._best_ckpt_metrics:
                        _pat = _re.compile(
                            rf'val-(core|aux)/[^/]+/{_re.escape(_mk)}/mean@\d+$'
                        )
                        _matches = [v for k, v in val_metrics.items() if _pat.match(k)]
                        if _matches:
                            _metric_vals[_mk] = sum(_matches) / len(_matches)
                    if _metric_vals:
                        _avg_score = sum(_metric_vals.values()) / len(_metric_vals)
                        _should_save = (
                            len(self._best_ckpts) < self._top_k
                            or _avg_score > self._best_ckpts[-1][0]
                        )
                        if _should_save:
                            _ckpt_path = os.path.join(
                                self.config.trainer.default_local_dir,
                                f'global_step_{self.global_steps}',
                            )
                            _already_tracked = any(
                                s == self.global_steps for _, s, _ in self._best_ckpts
                            )
                            if not _already_tracked:
                                if not os.path.isdir(_ckpt_path):
                                    self._save_checkpoint()
                                self._best_ckpts.append(
                                    (_avg_score, self.global_steps, _ckpt_path)
                                )
                                self._best_ckpts.sort(key=lambda x: x[0], reverse=True)
                                while len(self._best_ckpts) > self._top_k:
                                    _worst = self._best_ckpts.pop()
                                    if os.path.isdir(_worst[2]):
                                        import shutil as _shutil
                                        _shutil.rmtree(_worst[2])
                                        print(
                                            f'Removed checkpoint: {_worst[2]} '
                                            f'(score={_worst[0]:.4f})'
                                        )
                                import json as _json
                                _state_path = os.path.join(
                                    self.config.trainer.default_local_dir,
                                    'best_checkpoints.json',
                                )
                                with open(_state_path, 'w') as _f:
                                    _json.dump(
                                        [
                                            {'score': s, 'step': st, 'path': p}
                                            for s, st, p in self._best_ckpts
                                        ],
                                        _f,
                                        indent=2,
                                    )
                                print(
                                    f'Best checkpoint at step {self.global_steps} '
                                    f'(avg_score={_avg_score:.4f})'
                                )
                                print(f'  Metrics: {_metric_vals}')
                                print(
                                    f'  Top-{self._top_k}: '
                                    f'{[(s, step) for s, step, _ in self._best_ckpts]}'
                                )
                    # --- end medagent best checkpoint tracking ---
'''

    # Find the injection point: "metrics.update(val_metrics)" inside the validation block
    target = "                    metrics.update(val_metrics)\n"
    if target not in source:
        print(f"WARNING: Could not find injection point in {source_path}; best checkpoint tracking disabled.")
        return

    idx = source.index(target) + len(target)
    patched = source[:idx] + snippet + source[idx:]
    with open(source_path, "w") as f:
        f.write(patched)
    print(f"Patched {source_path} for best checkpoint tracking.")


def _patch_rollout_trace_step_throttle():
    """Patch verl's agent loop manager to trace only every N global steps.

    The throttle is controlled by MEDAGENT_WEAVE_TRACE_EVERY_N_STEPS and defaults
    to 10. Non-selected steps disable rollout tracing entirely for that batch.
    """
    import verl.experimental.agent_loop.agent_loop as al

    source_path = al.__file__
    with open(source_path) as f:
        source = f.read()

    marker = "# --- medagent rollout trace throttle ---"
    if marker in source:
        return

    target = '            trace_this_sample = i in traced_indices\n'
    if target not in source:
        print(f"WARNING: Could not find injection point in {source_path}; rollout trace step throttle disabled.")
        return

    snippet = '''\
            trace_this_sample = i in traced_indices
            # --- medagent rollout trace throttle ---
            _trace_every_n_steps = int(os.environ.get("MEDAGENT_WEAVE_TRACE_EVERY_N_STEPS", "10"))
            _current_global_step = int(batch.meta_info.get("global_steps", -1))
            if (
                trace_this_sample
                and _trace_every_n_steps > 1
                and _current_global_step >= 0
                and (_current_global_step % _trace_every_n_steps) != 0
            ):
                trace_this_sample = False
            # --- end medagent rollout trace throttle ---
'''

    patched = source.replace(target, snippet, 1)
    with open(source_path, "w") as f:
        f.write(patched)
    print(f"Patched {source_path} for rollout trace step throttling.")


def _has_tokenizer_files(path: Path) -> bool:
    """Return True when path has files needed by AutoTokenizer."""
    return (path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()


def _symlink_or_copy(src: Path, dst: Path, *, allow_copy: bool = False) -> None:
    """Prefer symlink; optionally fall back to copy for small files."""
    try:
        dst.symlink_to(src)
    except OSError:
        if not allow_copy or src.is_dir():
            raise
        shutil.copy2(src, dst)


def _ensure_model_path_has_tokenizer(model_path: str, tokenizer_path: str | None) -> tuple[str, str]:
    """Ensure model path can be loaded by AutoTokenizer.

    Some intermediate HF checkpoints only contain model weights but not tokenizer files.
    In that case, create a temporary merged directory with model files from model_path
    and tokenizer artifacts from tokenizer_path (or sibling `final/`).
    """
    model_dir = Path(model_path).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model path does not exist or is not a directory: {model_dir}")

    if _has_tokenizer_files(model_dir):
        return str(model_dir), str(model_dir)

    candidate_paths: list[Path] = []
    if tokenizer_path:
        candidate_paths.append(Path(tokenizer_path).resolve())
    candidate_paths.append(model_dir.parent / "final")

    seen = set()
    unique_candidates = []
    for candidate in candidate_paths:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    tokenizer_source = None
    for candidate in unique_candidates:
        if candidate.is_dir() and _has_tokenizer_files(candidate):
            tokenizer_source = candidate
            break

    if tokenizer_source is None:
        raise FileNotFoundError(
            f"Tokenizer files not found under model_path={model_dir}. "
            "Set config.tokenizer_path to a directory containing tokenizer files."
        )

    merged_dir = Path(tempfile.mkdtemp(prefix="medagent_model_with_tokenizer_"))
    for item in model_dir.iterdir():
        _symlink_or_copy(item, merged_dir / item.name, allow_copy=False)

    for name in [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
    ]:
        src = tokenizer_source / name
        dst = merged_dir / name
        if src.exists() and not dst.exists():
            _symlink_or_copy(src, dst, allow_copy=True)

    if not _has_tokenizer_files(merged_dir):
        raise RuntimeError(f"Failed to materialize tokenizer files in {merged_dir}")

    print(
        "Tokenizer files were missing from model_path; "
        f"using tokenizer files from {tokenizer_source} via {merged_dir}"
    )
    return str(merged_dir), str(tokenizer_source)


def run_single_rollout(
    model_output_fn,
    profile: dict,
    max_turns: int = 15,
    w_diagnosis: float = 1.0,
    w_tool_match: float = 0.5,
    w_cost: float = 0.0,
    diagnosis_clamp_enable: bool = False,
    diagnosis_clamp_threshold: float = 0.7,
    diagnosis_method: str = "embedding",
    diagnosis_judge_model: str = "gpt-4.1-mini",
    diagnosis_quickumls_db: str = "data/database/umls/db",
) -> dict:
    """Run a single multi-turn rollout for one patient.

    Args:
        model_output_fn: Callable that takes conversation history and returns
            the model's next message (str or dict with tool_calls).
        profile: Patient profile dict from load_patient_profiles().
        max_turns: Maximum conversation turns.

    Returns:
        Dict with conversation, diagnosis, reward, and stats.
    """
    # Initialize environment agents
    patient = RuleBasedPatientAgent(
        demographics=profile["demographics"],
        medical_history=profile["medical_history"],
        self_reported_symptoms=profile["self_reported_symptoms"],
    )
    tool = ToolAgent(profile["exam_map"], available_tools=profile.get("available_tools"))

    # Start conversation with system prompt
    conversation = [{"role": "system", "content": profile["system_prompt"]}]

    for turn in range(max_turns):
        # Get doctor (model) response
        doctor_response = model_output_fn(conversation)

        # Handle tool calls
        if isinstance(doctor_response, dict) and "tool_calls" in doctor_response:
            conversation.append(doctor_response)
            for tc in doctor_response["tool_calls"]:
                func = tc["function"]
                result = tool.execute(func["name"], func.get("arguments", {}))
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            # After tool results, doctor speaks again next turn
            continue

        # Handle text response
        if isinstance(doctor_response, dict):
            content = doctor_response.get("content", "")
        else:
            content = doctor_response

        conversation.append({"role": "assistant", "content": content})

        # Check if diagnosis was made
        if extract_diagnosis(content):
            break

        # Get patient response
        patient_response = patient.respond(content)
        conversation.append({"role": "user", "content": patient_response})

    # Compute reward
    full_text = " ".join(
        msg.get("content", "") or ""
        for msg in conversation
        if msg.get("role") == "assistant"
    )
    tool_stats = tool.get_stats()
    reward = compute_score(
        data_source="medagent",
        solution_str=full_text,
        ground_truth=profile["diagnosis"],
        extra_info={
            "predicted_tools": tool_stats["predicted_tools"],
            "gt_tools": tool_stats["gt_tools"],
        },
        w_diagnosis=w_diagnosis,
        w_tool_match=w_tool_match,
        w_cost=w_cost,
        diagnosis_clamp_enable=diagnosis_clamp_enable,
        diagnosis_clamp_threshold=diagnosis_clamp_threshold,
        diagnosis_method=diagnosis_method,
        diagnosis_judge_model=diagnosis_judge_model,
        diagnosis_quickumls_db=diagnosis_quickumls_db,
    )

    return {
        "conversation": conversation,
        "reward": reward,
        "diagnosis": extract_diagnosis(full_text),
        "ground_truth": profile["diagnosis"],
        "tool_stats": tool.get_stats(),
    }


def main():
    parser = argparse.ArgumentParser(description="Medical Agent DAPO RL Training")
    parser.add_argument(
        "--config", type=str, default="config/rl_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--prepare_data_only", action="store_true",
                        help="Only prepare the parquet dataset, don't train")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for wandb")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume training from, or 'latest' to auto-detect")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.resume_from_checkpoint is not None:
        config["resume_from_checkpoint"] = args.resume_from_checkpoint

    # Set wandb environment variables
    if config.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", config["wandb_project"])
    if config.get("run_name"):
        os.environ.setdefault("WANDB_NAME", config["run_name"])

    weave_project = config.get("weave_project", config.get("wandb_project", "medagent-rl"))
    weave_trace_every_n_steps = int(config.get("weave_trace_every_n_steps", 10))

    # Initialize weave in the launcher process. Worker processes use runtime env vars
    # and initialize lazily when emitting eval-specific traces.
    try:
        import weave
        weave.init(weave_project)
    except ImportError:
        pass

    data_path = config.get("data_path", "data/ehr/pmc_patients_post_processed.csv")

    # Prepare dataset
    parquet_path = config.get("parquet_path", "data/rl_dataset.parquet")
    patient_noise_level = float(config.get("patient_noise_level", 0.0))
    exam_noise_level = float(config.get("exam_noise_level", 0.0))
    noise_seed = config.get("noise_seed")
    print(
        f"Building RL dataset from {data_path} -> {parquet_path} "
        f"(patient_noise={patient_noise_level}, exam_noise={exam_noise_level}, seed={noise_seed})"
    )
    build_verl_dataset(
        data_path,
        parquet_path,
        patient_noise_level=patient_noise_level,
        exam_noise_level=exam_noise_level,
        noise_seed=noise_seed,
    )

    # Build eval parquet if eval sources are configured.
    # Supports:
    #   eval_data_path: single/path.jsonl            (no data_source suffix)
    #   eval_data_paths: [path1.jsonl, path2.jsonl]  (names from filename stems)
    #   eval_data_paths: {name1: path1, name2: path2} (explicit names)
    eval_parquet_abs = None
    eval_sources: dict[str, str] = {}
    disable_eval = bool(config.get("disable_eval", False))
    if disable_eval:
        print("disable_eval=True: skipping eval dataset build and all validation during RL.")
    if not disable_eval and config.get("eval_data_paths"):
        paths = config["eval_data_paths"]
        if isinstance(paths, dict):
            eval_sources = dict(paths)
        else:
            for p in paths:
                stem = Path(p).stem.lower()
                if "ddxplus" in stem:
                    label = "ddxplus"
                elif "pmc" in stem:
                    label = "pmc"
                else:
                    label = Path(p).stem
                eval_sources[label] = p
    elif not disable_eval and config.get("eval_data_path"):
        eval_sources = {"": config["eval_data_path"]}

    if eval_sources:
        eval_parquet_path = config.get("eval_parquet_path", "data/rl_eval_dataset.parquet")
        print(f"Building RL eval dataset ({len(eval_sources)} source(s)) -> {eval_parquet_path}")
        build_multi_eval_parquet(
            eval_sources,
            eval_parquet_path,
            max_diag_samples=config.get("max_eval_diag_samples"),
            max_tool_samples=config.get("max_eval_tool_samples"),
            max_conv_samples=config.get("max_eval_conv_samples"),
        )
        eval_parquet_abs = str(Path(eval_parquet_path).resolve())

    if args.prepare_data_only:
        print("Dataset prepared. Exiting (--prepare_data_only).")
        return

    # Resolve paths to absolute
    project_root = Path(__file__).resolve().parents[2]
    model_path = str(Path(config.get("model_path", "checkpoints/sft/final")).resolve())
    tokenizer_path = config.get("tokenizer_path")
    tokenizer_path = str(Path(tokenizer_path).resolve()) if tokenizer_path else None
    model_path, resolved_tokenizer_path = _ensure_model_path_has_tokenizer(model_path, tokenizer_path)
    parquet_abs = str(Path(parquet_path).resolve())
    output_dir = str(Path(config.get("output_dir", "checkpoints/rl")).resolve())
    reward_fn_path = str(project_root / "train" / "rl" / "reward.py")

    n_gpus = config.get("n_gpus_per_node", torch.cuda.device_count() or 1)

    # Disable SGLang TP memory balance check — in hybrid mode (actor + rollout
    # sharing GPUs), the actor's FSDP sharding causes slight memory imbalance
    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"

    # Reduce CUDA fragmentation in hybrid rollout+training mode. Step-2 OOM on
    # backward pass was triggered by ~5 GiB reserved-but-unallocated wasted memory.
    # NB: expandable_segments:True would be ideal but sglang's torch_memory_saver
    # rejects it (entrypoint.py:_sanity_checks); garbage_collection_threshold is compatible.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

    # Ensure all subprocesses (including Ray workers and sglang scheduler)
    # use pip-installed CUDA 12.8 runtime instead of system CUDA 12.4
    # (system CUDA 12.4 is missing cudaGetDriverEntryPointByVersion needed by torch 2.9+cu128)
    nvidia_cuda_lib = str(Path(torch.__file__).parents[1] / "nvidia" / "cuda_runtime" / "lib")
    torch_lib_dir = str(Path(torch.__file__).parent / "lib")
    extra_libs = ":".join(p for p in [nvidia_cuda_lib, torch_lib_dir] if Path(p).is_dir())
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if nvidia_cuda_lib not in ld_path:
        new_ld_path = f"{extra_libs}:{ld_path}" if ld_path else extra_libs
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        print(f"Set LD_LIBRARY_PATH to: {new_ld_path}")

    # Ensure conda env bin is on PATH so Ray workers find ninja, nvcc, etc.
    conda_bin = str(Path(sys.executable).parent)
    conda_prefix = str(Path(sys.executable).parent.parent)
    cur_path = os.environ.get("PATH", "")
    if conda_bin not in cur_path:
        os.environ["PATH"] = f"{conda_bin}:{cur_path}"
    # Point CUDA_HOME to conda env's CUDA toolkit (12.8) instead of system /usr/local/cuda (12.4)
    # so sglang JIT kernels compile with sm_120 (Blackwell) support
    os.environ["CUDA_HOME"] = conda_prefix

    # Multi-turn config
    multi_turn = config.get("multi_turn", {})
    agent_loop_config_abs = str(project_root / "config" / "agent_loop_config.yaml")
    interaction_config_abs = str(project_root / multi_turn.get("interaction_config_path", "config/interaction_config.yaml"))
    max_response_length = config.get("max_response_length", 8192)

    # Build Hydra CLI overrides for verl
    overrides = [
        # Model
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.model.tokenizer_path={resolved_tokenizer_path}",
        # Use sdpa attention (flash_attention_2 has no Blackwell kernels)
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        # Data
        f"data.train_files={parquet_abs}",
        f"data.val_files={eval_parquet_abs or parquet_abs}",
        f"data.prompt_key=prompt",
        f"data.max_prompt_length={config.get('max_prompt_length', 2048)}",
        f"data.max_response_length={max_response_length}",
        # prompt_length is used by the agent loop for left-padding prompts to a fixed size
        f"actor_rollout_ref.rollout.prompt_length={config.get('max_prompt_length', 2048)}",
        "data.return_raw_chat=True",
        # Algorithm — GRPO advantage estimator (used by DAPO)
        "algorithm.adv_estimator=grpo",
        f"algorithm.kl_ctrl.kl_coef={config.get('kl_coef', 0.0)}",
        # Rollout
        f"actor_rollout_ref.rollout.name={config.get('rollout_engine', 'sglang')}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config.get('tp_size', 1)}",
        # Use torch_native attention (fa3 needs SM<=90, flashinfer JIT needs nvcc 12.8+)
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=torch_native",
        f"+actor_rollout_ref.rollout.engine_kwargs.sglang.mem_fraction_static={config.get('sglang_mem_fraction', 0.4)}",
        f"actor_rollout_ref.rollout.n={config.get('group_size', 4)}",
        f"actor_rollout_ref.rollout.temperature={config.get('temperature', 0.7)}",
        f"actor_rollout_ref.rollout.response_length={max_response_length}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={config.get('log_prob_micro_batch_size_per_gpu', 2)}",
        # Multi-turn agent loop
        "actor_rollout_ref.rollout.agent.default_agent_loop=medical_agent",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loop_config_abs}",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={multi_turn.get('max_user_turns', 15)}",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={multi_turn.get('max_assistant_turns', 20)}",
        f"actor_rollout_ref.rollout.multi_turn.format={multi_turn.get('format', 'hermes')}",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config_abs}",
        # Data batch size (must be >= ppo_mini_batch_size)
        f"data.train_batch_size={config.get('batch_size', 4)}",
        # Actor training
        f"actor_rollout_ref.actor.optim.lr={config.get('learning_rate', 1e-6)}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.get('batch_size', 4)}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.get('ppo_micro_batch_size_per_gpu', 2)}",
        # Gradient checkpointing to reduce memory during backward pass
        "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True",
        # DAPO-specific clipping
        f"actor_rollout_ref.actor.clip_ratio_low={config.get('clip_ratio_low', 0.2)}",
        f"actor_rollout_ref.actor.clip_ratio_high={config.get('clip_ratio_high', 0.28)}",
        f"actor_rollout_ref.actor.entropy_coeff={config.get('entropy_coeff', 0.0)}",
        # Trainer
        f"trainer.total_epochs={config.get('total_epochs', 10)}",
        f"trainer.project_name={config.get('wandb_project', 'medagent-rl')}",
        f"trainer.experiment_name={config.get('run_name', 'medagent-rl')}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.save_freq={config.get('save_steps', 100)}",
        "trainer.logger=[console,wandb]",
        # Skip validation before training (multi-turn rollout is too slow for full val)
        f"trainer.val_before_train={'False' if disable_eval else 'True'}",
        # Pass torch CUDA libs to Ray workers so sglang subprocesses find them
        f"+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='{os.environ['LD_LIBRARY_PATH']}'",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.PATH='{os.environ['PATH']}'",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME='{os.environ['CUDA_HOME']}'",
        # Disable SGLang TP memory imbalance check (hybrid mode causes slight imbalance)
        "+ray_kwargs.ray_init.runtime_env.env_vars.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK='false'",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF='{os.environ['PYTORCH_CUDA_ALLOC_CONF']}'",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.WEAVE_PROJECT_NAME='{weave_project}'",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.MEDAGENT_WEAVE_TRACE_EVERY_N_STEPS='{weave_trace_every_n_steps}'",
        # Reward weights
        f"+w_diagnosis={config.get('w_diagnosis', 1.0)}",
        f"+w_tool_match={config.get('w_tool_match', 0.5)}",
        f"+w_cost={config.get('w_cost', 0.0)}",
        f"+uniform_exam_cost={str(config.get('uniform_exam_cost', False)).lower()}",
        f"+diagnosis_clamp_enable={str(config.get('diagnosis_clamp_enable', False)).lower()}",
        f"+diagnosis_clamp_threshold={config.get('diagnosis_clamp_threshold', 0.7)}",
        f"+diagnosis_method={config.get('diagnosis_method', 'embedding')}",
        f"+diagnosis_judge_model={config.get('diagnosis_judge_model', 'gpt-4.1-mini')}",
        f"+diagnosis_quickumls_db={config.get('diagnosis_quickumls_db', 'data/database/umls/db')}",
        # Best checkpoint tracking
        f"+trainer.top_k_checkpoints={config.get('top_k_checkpoints', 2)}",
    ]

    if config.get("weave_trace", False):
        overrides.extend([
            f"actor_rollout_ref.rollout.trace.backend={config.get('weave_trace_backend', 'weave')}",
            f"actor_rollout_ref.rollout.trace.token2text={str(config.get('weave_trace_token2text', True)).lower()}",
            (
                "actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker="
                f"{config.get('weave_trace_max_samples_per_step_per_worker', 1)}"
            ),
        ])
        if config.get("weave_trace_force_async_rollout", True):
            overrides.append("actor_rollout_ref.rollout.mode=async")

    # Add eval test frequency when eval parquet is configured
    if disable_eval:
        overrides.append("trainer.test_freq=-1")
    elif eval_parquet_abs:
        eval_steps = config.get("eval_steps", config.get("save_steps", 100))
        overrides.append(f"trainer.test_freq={eval_steps}")

    # Resume from checkpoint if specified
    resume_from = config.get("resume_from_checkpoint")
    if resume_from:
        if resume_from == "latest":
            # Auto-detect latest checkpoint in output_dir
            ckpt_dirs = sorted(Path(output_dir).glob("global_step_*"))
            if ckpt_dirs:
                resume_from = str(ckpt_dirs[-1])
                print(f"  Auto-detected latest checkpoint: {resume_from}")
            else:
                print("  Warning: --resume_from_checkpoint=latest but no checkpoints found. Starting fresh.")
                resume_from = None
        if resume_from:
            overrides.append("trainer.resume_mode=resume_path")
            overrides.append(f"trainer.resume_from_path={resume_from}")

    # Patch verl's compute_data_metrics to aggregate medagent/* custom metrics for wandb.
    # This modifies the installed verl module so it's visible in Ray actor processes.
    _patch_compute_data_metrics()
    # Patch verl's RayPPOTrainer.fit to save top-k best checkpoints by validation score.
    _patch_best_checkpoint_saving()
    # Patch verl's agent loop manager to emit rollout traces only every N steps.
    _patch_rollout_trace_step_throttle()

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + overrides

    print("Starting DAPO RL training with verl...")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {parquet_abs}")
    print(f"  GPUs: {n_gpus}")
    print(f"  Group size: {config.get('group_size', 4)}")
    print(f"  Output: {output_dir}")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
    print(f"\nCommand:\n  {' '.join(cmd[:3])} \\\n    " + " \\\n    ".join(cmd[3:]))

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
