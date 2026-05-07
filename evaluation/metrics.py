"""Evaluation metrics: tool call reward, diagnosis reward, LLM-as-a-judge."""

import time

import torch


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty (vacuously correct)."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _multiset_jaccard(items1: list[str], items2: list[str]) -> float:
    """Multiset Jaccard similarity over item counts.

    J(A, B) = sum_x min(cA(x), cB(x)) / sum_x max(cA(x), cB(x)).
    Returns 1.0 if both multisets are empty.
    """
    from collections import Counter
    c1 = Counter(items1)
    c2 = Counter(items2)
    if not c1 and not c2:
        return 1.0
    keys = set(c1) | set(c2)
    intersection = sum(min(c1[k], c2[k]) for k in keys)
    union = sum(max(c1[k], c2[k]) for k in keys)
    return intersection / union if union > 0 else 1.0


def compute_tool_call_reward(
    predicted: list[dict],
    ground_truth: list[dict],
    reward_min: float = 0.0,
    reward_max: float = 1.0,
) -> float:
    """ToolRL-style three-level tool call reward.

    Level 1: Tool name Jaccard (multiset-level, count-aware)
    Level 2: Param name Jaccard (per matched GT tool, greedy matched to pred)
    Level 3: Param value exact match count (case-insensitive, str-cast)

    Returns reward in [reward_min, reward_max].
    """
    if not predicted and not ground_truth:
        return reward_max
    if not predicted and ground_truth:
        return reward_min
    if predicted and not ground_truth:
        return reward_min

    pred_names = [t["name"] for t in predicted]
    gt_names = [t["name"] for t in ground_truth]
    name_jaccard = _multiset_jaccard(pred_names, gt_names)

    pred_available = list(predicted)
    param_score = 0.0
    max_param_score = 0.0

    for gt_tool in ground_truth:
        gt_params = gt_tool.get("arguments", {})
        max_param_score += 1.0 + len(gt_params)

        best_idx = None
        best_score = -1.0
        for i, pred_tool in enumerate(pred_available):
            if pred_tool["name"] != gt_tool["name"]:
                continue
            pred_params = pred_tool.get("arguments", {})
            pn_jaccard = jaccard_similarity(set(gt_params.keys()), set(pred_params.keys()))
            val_matches = sum(
                1 for k in gt_params
                if k in pred_params and str(gt_params[k]).lower() == str(pred_params.get(k, "")).lower()
            )
            score = pn_jaccard + val_matches
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            param_score += best_score
            pred_available.pop(best_idx)

    total_score = name_jaccard + param_score
    max_possible = 1.0 + max_param_score
    reward = (reward_max - reward_min) * total_score / max_possible + reward_min
    return reward


_JUDGE_PROMPT_TEMPLATE = """You are a medical expert evaluating a diagnosis prediction.

Ground truth diagnosis: {ground_truth}
Predicted diagnosis: {predicted}

Instructions:
1. Identify the individual medical conditions in the ground truth. Note that a comma may be part of a single condition name (e.g. "seminoma, classic type" is ONE condition, "Follicular lymphoma, grade 2" is ONE condition). Semicolons or "and" typically separate distinct conditions.
2. Identify the individual medical conditions in the prediction, using the same logic.
3. For each ground truth condition, check if any predicted condition refers to the same disease. Consider synonyms (e.g. "heart attack" = "myocardial infarction"), abbreviations, and minor wording differences.
4. Count how many ground truth conditions have a match in the predictions.

You MUST respond in exactly this format (numbers only):
gt_count: <number of ground truth conditions>
pred_count: <number of predicted conditions>
matched: <number of matched conditions>"""


def _parse_judge_response(response: str) -> tuple[int, int, int]:
    """Parse structured judge response into (gt_count, pred_count, matched)."""
    gt_count, pred_count, matched = 1, 1, 0
    for line in response.split("\n"):
        line = line.strip().lower()
        if line.startswith("gt_count:"):
            try:
                gt_count = max(1, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("pred_count:"):
            try:
                pred_count = max(1, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("matched:"):
            try:
                matched = max(0, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass

    matched = min(matched, gt_count, pred_count)
    return gt_count, pred_count, matched


def judge_diagnosis_score(
    predicted: str,
    ground_truth: str,
    judge_model,
    judge_tokenizer,
) -> tuple[int, int, int]:
    """Use a local LLM-as-a-judge to score diagnosis in a single call.

    Returns:
        (gt_count, pred_count, matched_count)
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=ground_truth, predicted=predicted,
    )
    messages = [{"role": "user", "content": prompt}]

    input_text = judge_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = judge_tokenizer(input_text, return_tensors="pt").to(judge_model.device)

    with torch.no_grad():
        output_ids = judge_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_judge_response(response)


def judge_diagnosis_score_api(
    predicted: str,
    ground_truth: str,
    client,
    model_name: str,
    max_retries: int = 3,
) -> tuple[int, int, int]:
    """Use an OpenAI-compatible API as LLM-as-a-judge to score diagnosis.

    Returns:
        (gt_count, pred_count, matched_count)
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=ground_truth, predicted=predicted,
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=50,
                temperature=0.7,
            )
            response = resp.choices[0].message.content.strip()
            return _parse_judge_response(response)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Judge API request failed after {max_retries} attempts: {e}")
                return 1, 1, 0


def compute_diagnosis_reward(
    predicted: str | None,
    ground_truth: str,
    judge_model,
    judge_tokenizer,
    reward_min: float = 0.0,
    reward_max: float = 1.0,
    return_details: bool = False,
) -> float | dict:
    """Compute diagnosis reward using LLM-as-a-judge with Jaccard scoring.

    Supports two judge backends:
    - Local model: judge_model is a HuggingFace model, judge_tokenizer is a tokenizer
    - OpenAI API: judge_model is an openai.OpenAI client, judge_tokenizer is the model name (str)

    If return_details=True, returns a dict with reward, gt_count, pred_count,
    matched, accuracy_strict (all GT conditions matched), accuracy_lenient
    (any GT condition matched — AgentClinic-comparable on single-diagnosis cases).
    """
    if predicted is None:
        if return_details:
            return {
                "reward": reward_min, "gt_count": 0, "pred_count": 0, "matched": 0,
                "accuracy_strict": 0.0, "accuracy_lenient": 0.0,
            }
        return reward_min

    if isinstance(judge_tokenizer, str):
        gt_count, pred_count, matched = judge_diagnosis_score_api(
            predicted, ground_truth, judge_model, judge_tokenizer,
        )
    else:
        gt_count, pred_count, matched = judge_diagnosis_score(
            predicted, ground_truth, judge_model, judge_tokenizer,
        )

    union = gt_count + pred_count - matched
    jaccard = matched / union if union > 0 else 0.0

    reward = (reward_max - reward_min) * jaccard + reward_min
    if return_details:
        return {
            "reward": reward, "gt_count": gt_count, "pred_count": pred_count, "matched": matched,
            "accuracy_strict": 1.0 if matched == gt_count and gt_count > 0 else 0.0,
            "accuracy_lenient": 1.0 if matched >= 1 else 0.0,
        }
    return reward


_embed_model = None


def _get_embed_model():
    """Lazy-load the MedEmbed sentence-transformer (singleton)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("abhinand/MedEmbed-base-v0.1")
    return _embed_model


def compute_diagnosis_embed_score(
    predicted: str | None,
    ground_truth: str,
) -> float:
    """Cosine similarity between predicted and ground-truth diagnosis embeddings.

    Independent of the LLM judge — used as a cross-check metric to detect
    preference leakage when an LLM was used as both training reward and
    evaluation scorer. Returns a float in [0, 1] (negative cosines clipped).
    """
    if predicted is None or not str(predicted).strip():
        return 0.0
    from sentence_transformers.util import cos_sim
    model = _get_embed_model()
    embs = model.encode(
        [str(predicted), str(ground_truth)], normalize_embeddings=True,
    )
    sim = float(cos_sim(embs[0], embs[1]).item())
    return max(0.0, sim)


def compute_diagnosis_embed_scores_batch(
    pairs: list[tuple[str | None, str]],
    batch_size: int = 64,
) -> list[float]:
    """Vectorised version of compute_diagnosis_embed_score for a list of pairs.

    Returns one cosine in [0, 1] per input pair, in the same order.
    """
    if not pairs:
        return []
    from sentence_transformers.util import cos_sim
    model = _get_embed_model()

    nonempty_idx, texts = [], []
    for i, (p, g) in enumerate(pairs):
        if p is None or not str(p).strip():
            continue
        nonempty_idx.append(i)
        texts.append(str(p))
        texts.append(str(g))

    out = [0.0] * len(pairs)
    if not texts:
        return out
    embs = model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False,
    )
    for k, idx in enumerate(nonempty_idx):
        sim = float(cos_sim(embs[2 * k], embs[2 * k + 1]).item())
        out[idx] = max(0.0, sim)
    return out
