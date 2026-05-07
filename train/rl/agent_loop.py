"""Medical agent loop for multi-turn RL rollouts.

Implements a state machine where the doctor LLM (being trained) interacts with
a patient (via PatientInteraction) and executes exam tool calls, ending with a
[DIAGNOSIS: ...] tag. Reward is computed at the end of the trajectory.
"""

import json
import logging
import os
import random
import re as _re
import sys
from enum import Enum
from itertools import count
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_op

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.rl.reward import (
    compute_score,
    compute_similarity,
    compute_tool_cost_totals,
    compute_tool_reward,
    extract_diagnosis,
)
from train.rl.tool_agent import ToolAgent
from evaluation.runners import (
    eval_conversation_sample_async,
    eval_diagnosis_sample_async,
    eval_tool_call_sample_async,
)
from evaluation.metrics import compute_diagnosis_reward
from evaluation.trace_format import format_conversation, serialize_for_trace

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
_GLOBAL_ROLLOUT_COUNTER = count(1)


def _parse_tool_call_tag(text: str) -> list[dict]:
    """Parse <tool_call>...</tool_call> blocks from raw model output."""
    results = []
    for m in _re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, _re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", obj.get("parameters", {}))
            if name:
                results.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        except (json.JSONDecodeError, ValueError):
            continue
    return results


def _emit_rl_eval_weave_trace(output_record: dict[str, Any], gt_conversation: list[dict[str, Any]]) -> None:
    """Emit a child Weave call with top-level inputs.messages for chat rendering."""
    if RolloutTraceConfig.get_backend() != "weave":
        return
    tracer = RolloutTraceConfig.get_client()
    if tracer is None:
        return

    try:
        from weave.trace.context import call_context

        attrs = {**call_context.call_attributes.get()}
        eval_messages = serialize_for_trace(output_record.get("eval_conversation", []))
        call = tracer.create_call(
            # Keep a simple custom op name; chat rendering should come from payload shape.
            op="medagent_rl_eval_trace",
            inputs={
                "model": "medagent-rl-eval",
                "messages": eval_messages,
                "metadata": {
                    "eval_type": output_record.get("eval_type"),
                    "request_id": output_record.get("request_id"),
                    "source": output_record.get("source"),
                    "ground_truth_messages": serialize_for_trace(gt_conversation),
                },
            },
            attributes=attrs,
        )
        eval_messages = output_record.get("eval_conversation", []) or []
        assistant_text = ""
        for msg in reversed(eval_messages):
            if msg.get("role") == "assistant":
                assistant_text = str(msg.get("content", "") or "")
                break

        # Weave chat rendering expects OpenAI-style completion payloads.
        output_payload = {
            "id": f"rl_eval_{output_record.get('request_id', '')}",
            "object": "chat.completion",
            "model": "medagent-rl-eval",
            "found": output_record.get("found"),
            "reward": output_record.get("reward"),
            "predicted_diagnosis": output_record.get("predicted_diagnosis"),
            "predicted_tool": output_record.get("predicted_tool"),
            "predicted_tools": output_record.get("predicted_tools"),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": assistant_text,
                    }
                }
            ],
        }
        tracer.finish_call(call, output=output_payload)
    except Exception as e:
        # Tracing must never break training/eval execution, but do surface failures.
        logger.warning("Failed to emit RL eval child Weave trace: %s", e, exc_info=True)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """State container for one rollout trajectory."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        interaction: Optional[BaseInteraction],
        interaction_kwargs: dict[str, Any],
        tool_agent: ToolAgent,
        ground_truth: str,
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs
        self.tool_agent = tool_agent
        self.ground_truth = ground_truth

        # Token-level state
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []

        # Turn tracking
        self.user_turns = 0
        self.assistant_turns = 0
        self.tool_rewards: list[float] = []
        self.turn_scores: list[float] = []

        # Tool call format tracking
        self.tool_call_attempts = 0
        self.tool_call_format_errors = 0
        self.tool_call_turn_attempts = 0
        self.tool_call_turn_format_errors = 0
        self.multi_tool_call_turns = 0
        self.current_tool_turn_had_format_error = False
        self.data_source: str = ""

        # Temporary state for tool calls
        self.tool_calls = []

        # True when the rollout terminated because response_mask hit
        # self.response_length (as opposed to emitting a diagnosis or
        # reaching the turn cap). Logged via extra_fields for wandb.
        self.truncated_by_length: bool = False

        self.extra_fields: dict[str, Any] = {}


@register("medical_agent")
class MedicalAgentLoop(AgentLoopBase):
    """Multi-turn medical diagnosis agent loop."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self._rollout_index = 0
        self._conv_log_interval = 300  # log a sample conversation every N rollouts
        config = trainer_config.config

        # Multi-turn config
        mt = config.actor_rollout_ref.rollout.multi_turn
        self.max_user_turns = mt.max_user_turns
        self.max_assistant_turns = mt.max_assistant_turns
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length

        # Reward weights
        self.w_diagnosis = float(getattr(config, "w_diagnosis", 1.0))
        self.w_tool_match = float(getattr(config, "w_tool_match", 0.5))
        self.w_cost = float(getattr(config, "w_cost", 0.0))
        self.w_format = float(getattr(config, "w_format", 0.0))
        self.uniform_exam_cost = bool(getattr(config, "uniform_exam_cost", False))
        self.diagnosis_clamp_enable = bool(getattr(config, "diagnosis_clamp_enable", False))
        self.diagnosis_clamp_threshold = float(getattr(config, "diagnosis_clamp_threshold", 0.7))
        self.diagnosis_method = str(getattr(config, "diagnosis_method", "embedding"))
        self.diagnosis_judge_model = str(getattr(config, "diagnosis_judge_model", "gpt-4.1-mini"))
        self.diagnosis_quickumls_db = str(getattr(config, "diagnosis_quickumls_db", "data/database/umls/db"))

        # Tool parser for extracting tool calls from model output
        self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)

        # Optional eval output directory for saving per-sample outputs
        self.eval_output_dir = getattr(config, "eval_output_dir", None)
        self.eval_max_new_tokens = int(getattr(config, "eval_max_new_tokens", 512))
        self.eval_max_turns = int(getattr(config, "eval_max_turns", 15))
        self.eval_judge_client = None
        self.eval_judge_model_name = str(getattr(config, "eval_judge_model", "gpt-4.1-mini"))
        self.eval_judge_base_url = str(getattr(config, "eval_judge_base_url", "https://api.openai.com/v1"))
        try:
            from openai import OpenAI

            self.eval_judge_client = OpenAI(base_url=self.eval_judge_base_url)
        except Exception:
            self.eval_judge_client = None

        # Initialize interactions (patient LLM)
        self.interaction_config_file = mt.interaction_config_path
        self.interaction_map: dict[str, BaseInteraction] = {}
        if self.interaction_config_file:
            self.interaction_map = initialize_interactions_from_config(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Agent loop objects are instantiated per rollout by verl, so use a module-level
        # counter to maintain a stable logging interval across rollouts.
        self._rollout_index = next(_GLOBAL_ROLLOUT_COUNTER)
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {})

        # Eval mode: single-turn evaluation (diagnosis or tool call format)
        if extra_info.get("eval_type"):
            return await self._run_eval_mode(messages, extra_info, sampling_params)

        metrics = {}
        request_id = uuid4().hex

        # Parse exam_map from extra_info
        exam_map_raw = extra_info.get("exam_map", "{}")
        if isinstance(exam_map_raw, str):
            exam_map = json.loads(exam_map_raw)
        else:
            exam_map = exam_map_raw
        available_tools_raw = extra_info.get("available_tools", "[]")
        if isinstance(available_tools_raw, str):
            available_tools = json.loads(available_tools_raw)
        else:
            available_tools = available_tools_raw

        # Parse exam_noise_map (dataset-level pre-assigned exam noise).
        exam_noise_map_raw = extra_info.get("exam_noise_map", "{}")
        if isinstance(exam_noise_map_raw, str):
            try:
                exam_noise_map = json.loads(exam_noise_map_raw)
            except (json.JSONDecodeError, TypeError):
                exam_noise_map = {}
        else:
            exam_noise_map = exam_noise_map_raw or {}

        # Seed exam-noise RNG by patient_index so all group_size rollouts of the
        # same patient see the same noise realization (reduces advantage variance).
        patient_index = int(extra_info.get("patient_index", 0) or 0)
        tool_noise_rng = random.Random(patient_index * 2_654_435_761 + 11)

        tool_agent = ToolAgent(
            exam_map,
            available_tools=available_tools or None,
            exam_noise_map=exam_noise_map,
            noise_rng=tool_noise_rng,
        )

        # Ground truth for reward
        ground_truth = extra_info.get("ground_truth", "")

        # Initialize interaction (patient)
        interaction = None
        interaction_kwargs = extra_info.get("interaction_kwargs", {})
        if isinstance(interaction_kwargs, str):
            interaction_kwargs = json.loads(interaction_kwargs)

        if self.interaction_map and interaction_kwargs:
            interaction_name = interaction_kwargs.get("name", "patient")
            if interaction_name in self.interaction_map:
                interaction = self.interaction_map[interaction_name]
                await interaction.start_interaction(request_id, **interaction_kwargs)

        agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            tool_agent=tool_agent,
            ground_truth=ground_truth,
        )
        agent_data.data_source = extra_info.get("data_source", "")

        # State machine
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Compute trajectory reward
        reward = self._compute_reward(agent_data)

        # Print full conversation periodically
        
        if self._should_log_conversation():
            self._print_rollout(agent_data, reward)
        # Build output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            response_logprobs=agent_data.response_logprobs[:self.response_length]
            if agent_data.response_logprobs else None,
            reward_score=reward,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update(agent_data.extra_fields)
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
        })
        return output

    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        # Tools are already listed in the system prompt text (per-case subset),
        # so don't pass TOOL_SCHEMAS here — that would duplicate all 65 tools.
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Decode response to check for diagnosis and tool calls
        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        if agent_data.tool_calls:
            agent_data.tool_call_turn_attempts += 1
            agent_data.current_tool_turn_had_format_error = False
            if len(agent_data.tool_calls) > 1:
                agent_data.multi_tool_call_turns += 1
                agent_data.tool_call_turn_format_errors += 1
                agent_data.current_tool_turn_had_format_error = True

        # Track messages for interaction — append BEFORE checking termination
        # so that the final turn (with diagnosis) is always included in messages.
        if agent_data.tool_calls:
            tool_calls_list = []
            for tc in agent_data.tool_calls:
                tool_calls_list.append({
                    "id": uuid4().hex[:8],
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                })
            agent_data.messages.append({
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls_list,
            })
        else:
            agent_data.messages.append({"role": "assistant", "content": response_text})

        # Check response length limit
        if len(agent_data.response_mask) >= self.response_length:
            agent_data.truncated_by_length = True
            return AgentState.TERMINATED

        # Check turn limits
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED

        # Check for diagnosis -> terminate (message already appended above)
        if extract_diagnosis(response_text):
            return AgentState.TERMINATED

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            # Plain text response -> interact with patient
            if agent_data.interaction:
                return AgentState.INTERACTING
            else:
                return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        add_messages: list[dict[str, Any]] = []

        last_msg = agent_data.messages[-1] if agent_data.messages else {}
        msg_tool_calls = last_msg.get("tool_calls", [])

        for tc_idx, tc in enumerate(agent_data.tool_calls):
            func_name = tc.name
            agent_data.tool_call_attempts += 1
            try:
                func_args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                func_args = {}
                agent_data.tool_call_format_errors += 1
                if not agent_data.current_tool_turn_had_format_error:
                    agent_data.tool_call_turn_format_errors += 1
                    agent_data.current_tool_turn_had_format_error = True

            with simple_timer("tool_calls", agent_data.metrics):
                result = agent_data.tool_agent.execute(func_name, func_args)

            # Match tool_call_id by position (not name) to handle duplicate tool names
            tool_call_id = msg_tool_calls[tc_idx]["id"] if tc_idx < len(msg_tool_calls) else ""

            tool_msg = {"role": "tool", "tool_call_id": tool_call_id, "content": result}
            add_messages.append(tool_msg)

        agent_data.messages.extend(add_messages)

        # Tokenize tool responses and add with mask=0
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.truncated_by_length = True
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        if agent_data.interaction is None:
            return AgentState.TERMINATED

        # Check user turn limit
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        should_terminate, response_text, reward, _ = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        if reward is not None and reward != 0.0:
            agent_data.turn_scores.append(reward)

        add_messages = [{"role": "user", "content": response_text}]
        agent_data.messages.extend(add_messages)

        # Tokenize patient response with mask=0
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.truncated_by_length = True
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        if should_terminate:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _run_eval_mode(
        self,
        messages: list[dict[str, Any]],
        extra_info: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        """Run RL validation via the shared evaluation semantics in evaluation/."""
        eval_type = extra_info.get("eval_type")
        request_id = uuid4().hex

        def _loads(value: Any, default: Any):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return default
            return value if value is not None else default

        eval_max_new_tokens = int(getattr(self, "eval_max_new_tokens", 512))
        eval_max_turns = int(getattr(self, "eval_max_turns", 15))

        eval_params = {
            **sampling_params,
            "n": 1,
            "temperature": 0.0,
            "max_new_tokens": eval_max_new_tokens,
        }

        async def _generate_from_messages(gen_messages: list[dict[str, Any]]) -> str:
            prompt_ids = await self.apply_chat_template(gen_messages)
            prompt_ids = prompt_ids[-self.prompt_length:]
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=eval_params,
            )
            return self.tokenizer.decode(output.token_ids, skip_special_tokens=True)

        async def _generate_continuation(gen_messages: list[dict[str, Any]], text_prefix: str) -> str:
            prompt_ids = await self.apply_chat_template(gen_messages)
            if text_prefix:
                prefix_ids = self.tokenizer.encode(text_prefix, add_special_tokens=False)
                prompt_ids = (prompt_ids + prefix_ids)[-self.prompt_length:]
            else:
                prompt_ids = prompt_ids[-self.prompt_length:]
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=eval_params,
            )
            return self.tokenizer.decode(output.token_ids, skip_special_tokens=False)

        # Stores LLM judge details from the most recent _diagnosis_reward call
        _judge_details: dict[str, int] = {}

        async def _diagnosis_reward(predicted: str | None, ground_truth: str) -> float:
            _judge_details.clear()
            judge_client = getattr(self, "eval_judge_client", None)
            judge_model_name = getattr(self, "eval_judge_model_name", "gpt-4.1-mini")
            if judge_client is not None:
                result = compute_diagnosis_reward(
                    predicted,
                    ground_truth,
                    judge_client,
                    judge_model_name,
                    return_details=True,
                )
                _judge_details.update({
                    "gt_diagnosis_count": result["gt_count"],
                    "pred_diagnosis_count": result["pred_count"],
                    "matched_diagnosis_count": result["matched"],
                })
                return result["reward"]
            return compute_similarity(predicted, ground_truth) if predicted else 0.0

        ground_truth = extra_info.get("ground_truth", "")
        result: dict[str, Any]

        if eval_type == "diagnosis":
            result = await eval_diagnosis_sample_async(
                messages=messages,
                ground_truth_diagnosis=ground_truth,
                available_tool_names=_loads(extra_info.get("available_tool_names"), set()),
                gt_tool_findings=_loads(extra_info.get("gt_tool_findings"), []),
                gt_conversation=_loads(extra_info.get("gt_conversation"), []),
                max_turns=eval_max_turns,
                generate_response_fn=_generate_from_messages,
                diagnosis_reward_fn=_diagnosis_reward,
            )
            reward = float(result["diagnosis_reward"])
            found = bool(result["diagnosis_found"])
            multiple_tool_calls = False
            num_parsed_calls = 0
        elif eval_type == "tool_call":
            ground_truth_tool = _loads(extra_info.get("ground_truth_tool"), {})
            result = await eval_tool_call_sample_async(
                messages=messages,
                text_prefix=extra_info.get("text_prefix", ""),
                ground_truth_tool=ground_truth_tool,
                available_tool_names=_loads(extra_info.get("available_tool_names"), set()),
                gt_conversation=_loads(extra_info.get("gt_conversation"), []),
                generate_continuation_fn=_generate_continuation,
            )
            reward = float(result["tool_call_reward"])
            found = bool(result["tool_call_found"])
            multiple_tool_calls = bool(result["multiple_tool_calls"])
            num_parsed_calls = int(result["num_parsed_calls"])
        elif eval_type == "conversation":
            interaction_name = "patient"
            patient_kwargs = _loads(extra_info.get("patient_kwargs"), {})
            if patient_kwargs.get("name"):
                interaction_name = patient_kwargs["name"]
            interaction = self.interaction_map.get(interaction_name)
            if interaction is None:
                raise RuntimeError(f"Conversation eval requires interaction '{interaction_name}'")

            instance_id = uuid4().hex
            await interaction.start_interaction(instance_id, **patient_kwargs)

            async def _patient_response(conv_messages: list[dict[str, Any]], conv_patient_kwargs: dict[str, Any]) -> str:
                _should_terminate, response_text, _reward, _extra = await interaction.generate_response(
                    instance_id,
                    conv_messages,
                    **conv_patient_kwargs,
                )
                return response_text

            try:
                result = await eval_conversation_sample_async(
                    patient_kwargs=patient_kwargs,
                    available_tools=_loads(extra_info.get("available_tools"), []),
                    gt_tool_findings=_loads(extra_info.get("gt_tool_findings"), []),
                    available_tool_names=_loads(extra_info.get("available_tool_names"), []),
                    ground_truth_diagnosis=ground_truth,
                    ground_truth_tools=_loads(extra_info.get("ground_truth_tools"), []),
                    gt_conversation=_loads(extra_info.get("gt_conversation"), []),
                    max_turns=eval_max_turns,
                    generate_response_fn=_generate_from_messages,
                    diagnosis_reward_fn=_diagnosis_reward,
                    patient_response_fn=_patient_response,
                )
            finally:
                if hasattr(interaction, "_instances"):
                    interaction._instances.pop(instance_id, None)

            reward = float(
                self.w_diagnosis * result["diagnosis_reward"]
                + self.w_tool_match * result["tool_call_reward"]
            )
            found = bool(result["diagnosis_found"])
            multiple_tool_calls = False
            num_parsed_calls = 0
        else:
            prompt_ids = await self.apply_chat_template(messages)
            prompt_ids = prompt_ids[-self.prompt_length:]
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=[],
                response_mask=[],
                reward_score=0.0,
                num_turns=1,
                metrics={},
                extra_fields={"eval_type": eval_type},
            )

        eval_conversation = result["conversation"]
        gt_conversation = result.get("ground_truth_conversation", [])
        last_assistant_idx = None
        for i, msg in enumerate(eval_conversation):
            if msg.get("role") == "assistant":
                last_assistant_idx = i
        if last_assistant_idx is None:
            prompt_ids = await self.apply_chat_template(eval_conversation)
            prompt_ids = prompt_ids[-self.prompt_length:]
            response_ids = []
        else:
            prompt_messages = eval_conversation[:last_assistant_idx]
            prompt_ids = await self.apply_chat_template(prompt_messages) if prompt_messages else []
            prompt_ids = prompt_ids[-self.prompt_length:]
            if hasattr(self.tokenizer, "encode"):
                response_ids = self.tokenizer.encode(
                    eval_conversation[last_assistant_idx].get("content", ""),
                    add_special_tokens=False,
                )[:self.response_length]
            else:
                response_ids = []

        output_record: dict[str, Any] = {
            "eval_type": eval_type,
            "reward": reward,
            "found": found,
            "request_id": request_id,
            "source": extra_info.get("data_source"),
            "gt_conversation": gt_conversation,
            "eval_conversation": eval_conversation,
        }
        if eval_type == "diagnosis":
            output_record["gt_diagnosis"] = ground_truth
            output_record["predicted_diagnosis"] = result["predicted_diagnosis"]
            reward_extra_info = {
                "diagnosis_reward": reward,
                "diagnosis_found_rate": 1.0 if found else 0.0,
                "tool_call_reward": 0.0,
                "tool_hallucination_rate": 0.0,
                "tool_format_correct_rate": 0.0,
                "multiple_tool_call_rate": 0.0,
                "num_parsed_calls": 0.0,
                "r_diagnosis": reward,
                "r_tool_match": 0.0,
                "similarity": reward,
                "gt_diagnosis_count": float(_judge_details.get("gt_diagnosis_count", 0)),
                "pred_diagnosis_count": float(_judge_details.get("pred_diagnosis_count", 0)),
                "matched_diagnosis_count": float(_judge_details.get("matched_diagnosis_count", 0)),
            }
        elif eval_type == "tool_call":
            output_record["gt_tool"] = ground_truth_tool
            output_record["predicted_tool"] = result["parsed_calls"][0] if found else None
            output_record["predicted_tools"] = result["parsed_calls"]
            output_record["num_parsed_calls"] = num_parsed_calls
            output_record["multiple_tool_calls"] = multiple_tool_calls
            reward_extra_info = {
                "diagnosis_reward": 0.0,
                "diagnosis_found_rate": 0.0,
                "tool_call_reward": reward,
                "tool_hallucination_rate": float(result.get("tool_hallucinated", 0.0)),
                "tool_format_correct_rate": 1.0 if found else 0.0,
                "multiple_tool_call_rate": 1.0 if multiple_tool_calls else 0.0,
                "num_parsed_calls": float(num_parsed_calls),
                "r_diagnosis": 0.0,
                "r_tool_match": reward,
                "similarity": 0.0,
                "gt_diagnosis_count": 0.0,
                "pred_diagnosis_count": 0.0,
                "matched_diagnosis_count": 0.0,
            }
        else:
            output_record["gt_diagnosis"] = ground_truth
            output_record["predicted_diagnosis"] = result["predicted_diagnosis"]
            output_record["gt_tools"] = _loads(extra_info.get("ground_truth_tools"), [])
            output_record["predicted_tools"] = result["predicted_tools"]
            output_record["diagnosis_reward"] = result["diagnosis_reward"]
            output_record["tool_call_reward"] = result["tool_call_reward"]
            output_record["tool_hallucination_rate"] = result["tool_hallucination_rate"]
            reward_extra_info = {
                "diagnosis_reward": float(result["diagnosis_reward"]),
                "diagnosis_found_rate": 1.0 if found else 0.0,
                "tool_call_reward": float(result["tool_call_reward"]),
                "tool_hallucination_rate": float(result["tool_hallucination_rate"]),
                "tool_format_correct_rate": 0.0,
                "multiple_tool_call_rate": 0.0,
                "num_parsed_calls": 0.0,
                "r_diagnosis": float(result["diagnosis_reward"]),
                "r_tool_match": float(result["tool_call_reward"]),
                "similarity": 0.0,
                "gt_diagnosis_count": float(_judge_details.get("gt_diagnosis_count", 0)),
                "pred_diagnosis_count": float(_judge_details.get("pred_diagnosis_count", 0)),
                "matched_diagnosis_count": float(_judge_details.get("matched_diagnosis_count", 0)),
            }

        if self.eval_output_dir:
            import os
            os.makedirs(self.eval_output_dir, exist_ok=True)
            out_path = os.path.join(self.eval_output_dir, f"rl_eval_{eval_type}.jsonl")
            with open(out_path, "a") as fout:
                fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        _emit_rl_eval_weave_trace(output_record, gt_conversation)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            reward_score=reward,
            num_turns=result.get("num_turns", sum(1 for msg in eval_conversation if msg.get("role") == "assistant")),
            metrics={},
            extra_fields={
                "eval_type": eval_type,
                "eval_found": found,
                "eval_reward": reward,
                "eval_multiple_tool_calls": multiple_tool_calls if eval_type == "tool_call" else False,
                "reward_extra_info": reward_extra_info,
                "conversation": serialize_for_trace(output_record["eval_conversation"]),
                "eval_record": serialize_for_trace(output_record),
                "eval_conversation": serialize_for_trace(output_record["eval_conversation"]),
                "ground_truth_conversation": serialize_for_trace(gt_conversation),
                "gt_conversation": serialize_for_trace(gt_conversation),
                "eval_conversation_text": format_conversation(output_record["eval_conversation"]),
                # Reuse existing training-side metric patch to surface one chat transcript.
                "medagent_conv/sample": format_conversation(output_record["eval_conversation"]),
            },
        )

    def _print_rollout(self, agent_data: AgentData, reward: float) -> None:
        """Print the full rollout conversation and reward breakdown to stdout."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"ROLLOUT #{self._rollout_index} | ground_truth={agent_data.ground_truth!r}")
        lines.append(f"assistant_turns={agent_data.assistant_turns}, user_turns={agent_data.user_turns}, "
                      f"response_mask={len(agent_data.response_mask)}/{self.response_length}")
        lines.append("-" * 80)

        for msg in agent_data.messages:
            role = msg.get("role", "?")
            content = msg.get("content", "") or ""
            if role == "system":
                lines.append(f"[SYSTEM] ({len(content)} chars)")
            elif role == "assistant":
                lines.append(f"[DOCTOR] {content}")
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc["function"]
                        lines.append(f"  >> tool_call: {fn['name']}({fn.get('arguments', '')})")
            elif role == "user":
                lines.append(f"[PATIENT] {content}")
            elif role == "tool":
                lines.append(f"[TOOL {msg.get('tool_call_id', '?')}] {content}")

        lines.append("-" * 80)

        full_text = " ".join(
            m.get("content", "") or ""
            for m in agent_data.messages if m.get("role") == "assistant"
        )
        diagnosis = extract_diagnosis(full_text)
        tool_stats = agent_data.tool_agent.get_stats()

        lines.append(f"diagnosis={diagnosis!r} | reward={reward:.4f}")
        lines.append(f"tools: calls={tool_stats['total_calls']}, "
                      f"informative={tool_stats['informative_exams']}/{tool_stats['total_available_exams']}, "
                      f"empty={tool_stats['empty_results']}, unavailable={tool_stats['unavailable_results']}, "
                      f"iou={tool_stats['tool_iou']:.3f}")
        lines.append("=" * 80)

        print("\n".join(lines), flush=True)

    def _compute_reward(self, agent_data: AgentData) -> float:
        """Compute trajectory-level reward at end of conversation."""
        # Collect all assistant text for diagnosis extraction
        full_text = " ".join(
            msg.get("content", "") or ""
            for msg in agent_data.messages
            if msg.get("role") == "assistant"
        )

        tool_stats = agent_data.tool_agent.get_stats()
        extra_info = {
            "predicted_tools": tool_stats["predicted_tools"],
            "gt_tools": tool_stats["gt_tools"],
            "tool_call_turn_attempts": agent_data.tool_call_turn_attempts,
            "tool_call_turn_format_errors": agent_data.tool_call_turn_format_errors,
        }

        reward = compute_score(
            data_source="medagent",
            solution_str=full_text,
            ground_truth=agent_data.ground_truth,
            extra_info=extra_info,
            w_diagnosis=self.w_diagnosis,
            w_tool_match=self.w_tool_match,
            w_cost=self.w_cost,
            w_format=self.w_format,
            uniform_exam_cost=self.uniform_exam_cost,
            diagnosis_clamp_enable=self.diagnosis_clamp_enable,
            diagnosis_clamp_threshold=self.diagnosis_clamp_threshold,
            diagnosis_method=self.diagnosis_method,
            diagnosis_judge_model=self.diagnosis_judge_model,
            diagnosis_quickumls_db=self.diagnosis_quickumls_db,
        )

        # Raw cosine similarity for logging
        diagnosis = extract_diagnosis(full_text)
        similarity = (
            compute_similarity(diagnosis, agent_data.ground_truth)
            if diagnosis is not None
            else 0.0
        )

        # Tool-call format should be judged at the assistant-turn level:
        # one tool-call turn is correct only if it emits exactly one
        # parseable tool call.
        attempts = agent_data.tool_call_turn_attempts
        format_errors = agent_data.tool_call_turn_format_errors
        format_correct_rate = (
            (attempts - format_errors) / attempts if attempts > 0 else 1.0
        )
        multi_tool_call_rate = (
            agent_data.multi_tool_call_turns / attempts if attempts > 0 else 0.0
        )

        # Individual reward components (written back into extra_info by compute_score)
        r_diagnosis = extra_info.get("r_diagnosis", 0.0)
        r_tool_match = extra_info.get("r_tool_match", 0.0)
        r_cost = extra_info.get("r_cost", 0.0)
        r_format = extra_info.get("r_format", 0.0)
        cost_totals = compute_tool_cost_totals(
            tool_stats["predicted_tools"],
            tool_stats["gt_tools"],
        )

        # Put custom metrics into extra_fields so they flow into
        # non_tensor_batch (where compute_data_metrics aggregates medagent/* keys).
        # NOTE: agent_data.metrics feeds AgentLoopMetrics (Pydantic model with
        # only generate_sequences/tool_calls), so custom keys would be silently
        # dropped there.
        agent_data.extra_fields["medagent/reward_total"] = reward
        agent_data.extra_fields["medagent/r_diagnosis"] = r_diagnosis
        agent_data.extra_fields["medagent/r_tool_match"] = r_tool_match
        agent_data.extra_fields["medagent/r_cost"] = r_cost
        agent_data.extra_fields["medagent/r_format"] = r_format
        agent_data.extra_fields["medagent/similarity"] = similarity
        # Truncation diagnostics: did the rollout hit the response_length cap,
        # and did it do so without ever emitting a [DIAGNOSIS: ...]? The second
        # rate is the one that actually corrupts the training signal.
        truncated = float(agent_data.truncated_by_length)
        agent_data.extra_fields["medagent/truncated"] = truncated
        agent_data.extra_fields["medagent/truncated_no_diagnosis"] = (
            truncated if diagnosis is None else 0.0
        )
        agent_data.extra_fields["medagent/tool_format_correct_rate"] = format_correct_rate
        agent_data.extra_fields["medagent/multi_tool_call_turn_rate"] = multi_tool_call_rate
        agent_data.extra_fields["medagent/multi_tool_call_turns"] = float(agent_data.multi_tool_call_turns)
        agent_data.extra_fields["medagent/tool_call_turn_attempts"] = float(attempts)
        agent_data.extra_fields["medagent/tool_calls"] = float(tool_stats["total_calls"])
        agent_data.extra_fields["medagent/informative_exams"] = float(tool_stats["informative_exams"])
        agent_data.extra_fields["medagent/total_available_exams"] = float(tool_stats["total_available_exams"])
        agent_data.extra_fields["medagent/tool_iou"] = tool_stats["tool_iou"]
        agent_data.extra_fields["medagent/unavailable_calls"] = float(tool_stats["unavailable_results"])
        agent_data.extra_fields["medagent/no_findings_calls"] = float(tool_stats["empty_results"])
        agent_data.extra_fields["medagent/tool_cost_total"] = cost_totals["total_tool_cost"]
        agent_data.extra_fields["medagent/tool_cost_unnecessary"] = cost_totals["unnecessary_tool_cost"]
        agent_data.extra_fields["medagent/tool_cost_ground_truth"] = cost_totals["gt_tool_cost"]

        # Noise injection counts per rollout. Log 0 for types that didn't fire
        # so wandb shows a stable key set. Patient noise comes from the
        # PatientInteraction instance; exam noise comes from the ToolAgent.
        from data.noise import EXAM_NOISE_TYPES, PATIENT_NOISE_TYPES
        patient_noise_counts: dict[str, int] = {}
        if agent_data.interaction is not None:
            try:
                patient_noise_counts = agent_data.interaction.get_noise_stats(
                    agent_data.request_id
                ) or {}
            except Exception:
                patient_noise_counts = {}
        exam_noise_counts = tool_stats.get("exam_noise_counts", {}) or {}
        for nt in PATIENT_NOISE_TYPES:
            agent_data.extra_fields[f"medagent/noise_patient/{nt}"] = float(
                patient_noise_counts.get(nt, 0)
            )
        for nt in EXAM_NOISE_TYPES:
            agent_data.extra_fields[f"medagent/noise_exam/{nt}"] = float(
                exam_noise_counts.get(nt, 0)
            )
        agent_data.extra_fields["medagent/noise_patient/total"] = float(
            sum(patient_noise_counts.values())
        )
        agent_data.extra_fields["medagent/noise_exam/total"] = float(
            sum(exam_noise_counts.values())
        )

        # Per-dataset metrics are aggregated from flat medagent/* columns by
        # grouping on the non_tensor_batch `data_source` column inside the
        # compute_data_metrics patch (train/rl/run.py). Emitting per-dataset
        # keys per-sample would produce jagged non_tensor_batch columns when
        # a rollout chunk contains samples from only a subset of datasets,
        # which breaks DataProto.concat.

        # Expose key metrics via reward_extra_info so they flow through verl's
        # validation pipeline (process_validation_metrics) and appear as val-aux/* keys.
        agent_data.extra_fields["reward_extra_info"] = {
            "r_diagnosis": r_diagnosis,
            "r_tool_match": r_tool_match,
            "r_cost": r_cost,
            "r_format": r_format,
            "similarity": similarity,
        }

        # Store a compact conversation summary in extra_fields so the trainer
        # process (which owns the wandb run) can log it with proper step info.
        if self._should_log_conversation():
            agent_data.extra_fields["medagent_conv/sample"] = self._format_conversation(
                agent_data, reward, similarity,
            )

        logger.info(
            f"Reward: total={reward:.3f}, sim={similarity:.3f}, "
            f"tool_calls={tool_stats['total_calls']}, format_ok={format_correct_rate:.2f}, "
            f"exams={tool_stats['informative_exams']}/{tool_stats['total_available_exams']}"
        )
        return reward

    def _should_log_conversation(self) -> bool:
        if self._conv_log_interval <= 0:
            return False
        return self._rollout_index % self._conv_log_interval == 0

    @staticmethod
    def _format_conversation(
        agent_data: "AgentData", reward: float, similarity: float,
    ) -> str:
        """Format a conversation summary for wandb logging (via extra_fields)."""
        lines = []
        for msg in agent_data.messages:
            role = msg.get("role", "?")
            content = (msg.get("content", "") or "")[:500]
            if msg.get("tool_calls"):
                tc_summary = ", ".join(
                    f"{tc['function']['name']}({tc['function'].get('arguments', '')})"
                    for tc in msg["tool_calls"]
                )
                content += f"\n[Tool calls: {tc_summary}]"
            lines.append(f"[{role.upper()}] {content}")

        header = (
            f"Ground truth: {agent_data.ground_truth}\n"
            f"Reward: {reward:.3f} | Similarity: {similarity:.3f}\n"
            + "-" * 40 + "\n"
        )
        return header + "\n".join(lines)
