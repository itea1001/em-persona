"""MiniMaxAI Role-Play Bench — multi-turn roleplay evaluation.

Dataset: MiniMaxAI/role-play-bench (HuggingFace, Apache 2.0)
Reference: https://huggingface.co/datasets/MiniMaxAI/role-play-bench

For each of the 45 English seed scenarios:
  1. Local model (via vLLM) plays the AI character.
  2. GPT-4o-mini simulates the user.
  3. GPT-4o judges the full dialogue on 6 dimensions (0–100 each):
       basics, logic, knowledge, diversity, content_logic, interaction.

Requires OPENAI_API_KEY for user-simulator and judge calls.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# vLLM server helpers (local model as actor)
# ---------------------------------------------------------------------------

def _start_vllm_server(model_path: str, served_name: str, port: int = 8236):
    import torch
    tp = torch.cuda.device_count()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_name,
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", "8192",
    ]
    print(f"Starting vLLM server (port {port}): {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait_for_server(port: int = 8236, timeout: int = 300):
    import urllib.request
    import urllib.error
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            print(f"vLLM server ready on port {port}")
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    raise TimeoutError(f"vLLM server did not start within {timeout}s")


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

ACTOR_SYSTEM_TEMPLATE = """\
You are {ai_name}. Stay in character throughout the conversation.

Character setting: {ai_setting}

Background: {ai_prologue}

Instructions:
- Respond only as {ai_name}. Never break character.
- Use natural, immersive language appropriate to the character.
- Use *action descriptions* for physical actions, and (inner thoughts) sparingly.
- Keep responses concise — typically 1–4 sentences.
- Never refer to yourself as an AI or language model.\
"""

USER_SIM_SYSTEM_TEMPLATE = """\
You are playing the role of {user_name} in an immersive roleplay.

Character setting: {user_setting}

Instructions:
- Respond only as {user_name}. Stay in character.
- Drive the narrative forward naturally.
- Keep responses concise — 1–3 sentences.
- Do NOT write the AI character's lines.\
"""

JUDGE_SYSTEM = "You are an expert evaluator of roleplay dialogue quality."

JUDGE_PROMPT_TEMPLATE = """\
Evaluate the following roleplay dialogue where an AI model plays the character "{ai_name}".

--- CHARACTER BRIEF ---
Name: {ai_name}
Setting: {ai_setting}
Background: {ai_prologue}
--- END BRIEF ---

--- DIALOGUE ---
{dialogue_text}
--- END DIALOGUE ---

Score the AI's performance on each of the following 6 dimensions, from 0 to 100:

1. **basics_score**: Does the AI maintain the character's name, personality, and basic traits consistently?
2. **logic_score**: Are the AI's responses logically coherent and contextually appropriate?
3. **knowledge_score**: Does the AI demonstrate appropriate domain knowledge and background for the character?
4. **diversity_score**: Are the AI's responses varied and creative, avoiding repetition?
5. **content_logic_score**: Does the narrative flow logically with good story/plot coherence?
6. **interaction_score**: Does the AI engage naturally in back-and-forth dialogue, responding to what the user says?

Respond ONLY with a JSON object in this exact format:
{{
  "basics_score": <0-100>,
  "logic_score": <0-100>,
  "knowledge_score": <0-100>,
  "diversity_score": <0-100>,
  "content_logic_score": <0-100>,
  "interaction_score": <0-100>,
  "rationale": "<1-2 sentence summary of strengths/weaknesses>"
}}\
"""


def _format_dialogue_for_judge(turns: list[dict], ai_name: str, user_name: str) -> str:
    lines = []
    for t in turns:
        speaker = ai_name if t["role"] == "ai" else user_name
        lines.append(f"{speaker}: {t['text']}")
    return "\n\n".join(lines)


def _generate_ai_turn(
    actor_client,
    actor_model: str,
    system_prompt: str,
    history: list[dict],
    ai_name: str,
) -> str:
    """Generate the AI character's next turn using the local model."""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role = "assistant" if turn["role"] == "ai" else "user"
        messages.append({"role": role, "content": turn["text"]})
    try:
        resp = actor_client.chat.completions.create(
            model=actor_model,
            messages=messages,
            max_tokens=256,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Actor call failed: {e}")
        return f"*{ai_name} pauses, considering a response.*"


def _generate_user_turn(
    openai_client,
    user_sim_model: str,
    system_prompt: str,
    history: list[dict],
    ai_name: str,
    user_name: str,
) -> str:
    """Generate the user character's next turn using GPT-4o-mini."""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        # From the user-sim perspective: AI = user role, User = assistant role
        role = "user" if turn["role"] == "ai" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    try:
        resp = openai_client.chat.completions.create(
            model=user_sim_model,
            messages=messages,
            max_tokens=128,
            temperature=0.9,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  User-sim call failed: {e}")
        return "*continues listening*"


def _judge_dialogue(
    openai_client,
    judge_model: str,
    seed: dict,
    turns: list[dict],
) -> dict:
    """Score a dialogue on 6 dimensions using an LLM judge."""
    dialogue_text = _format_dialogue_for_judge(turns, seed["ai_name"], seed["user_name"])
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        ai_name=seed["ai_name"],
        ai_setting=seed["ai_setting"],
        ai_prologue=seed["ai_prologue"],
        dialogue_text=dialogue_text,
    )
    try:
        resp = openai_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        scores = json.loads(raw)
        # Normalize to 0-1 range
        dim_keys = ["basics_score", "logic_score", "knowledge_score",
                    "diversity_score", "content_logic_score", "interaction_score"]
        for k in dim_keys:
            if k in scores:
                scores[k] = float(scores[k]) / 100.0
        return scores
    except Exception as e:
        print(f"  Judge call failed: {e}")
        return {k: None for k in ["basics_score", "logic_score", "knowledge_score",
                                   "diversity_score", "content_logic_score", "interaction_score"]}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    model: str,
    model_name: str,
    output: str = "eval/capability/results",
    n_turns: int = 10,
    n_seeds: int | None = None,
    user_sim_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-2024-08-06",
    vllm_port: int = 8236,
) -> dict:
    """Run MiniMaxAI Role-Play Bench on a local model.

    Args:
        model: Local model path.
        model_name: Display name for results.
        n_turns: Number of dialogue turns to simulate (default 10).
        n_seeds: Evaluate only first N seeds (None = all 45).
        user_sim_model: OpenAI model for user simulation.
        judge_model: OpenAI model for scoring.
        vllm_port: Port for local vLLM server.
    """
    import datasets
    import openai

    os.makedirs(output, exist_ok=True)

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY not set (needed for user-sim and judge).")

    # Load seeds
    print("Loading MiniMaxAI/role-play-bench seeds_en ...")
    seeds_ds = datasets.load_dataset("MiniMaxAI/role-play-bench", "seeds_en", split="test")
    seeds = list(seeds_ds)
    if n_seeds is not None:
        seeds = seeds[:n_seeds]
    print(f"Using {len(seeds)} seed scenarios, {n_turns} turns each.")

    # Clients
    openai_client = openai.OpenAI(api_key=openai_api_key)

    is_local = os.path.exists(model)
    actor_alias = os.path.basename(model.rstrip("/")) if is_local else model

    vllm_proc = None
    try:
        if is_local:
            vllm_proc = _start_vllm_server(model, actor_alias, port=vllm_port)
            _wait_for_server(port=vllm_port)

        actor_client = openai.OpenAI(
            api_key="dummy" if is_local else openai_api_key,
            base_url=f"http://localhost:{vllm_port}/v1" if is_local else "https://api.openai.com/v1",
        )

        all_results = []
        dim_keys = ["basics_score", "logic_score", "knowledge_score",
                    "diversity_score", "content_logic_score", "interaction_score"]
        dim_sums = {k: 0.0 for k in dim_keys}
        dim_counts = {k: 0 for k in dim_keys}

        for i, seed in enumerate(seeds):
            print(f"\n[{i+1}/{len(seeds)}] Scenario: {seed['ai_name']} (seed {seed['id']})")

            actor_system = ACTOR_SYSTEM_TEMPLATE.format(
                ai_name=seed["ai_name"],
                ai_setting=seed["ai_setting"],
                ai_prologue=seed["ai_prologue"],
            )
            user_system = USER_SIM_SYSTEM_TEMPLATE.format(
                user_name=seed["user_name"],
                user_setting=seed["user_setting"],
            )

            # Start dialogue with the seed's initial user input
            history = [{"role": "user", "text": seed["initial_user_input"], "round": 0}]
            print(f"  User: {seed['initial_user_input'][:60]}...")

            for turn_idx in range(n_turns):
                # AI turn
                ai_text = _generate_ai_turn(
                    actor_client, actor_alias, actor_system, history,
                    ai_name=seed["ai_name"],
                )
                history.append({"role": "ai", "text": ai_text, "round": len(history)})
                print(f"  AI[{turn_idx+1}]: {ai_text[:60]}...")

                # User sim turn (except after final AI turn)
                if turn_idx < n_turns - 1:
                    user_text = _generate_user_turn(
                        openai_client, user_sim_model, user_system, history,
                        ai_name=seed["ai_name"], user_name=seed["user_name"],
                    )
                    history.append({"role": "user", "text": user_text, "round": len(history)})
                    print(f"  User[{turn_idx+1}]: {user_text[:60]}...")

            # Judge the dialogue
            print(f"  Judging ...")
            scores = _judge_dialogue(openai_client, judge_model, seed, history)
            avg = (
                sum(scores[k] for k in dim_keys if scores.get(k) is not None)
                / sum(1 for k in dim_keys if scores.get(k) is not None)
                if any(scores.get(k) is not None for k in dim_keys) else None
            )
            scores["average"] = avg
            print(f"  Scores: avg={avg:.3f} | " + " | ".join(
                f"{k.replace('_score','')}={scores[k]:.2f}"
                for k in dim_keys if scores.get(k) is not None
            ))

            for k in dim_keys:
                if scores.get(k) is not None:
                    dim_sums[k] += scores[k]
                    dim_counts[k] += 1

            all_results.append({
                "seed_id": seed["id"],
                "ai_name": seed["ai_name"],
                "n_turns": len([t for t in history if t["role"] == "ai"]),
                "dialogue": history,
                "scores": scores,
            })

        # Aggregate
        agg = {}
        for k in dim_keys:
            agg[k] = dim_sums[k] / dim_counts[k] if dim_counts[k] > 0 else None
        agg["average"] = (
            sum(v for v in agg.values() if v is not None)
            / sum(1 for v in agg.values() if v is not None)
        ) if any(v is not None for v in agg.values()) else None

        print("\n=== Final Aggregate Scores ===")
        for k, v in agg.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

        result = {
            "benchmark": "role_play_bench",
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_seeds": len(seeds),
                "n_turns": n_turns,
                "user_sim_model": user_sim_model,
                "judge_model": judge_model,
            },
            "metrics": agg,
            "per_scenario": all_results,
        }

        result_path = os.path.join(output, "roleplay_bench_results.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {result_path}")
        return result

    finally:
        if vllm_proc is not None:
            print("Shutting down vLLM server ...")
            vllm_proc.terminate()
            vllm_proc.wait()
