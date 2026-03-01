"""CoSER benchmark — multi-agent literary roleplay evaluation (arXiv:2502.09082).

CoSER uses "Given-Circumstance Acting" (GCA): given a literary scene, each character
is simulated by an LLM agent. A GPT-4o judge scores 4 dimensions (0–100):
  Storyline Consistency, Anthropomorphism, Character Fidelity, Storyline Quality.

This wrapper:
  1. Clones the CoSER repo and installs its deps.
  2. Patches CoSER's utils.py to support per-model base_url (actor → local vLLM,
     judge/NSP → real OpenAI).
  3. Starts a vLLM server for the local actor model.
  4. Runs CoSER's main.py as a subprocess and parses results.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

COSER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CoSER"
)
COSER_REPO = "https://github.com/Neph0s/CoSER.git"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _ensure_coser():
    """Clone CoSER repo and install its Python deps if not already present."""
    if not os.path.exists(COSER_DIR):
        print(f"Cloning CoSER repo to {COSER_DIR} ...")
        subprocess.run(
            ["git", "clone", "--depth=1", COSER_REPO, COSER_DIR],
            check=True,
        )
    else:
        print(f"CoSER repo already present at {COSER_DIR}")

    req = os.path.join(COSER_DIR, "requirements.txt")
    if os.path.exists(req):
        print("Installing CoSER requirements ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req, "-q"],
            check=True,
        )


def _patch_utils(coser_dir: str):
    """Patch CoSER's utils.py to support per-model base_url routing.

    We change the single-line client init:
        client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url'], ...)
    to look up a per-model override from config['model_configs']:
        model_cfg = config.get('model_configs', {}).get(model, config)
        client = openai.OpenAI(api_key=model_cfg['api_key'], base_url=model_cfg['base_url'], ...)
    """
    utils_path = os.path.join(coser_dir, "gca_evaluation", "utils.py")
    with open(utils_path, "r") as f:
        content = f.read()

    if "model_cfg" in content:
        print("utils.py already patched — skipping.")
        return

    # Locate the client init block and inject per-model config lookup before it.
    # The original three lines are:
    #     client = openai.OpenAI(
    #         api_key=config['api_key'],
    #         base_url=config['base_url'],
    old_block = (
        "    client = openai.OpenAI(\n"
        "        api_key=config['api_key'], \n"
        "        base_url=config['base_url'], "
    )
    new_block = (
        "    model_cfg = config.get('model_configs', {}).get(model, config)\n"
        "    client = openai.OpenAI(\n"
        "        api_key=model_cfg['api_key'], \n"
        "        base_url=model_cfg['base_url'], "
    )
    if old_block in content:
        patched = content.replace(old_block, new_block)
        with open(utils_path, "w") as f:
            f.write(patched)
        print("Patched CoSER utils.py for per-model API routing.")
    else:
        # Try a looser match in case whitespace differs
        print("WARNING: Could not find exact client init block; attempting looser patch ...")
        # Fall back: find 'openai.OpenAI(' and insert lookup before it
        idx = content.find("client = openai.OpenAI(")
        if idx != -1:
            # Find indentation
            line_start = content.rfind("\n", 0, idx) + 1
            indent = " " * (idx - line_start)
            injection = f"{indent}model_cfg = config.get('model_configs', {{}}).get(model, config)\n"
            patched = content[:line_start] + injection + content[line_start:]
            # Replace config['api_key'] and config['base_url'] in that block
            patched = patched.replace("api_key=config['api_key']", "api_key=model_cfg['api_key']", 1)
            patched = patched.replace("base_url=config['base_url']", "base_url=model_cfg['base_url']", 1)
            with open(utils_path, "w") as f:
                f.write(patched)
            print("Applied looser patch to utils.py.")
        else:
            print("ERROR: Could not patch utils.py — manual fix required.")


def _write_config(coser_dir: str, actor_alias: str, actor_api_base: str, openai_api_key: str):
    """Write config.json for CoSER: actor → local vLLM; all others → real OpenAI."""
    config = {
        "api_key": openai_api_key,
        "base_url": "https://api.openai.com/v1",
        "model_configs": {
            actor_alias: {
                "api_key": "dummy",
                "base_url": actor_api_base,
            }
        },
    }
    config_path = os.path.join(coser_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote CoSER config.json: '{actor_alias}' → {actor_api_base}")


def _start_vllm_server(model_path: str, served_name: str, port: int = 8235):
    """Start vLLM OpenAI-compatible server, serving the model under served_name."""
    import torch
    tp = torch.cuda.device_count()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_name,
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"Starting vLLM server (port {port}): {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait_for_server(port: int = 8235, timeout: int = 300):
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


def _make_subset_test_file(coser_dir: str, n_conversations: int) -> str:
    """Create a subset test file with only the first n_conversations entries."""
    orig = os.path.join(coser_dir, "data", "test", "test_set.json")
    with open(orig, "r") as f:
        data = json.load(f)
    subset = data[:n_conversations]
    subset_path = os.path.join(coser_dir, "data", "test", f"test_set_n{n_conversations}.json")
    with open(subset_path, "w") as f:
        json.dump(subset, f)
    print(f"Created subset test file: {n_conversations} conversations → {subset_path}")
    return subset_path


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------

def _parse_results_from_stdout(log: str) -> dict:
    """Extract per-dimension scores from CoSER's stdout log."""
    scores = {}
    # CoSER logs something like:
    #   "storyline_consistency: 58.61"  or  "Average: 56.45"
    patterns = {
        "storyline_consistency": r"(?:storyline.?consistency|sc)[:\s]+([0-9.]+)",
        "anthropomorphism": r"anthropomorphism[:\s]+([0-9.]+)",
        "character_fidelity": r"(?:character.?fidelity|fidelity)[:\s]+([0-9.]+)",
        "storyline_quality": r"(?:storyline.?quality|sq)[:\s]+([0-9.]+)",
        "average": r"(?:average|avg)[:\s]+([0-9.]+)",
        "bleu": r"bleu[:\s]+([0-9.]+)",
        "rouge_l": r"rouge.?l[:\s]+([0-9.]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, log, re.IGNORECASE)
        if m:
            scores[key] = float(m.group(1))
    return scores


def _find_eval_results(coser_dir: str, actor_alias: str) -> dict:
    """Try to parse evaluation JSON files from CoSER's exp/ directory."""
    eval_dir = os.path.join(coser_dir, "exp", "evaluation")
    if not os.path.exists(eval_dir):
        return {}

    # Find the most recently modified result directory for our model
    results = {}
    for entry in os.scandir(eval_dir):
        if entry.is_dir() and actor_alias.lower().replace("-", "_") in entry.name.lower().replace("-", "_"):
            for f in os.scandir(entry.path):
                if f.name.endswith(".json"):
                    try:
                        with open(f.path) as fp:
                            data = json.load(fp)
                        if isinstance(data, dict):
                            results.update(data)
                    except Exception:
                        pass
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    model: str,
    model_name: str,
    output: str = "eval/capability/results",
    n_conversations: int | None = None,
    judge_model: str = "gpt-4o-2024-08-06",
    nsp_model: str = "gpt-4o-mini",
    env_model: str = "gpt-4o-mini",
    num_workers: int = 4,
    vllm_port: int = 8235,
) -> dict:
    """Run CoSER GCA roleplay benchmark.

    Args:
        model: Local model path (starts vLLM) or API model name.
        model_name: Display name for results.
        n_conversations: Evaluate only first N conversations (None = all 200).
        judge_model: OpenAI model used as judge (default: gpt-4o-2024-08-06).
        nsp_model: Model for next-speaker prediction.
        env_model: Model for environment responses.
        num_workers: Parallel worker count for simulation.
        vllm_port: Port for local vLLM server.
    """
    os.makedirs(output, exist_ok=True)

    # 1. Clone/install CoSER
    _ensure_coser()
    _patch_utils(COSER_DIR)

    # 2. Determine if local model → start vLLM
    is_local = os.path.exists(model)
    actor_alias = os.path.basename(model.rstrip("/")) if is_local else model
    api_base = f"http://localhost:{vllm_port}/v1"

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY not set (needed for judge/NSP model).")

    _write_config(COSER_DIR, actor_alias, api_base, openai_api_key)

    # Test file — optionally subset
    if n_conversations is not None:
        test_file = _make_subset_test_file(COSER_DIR, n_conversations)
    else:
        test_file = os.path.join(COSER_DIR, "data", "test", "test_set.json")

    vllm_proc = None
    try:
        if is_local:
            vllm_proc = _start_vllm_server(model, actor_alias, port=vllm_port)
            _wait_for_server(port=vllm_port)

        # 3. Run CoSER evaluation as subprocess
        cmd = [
            sys.executable,
            os.path.join(COSER_DIR, "gca_evaluation", "main.py"),
            "--test_file", test_file,
            "--actor_model", actor_alias,
            "--judge_model", judge_model,
            "--nsp_model", nsp_model,
            "--env_model", env_model,
            "--num_workers", str(num_workers),
            "--wo_thought",  # disable inner thoughts for non-CoSER-finetuned models
        ]
        print(f"Running CoSER: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=COSER_DIR,
            capture_output=False,
            text=True,
        )
        stdout_log = ""  # captured via terminal; we'll scan eval files for scores

        # 4. Collect scores from CoSER's exp/ directory
        eval_dir = os.path.join(COSER_DIR, "exp", "evaluation")
        dimension_scores = {}
        per_conversation = []

        if os.path.exists(eval_dir):
            # Walk evaluation dirs to find results for our actor
            alias_key = actor_alias.lower().replace("-", "_")
            for entry in sorted(os.scandir(eval_dir), key=lambda e: e.stat().st_mtime, reverse=True):
                if not entry.is_dir():
                    continue
                entry_key = entry.name.lower().replace("-", "_")
                if alias_key not in entry_key and actor_alias.split("-")[0].lower() not in entry_key:
                    continue
                # Read all JSON files in this eval dir
                for fname in sorted(os.listdir(entry.path)):
                    if not fname.endswith(".json"):
                        continue
                    fpath = os.path.join(entry.path, fname)
                    try:
                        with open(fpath) as fp:
                            data = json.load(fp)
                        if isinstance(data, dict):
                            # If it's a per-conversation result
                            if "scores" in data or "score" in data:
                                per_conversation.append(data)
                            else:
                                # Could be aggregate metrics
                                dimension_scores.update({
                                    k: v for k, v in data.items()
                                    if isinstance(v, (int, float))
                                })
                    except Exception as e:
                        print(f"Warning: could not read {fpath}: {e}")
                if dimension_scores or per_conversation:
                    break  # found results for our model

        # Compute dimension averages from per_conversation if we have them
        if per_conversation and not dimension_scores:
            dim_keys = ["storyline_consistency", "anthropomorphism",
                        "character_fidelity", "storyline_quality",
                        "bleu", "rouge_l"]
            for k in dim_keys:
                vals = [c[k] for c in per_conversation if k in c]
                if vals:
                    dimension_scores[k] = sum(vals) / len(vals)

        if dimension_scores:
            dim_avg = [
                dimension_scores.get(k)
                for k in ["storyline_consistency", "anthropomorphism",
                          "character_fidelity", "storyline_quality"]
                if k in dimension_scores
            ]
            if dim_avg:
                dimension_scores["average"] = sum(dim_avg) / len(dim_avg)

        print("CoSER scores:")
        for k, v in dimension_scores.items():
            print(f"  {k}: {v:.2f}")

        result = {
            "benchmark": "coser",
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "n_conversations": n_conversations,
                "judge_model": judge_model,
                "nsp_model": nsp_model,
                "env_model": env_model,
                "num_workers": num_workers,
                "wo_thought": True,
            },
            "metrics": dimension_scores,
            "per_conversation": per_conversation,
        }

        result_path = os.path.join(output, "coser_results.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {result_path}")
        return result

    finally:
        if vllm_proc is not None:
            print("Shutting down vLLM server ...")
            vllm_proc.terminate()
            vllm_proc.wait()
