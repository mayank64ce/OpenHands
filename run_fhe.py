#!/usr/bin/env python3
"""
FHE Challenge Runner for OpenHands.

Runs the FHE challenge solving loop:
1. Parse challenge specification
2. Create interpreter (handles Docker judge)
3. Build system prompt with challenge context
4. Loop: LLM generates code → inject → docker build/run → feedback → repeat

Uses litellm directly for LLM calls (compatible with Python 3.10+).
OpenHands Agent/Runtime infrastructure is available but not required for this runner.

Usage:
    python run_fhe.py \
        --model gpt-4o \
        --challenge-dir ../fhe_challenge/black_box/challenge_relu \
        --max-steps 30 \
        --log-dir logs/fhe/relu_test \
        --build-timeout 600 \
        --run-timeout 600
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from litellm import completion as litellm_completion

# Add OpenHands to path
sys.path.insert(0, str(Path(__file__).parent))

from fhe.challenge_parser import parse_challenge, FHEChallengeSpec, ChallengeType
from fhe.interpreters import create_interpreter, ExecutionResult
from openhands.core.config import AppConfig, load_from_toml, load_from_env, finalize_config

logger = logging.getLogger("openhands.fhe.runner")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FHE Challenge Runner for OpenHands")

    parser.add_argument("--challenge-dir", type=str, required=True,
                        help="Path to challenge directory")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model name (litellm format)")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Maximum number of solve attempts")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for logs and outputs")
    parser.add_argument("--build-timeout", type=int, default=600,
                        help="Docker build timeout in seconds")
    parser.add_argument("--run-timeout", type=int, default=600,
                        help="Docker run timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM temperature")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (or set via env var)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for API (for local LLMs)")
    parser.add_argument("--early-stop", type=float, default=0.99,
                        help="Stop when accuracy reaches this threshold")

    return parser.parse_args()


# ============================================================
# Code extraction
# ============================================================

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract code from LLM response.

    Handles:
    1. ### CONFIG ### / ### CODE ### sections
    2. ```cpp ... ``` code blocks
    3. Generic ``` ... ``` blocks
    4. Raw code heuristic
    """
    if "### CONFIG ###" in response or "### TRAINING CODE ###" in response:
        for marker in ["### CONFIG ###", "### TRAINING CODE ###", "### INFERENCE CODE ###", "### CODE ###"]:
            idx = response.find(marker)
            if idx >= 0:
                return response[idx:].strip()

    for lang in ['cpp', 'c\\+\\+', 'python', 'swift']:
        pattern = rf'```{lang}\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    pattern = r'```\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = response.strip().split('\n')
    code_indicators = sum(1 for l in lines if any(c in l for c in [';', '{', '}', 'auto ', 'int ', 'double ']))
    if code_indicators > len(lines) * 0.3:
        return response.strip()

    return None


# ============================================================
# System prompt builder
# ============================================================

def build_system_prompt(spec: FHEChallengeSpec, template_code: dict[str, str]) -> str:
    """Build system prompt with challenge context and template info."""
    lines = [
        "You are an expert FHE (Fully Homomorphic Encryption) programmer.",
        "Your task is to implement the eval() function body for an FHE challenge.",
        "",
        "IMPORTANT RULES:",
        "1. Output ONLY the eval() function BODY (not the full function definition).",
        "2. Wrap your code in a ```cpp code block.",
        "3. Use the exact variable names from the template (class member variables).",
        "4. The eval() function is void - assign the result to the output variable, do NOT return.",
        "5. Stay within the multiplicative depth budget.",
        "6. Use only the rotation keys that are available.",
        "",
    ]

    if spec:
        lines.append("=== CHALLENGE SPECIFICATION ===")
        lines.append(f"Task: {spec.task}")
        if spec.task_description:
            lines.append(f"Description: {spec.task_description}")
        if spec.function_signature:
            lines.append(f"Function: {spec.function_signature}")
        lines.append(f"Scheme: {spec.scheme.value if spec.scheme else 'CKKS'}")
        lines.append(f"Library: {spec.library.value if spec.library else 'OpenFHE'}")
        lines.append(f"Challenge Type: {spec.challenge_type.value}")
        lines.append("")

        if spec.constraints:
            lines.append("Constraints:")
            lines.append(f"  Multiplicative Depth: {spec.constraints.depth}")
            lines.append(f"  Batch Size: {spec.constraints.batch_size}")
            lines.append(f"  Input Range: [{spec.constraints.input_range[0]}, {spec.constraints.input_range[1]}]")
            lines.append("")

        if spec.keys:
            lines.append("Available Keys:")
            lines.append(f"  Public Key: {spec.keys.public}")
            lines.append(f"  Multiplication Key: {spec.keys.multiplication}")
            if spec.keys.rotation_indices:
                lines.append(f"  Rotation Indices: {spec.keys.rotation_indices}")
            lines.append(f"  Bootstrapping: {spec.keys.bootstrapping}")
            lines.append("")

        if spec.scoring:
            lines.append("Scoring:")
            lines.append(f"  Error Threshold: {spec.scoring.error_threshold}")
            lines.append(f"  Accuracy Threshold: {spec.scoring.accuracy_threshold}")
            lines.append("")

        if spec.useful_links:
            lines.append("Useful Resources:")
            for link in spec.useful_links:
                lines.append(f"  - {link['name']}: {link['url']}")
            lines.append("")

    # Add template code context
    if template_code:
        lines.append("=== TEMPLATE FILES ===")
        for name, content in template_code.items():
            if len(content) > 3000:
                content = content[:2500] + "\n\n// ... (truncated) ...\n\n" + content[-500:]
            lines.append(f"\n--- {name} ---")
            lines.append(content)
        lines.append("")

        # Extract variable hints from header
        lines.append("=== VARIABLE NAMES ===")
        lines.extend(_extract_variable_hints(template_code))
        lines.append("")

    # Format-specific instructions
    if spec and spec.challenge_type == ChallengeType.ML_INFERENCE:
        lines.extend([
            "=== ML INFERENCE FORMAT ===",
            "For ML inference challenges, provide your code in this format:",
            "",
            "### TRAINING CODE ###",
            "# Python training script (saves weights to workspace/weights/)",
            "",
            "### INFERENCE CODE ###",
            "# C++ eval() body that loads weights and does encrypted inference",
            "",
        ])

    if spec and spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
        lines.extend([
            "=== CONFIG FORMAT ===",
            "For white box challenges, you may optionally provide a config.json:",
            "",
            "### CONFIG ###",
            '{"mult_depth": 10, "scale_mod_size": 50, ...}',
            "",
            "### CODE ###",
            "// Your eval() body here",
            "",
            "If no CONFIG section is provided, the template's default config.json is used.",
            "",
        ])

    return "\n".join(lines)


def _extract_variable_hints(template_code: dict[str, str]) -> list[str]:
    """Extract variable names from template header files."""
    lines = []

    header = template_code.get("yourSolution.h", "")
    if not header:
        for name, content in template_code.items():
            if name.endswith(".h"):
                header = content
                break

    if header:
        ct_vars = re.findall(r'Ciphertext<DCRTPoly>\s+(m_\w+)', header)
        cc_match = re.search(r'CryptoContext<DCRTPoly>\s+(m_\w+)', header)
        pk_match = re.search(r'PublicKey<DCRTPoly>\s+(m_\w+)', header)

        inputs = [v for v in ct_vars if 'Output' not in v]
        outputs = [v for v in ct_vars if 'Output' in v]

        lines.append("Use these EXACT member variable names:")
        if cc_match:
            lines.append(f"  CryptoContext: {cc_match.group(1)}")
        if inputs:
            lines.append(f"  Input(s): {', '.join(inputs)}")
        if outputs:
            lines.append(f"  Output: {', '.join(outputs)} (assign to this, don't return)")
        if pk_match:
            lines.append(f"  PublicKey: {pk_match.group(1)}")

    py_template = template_code.get("app.py", "")
    if py_template:
        solve_match = re.search(r'def\s+solve\s*\(([^)]*)\)', py_template)
        if solve_match:
            lines.append(f"solve() function signature: def solve({solve_match.group(1)})")
            lines.append("Return the encrypted result from solve().")

    return lines


def get_initial_prompt(spec: FHEChallengeSpec) -> str:
    """Get the initial user prompt."""
    task = spec.task or "unknown"
    scheme = spec.scheme.value if spec.scheme else "CKKS"
    depth = spec.constraints.depth if spec.constraints else 10

    return (
        f"Implement the eval() function body for the '{task}' FHE challenge "
        f"using the {scheme} encryption scheme.\n\n"
        f"Remember:\n"
        f"- Output ONLY the function body (not the full function definition)\n"
        f"- Use the exact member variable names from the template\n"
        f"- Stay within the multiplicative depth budget of {depth}\n"
        f"- Wrap your code in a ```cpp code block\n"
    )


# ============================================================
# Template reading
# ============================================================

def read_templates(spec: FHEChallengeSpec) -> dict[str, str]:
    """Read template files for the challenge."""
    templates = {}

    if not spec.template_dir or not spec.template_dir.exists():
        return templates

    file_lists = {
        ChallengeType.BLACK_BOX: ["yourSolution.cpp", "yourSolution.h"],
        ChallengeType.WHITE_BOX_OPENFHE: ["yourSolution.cpp", "yourSolution.h", "config.json"],
        ChallengeType.ML_INFERENCE: ["app.py", "yourSolution.cpp", "config.json"],
        ChallengeType.NON_OPENFHE: ["main.swift", "app.py", "Package.swift", "config.json"],
    }

    files_to_load = file_lists.get(spec.challenge_type, ["yourSolution.cpp", "yourSolution.h"])

    for name in files_to_load:
        path = spec.template_dir / name
        if path.exists():
            templates[name] = path.read_text()

    if spec.challenge_type == ChallengeType.ML_INFERENCE:
        for header in spec.template_dir.glob("*.h"):
            templates[header.name] = header.read_text()

    if spec.challenge_type == ChallengeType.NON_OPENFHE and "main.swift" not in templates:
        swift_main = spec.template_dir / "Sources" / "main.swift"
        if swift_main.exists():
            templates["main.swift"] = swift_main.read_text()

    return templates


# ============================================================
# Result aggregation
# ============================================================

@dataclass
class AggregatedResult:
    build_success: bool
    run_success: bool
    output_generated: bool
    accuracy: Optional[float]
    max_error: Optional[float]
    mean_error: Optional[float]
    run_output: list
    build_output: list
    total_time: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def get_feedback(self) -> str:
        lines = []

        if not self.build_success:
            lines.append("BUILD FAILED")
            lines.append(f"Error: {self.error_type}")
            lines.append(f"Message: {self.error_message}")
            lines.append("")
            lines.append("Build output (last 50 lines):")
            lines.extend(self.build_output[-50:])
            return "\n".join(lines)

        if not self.run_success:
            lines.append("RUNTIME FAILED")
            lines.append(f"Error: {self.error_type}")
            lines.append(f"Message: {self.error_message}")
            lines.append("")
            lines.append("Runtime output (last 50 lines):")
            lines.extend(self.run_output[-50:])
            return "\n".join(lines)

        if not self.output_generated:
            lines.append("NO OUTPUT GENERATED")
            lines.append("The solution ran but did not produce an output file.")
            return "\n".join(lines)

        if self.accuracy is not None:
            lines.append(f"VALIDATION: {'PASSED' if self.accuracy >= 0.8 else 'NEEDS IMPROVEMENT'}")
            lines.append(f"Accuracy: {self.accuracy:.4f}")
            if self.mean_error is not None:
                lines.append(f"Mean error: {self.mean_error:.6f}")
            if self.max_error is not None:
                lines.append(f"Max error: {self.max_error:.6f}")
            return "\n".join(lines)

        lines.append("EXECUTION COMPLETE (no accuracy metric)")
        return "\n".join(lines)


def aggregate_results(results: list) -> AggregatedResult:
    """Aggregate execution results using minimum accuracy (worst case)."""
    build_success = all(getattr(r, 'build_success', False) for r in results)
    run_success = all(getattr(r, 'run_success', False) for r in results)
    output_generated = all(getattr(r, 'output_generated', False) for r in results)

    error_type = None
    error_message = None
    build_output = []
    for r in results:
        if not getattr(r, 'build_success', True):
            error_type = getattr(r, 'error_type', 'BUILD_ERROR')
            error_message = getattr(r, 'error_message', 'Build failed')
            build_output = getattr(r, 'build_output', [])
            break
        if not getattr(r, 'run_success', True):
            error_type = getattr(r, 'error_type', 'RUNTIME_ERROR')
            error_message = getattr(r, 'error_message', 'Runtime error')
            break

    validations = [getattr(r, 'validation', None) for r in results]
    valid_validations = [v for v in validations if v and v.accuracy is not None]

    accuracy = None
    max_error = None
    mean_error = None
    if valid_validations:
        accuracy = min(v.accuracy for v in valid_validations)
        errors_max = [v.max_error for v in valid_validations if v.max_error is not None]
        max_error = max(errors_max) if errors_max else None
        errors_mean = [v.mean_error for v in valid_validations if v.mean_error is not None]
        mean_error = sum(errors_mean) / len(errors_mean) if errors_mean else None

    run_output = []
    for i, r in enumerate(results):
        if hasattr(r, 'run_output'):
            run_output.append(f"=== Testcase {i+1} ===")
            run_output.extend(r.run_output if isinstance(r.run_output, list) else [r.run_output])

    total_time = sum(getattr(r, 'total_time', 0) for r in results)

    return AggregatedResult(
        build_success=build_success, run_success=run_success,
        output_generated=output_generated, accuracy=accuracy,
        max_error=max_error, mean_error=mean_error,
        run_output=run_output, build_output=build_output,
        total_time=total_time, error_type=error_type, error_message=error_message,
    )


def run_on_testcases(interpreter, code: str, testcase_dirs: list) -> AggregatedResult:
    """Run code on all testcases and aggregate results."""
    if not testcase_dirs:
        result = interpreter.execute(code, None)
        return aggregate_results([result])

    results = []
    for testcase_path in testcase_dirs:
        result = interpreter.execute(code, testcase_path)
        results.append(result)

    if len(results) == 1:
        r = results[0]
        return AggregatedResult(
            build_success=r.build_success, run_success=r.run_success,
            output_generated=r.output_generated,
            accuracy=r.accuracy,
            max_error=r.validation.max_error if r.validation else None,
            mean_error=r.validation.mean_error if r.validation else None,
            run_output=r.run_output, build_output=r.build_output,
            total_time=r.total_time, error_type=r.error_type,
            error_message=r.error_message,
        )

    return aggregate_results(results)


# ============================================================
# LLM wrapper
# ============================================================

def llm_call(
    messages: list[dict],
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call LLM via litellm and return response content."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    response = litellm_completion(**kwargs)
    return response['choices'][0]['message']['content']


# ============================================================
# Persistence
# ============================================================

def save_step(log_dir: Path, step: int, code: str, result: AggregatedResult, feedback: str):
    step_dir = log_dir / "steps" / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    (step_dir / "code.cpp").write_text(code)
    (step_dir / "feedback.txt").write_text(feedback)

    result_dict = {
        "build_success": result.build_success,
        "run_success": result.run_success,
        "output_generated": result.output_generated,
        "accuracy": result.accuracy,
        "max_error": result.max_error,
        "mean_error": result.mean_error,
        "total_time": result.total_time,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }
    (step_dir / "result.json").write_text(json.dumps(result_dict, indent=2))


def save_trajectory(log_dir: Path, trajectory: list, metadata: dict):
    data = {"metadata": metadata, "steps": trajectory}
    (log_dir / "trajectory.json").write_text(json.dumps(data, indent=2, default=str))


def save_best_solution(log_dir: Path, best: dict):
    (log_dir / "best_solution.cpp").write_text(best["code"])
    info = {k: v for k, v in best.items() if k != "code"}
    (log_dir / "best_solution_info.json").write_text(json.dumps(info, indent=2, default=str))


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 1. Parse challenge
    logger.info(f"Parsing challenge: {args.challenge_dir}")
    spec = parse_challenge(args.challenge_dir)
    logger.info(f"Challenge: {spec.challenge_name} ({spec.challenge_type.value})")
    logger.info(f"Task: {spec.task}, Scheme: {spec.scheme.value if spec.scheme else 'N/A'}")

    # 2. Setup workspace
    challenge_name = spec.challenge_dir.name if spec.challenge_dir else "unknown"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path("logs/fhe") / f"{challenge_name}_{timestamp}"

    log_dir.mkdir(parents=True, exist_ok=True)
    workspace = log_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (log_dir / "steps").mkdir(parents=True, exist_ok=True)

    logger.info(f"Log directory: {log_dir}")

    # Save config
    config_dict = {
        "model": args.model,
        "challenge_dir": str(args.challenge_dir),
        "max_steps": args.max_steps,
        "build_timeout": args.build_timeout,
        "run_timeout": args.run_timeout,
        "temperature": args.temperature,
        "early_stop": args.early_stop,
    }
    (log_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Save challenge spec
    try:
        spec.save(log_dir / "challenge_spec.json")
    except Exception:
        (log_dir / "challenge_spec.json").write_text(json.dumps({
            "task": spec.task,
            "challenge_type": spec.challenge_type.value,
            "scheme": spec.scheme.value if spec.scheme else None,
        }, indent=2))

    # 3. Create interpreter
    interpreter = create_interpreter(
        spec=spec,
        workspace_dir=workspace,
        build_timeout=args.build_timeout,
        run_timeout=args.run_timeout,
    )
    logger.info(f"Interpreter: {interpreter.__class__.__name__}")

    # 4. Build prompts
    template_code = read_templates(spec)
    system_prompt = build_system_prompt(spec, template_code)
    initial_prompt = get_initial_prompt(spec)

    # 5. LLM config
    app_config = AppConfig()
    config_path = Path(__file__).parent / "config.toml"
    if config_path.exists():
        load_from_toml(app_config, str(config_path))
    load_from_env(app_config, os.environ)
    finalize_config(app_config)
    
    llm_config = app_config.get_llm_config()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or llm_config.api_key
    base_url = args.base_url or llm_config.base_url

    # 6. Run the loop
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    trajectory = []
    best_solution = {"accuracy": 0.0, "code": "", "step": -1}
    start_time = time.time()

    metadata = {
        "model": args.model,
        "challenge": challenge_name,
        "challenge_type": spec.challenge_type.value,
        "max_steps": args.max_steps,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info(f"Starting solve loop (max {args.max_steps} steps)")

    for step in range(args.max_steps):
        step_start = time.time()
        logger.info(f"=== Step {step}/{args.max_steps} ===")

        # Call LLM
        try:
            response = llm_call(
                messages=messages,
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                temperature=args.temperature,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            trajectory.append({"step": step, "error": str(e), "timestamp": time.time()})
            # Add error to conversation so LLM sees it
            messages.append({"role": "assistant", "content": f"[Error: {e}]"})
            messages.append({"role": "user", "content": "The previous call failed. Please try again with a valid implementation."})
            continue

        if not response or not response.strip():
            logger.warning("Empty LLM response")
            continue

        # Extract code
        code = extract_code_from_response(response)
        if not code:
            logger.warning("No code found in LLM response")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "ERROR: No code block found. Please wrap your code in ```cpp ... ```"})
            trajectory.append({"step": step, "error": "no_code_extracted", "timestamp": time.time()})
            continue

        # Add to conversation
        messages.append({"role": "assistant", "content": response})

        # Execute
        logger.info(f"Executing code ({len(code)} chars)...")
        result = run_on_testcases(interpreter, code, spec.testcase_dirs)

        # Track best
        is_best = False
        if result.accuracy is not None and result.accuracy > best_solution["accuracy"]:
            best_solution = {
                "accuracy": result.accuracy,
                "code": code,
                "step": step,
                "timestamp": time.time(),
            }
            is_best = True
            save_best_solution(log_dir, best_solution)
            logger.info(f"New best: accuracy={result.accuracy:.4f} (step {step})")

        # Feedback
        feedback = result.get_feedback()
        remaining = args.max_steps - step - 1
        feedback_msg = f"Execution result:\n{feedback}\n\n[Remaining attempts: {remaining}] Please provide an improved implementation."
        messages.append({"role": "user", "content": feedback_msg})

        # Log
        step_info = {
            "step": step,
            "timestamp": time.time(),
            "code": code[:500] + "..." if len(code) > 500 else code,
            "build_success": result.build_success,
            "run_success": result.run_success,
            "accuracy": result.accuracy,
            "feedback": feedback[:200],
            "exec_time": time.time() - step_start,
            "is_best": is_best,
        }
        trajectory.append(step_info)

        save_step(log_dir, step, code, result, feedback)
        save_trajectory(log_dir, trajectory, metadata)

        acc_str = f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A"
        best_str = f"{best_solution['accuracy']:.4f}"
        logger.info(f"Step {step}: accuracy={acc_str}, best={best_str}, "
                     f"build={'OK' if result.build_success else 'FAIL'}, "
                     f"run={'OK' if result.run_success else 'FAIL'}")

        # Early stop
        if result.accuracy is not None and result.accuracy >= args.early_stop:
            logger.info(f"Early stop: accuracy {result.accuracy:.4f} >= {args.early_stop}")
            break

    # 7. Final output
    end_time = time.time()
    metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "total_steps": len(trajectory),
        "total_time": end_time - start_time,
        "best_accuracy": best_solution["accuracy"],
        "best_step": best_solution["step"],
        "good_solutions": sum(1 for t in trajectory if t.get("accuracy") and t["accuracy"] > 0),
        "failed_solutions": sum(1 for t in trajectory
                                if not t.get("build_success", True) or not t.get("run_success", True)),
    }

    metadata["summary"] = summary
    save_trajectory(log_dir, trajectory, metadata)

    (log_dir / "summary.json").write_text(json.dumps({
        "metadata": metadata,
        "summary": summary,
        "best_solution": {k: v for k, v in best_solution.items() if k != "code"},
    }, indent=2, default=str))

    print("\n" + "=" * 60)
    print("FHE Challenge Run Complete")
    print("=" * 60)
    print(f"\nChallenge: {spec.challenge_name or spec.task}")
    print(f"Type: {spec.challenge_type.value}")
    print(f"Model: {args.model}")
    print(f"\nTotal steps: {summary['total_steps']}")
    print(f"Total time: {summary['total_time']:.1f}s")
    print(f"Best accuracy: {summary['best_accuracy']:.4f} (step {summary['best_step']})")
    print(f"Good solutions: {summary['good_solutions']}")
    print(f"Failed solutions: {summary['failed_solutions']}")
    print(f"\nOutput: {log_dir}")
    print(f"Best solution: {log_dir / 'best_solution.cpp'}")
    print(f"Trajectory: {log_dir / 'trajectory.json'}")

    interpreter.cleanup()


if __name__ == "__main__":
    main()
