#!/usr/bin/env python3
"""
FHE Challenge Runner for OpenHands.

Runs the FHE challenge solving loop using OpenHands infrastructure:
1. Parse challenge specification
2. Create interpreter (handles Docker judge)
3. Configure OpenHands LLM + FHEAgent
4. Loop: agent.step(state) generates code → inject → docker build/run → feedback → repeat

Uses OpenHands' LLM class (retry logic, metrics, cost tracking) and FHEAgent
(encapsulates prompt building, conversation history via EventStream).

Usage:
    conda run -n openhands python run_fhe.py \
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

import coolname

# Add OpenHands to path
sys.path.insert(0, str(Path(__file__).parent))

from openhands.core.config import (
    AppConfig,
    LLMConfig,
    AgentConfig,
    load_from_toml,
    load_from_env,
    finalize_config,
)
from openhands.llm.llm import LLM
from openhands.events.stream import EventStream
from openhands.events.event import EventSource
from openhands.events.action import MessageAction, AgentFinishAction
from openhands.controller.state.state import State
from openhands.storage.local import LocalFileStore

from agenthub.fhe_agent.fhe_agent import FHEAgent, extract_code_from_response
from fhe.challenge_parser import parse_challenge, FHEChallengeSpec, ChallengeType
from fhe.interpreters import create_interpreter

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
# Run naming
# ============================================================

def _get_next_logindex(dir_path: Path) -> int:
    """Get the next consecutive index from existing log directories.

    Scans for directories matching the pattern '{index}-{name}' and returns
    the next available index.
    """
    if not dir_path.exists():
        return 0
    max_index = -1
    for child in dir_path.iterdir():
        if child.is_dir():
            match = re.match(r'^(\d+)-', child.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def generate_run_name(log_parent: Path) -> str:
    """Generate a unique run name like '0-ancient-bitter-olive'.

    Uses coolname for a memorable random slug, prefixed with a consecutive
    index to preserve ordering (same approach as AIDE-FHE).
    """
    idx = _get_next_logindex(log_parent)
    slug = coolname.generate_slug(3)
    return f"{idx}-{slug}"


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

    if args.log_dir:
        log_parent = Path(args.log_dir)
    else:
        log_parent = Path("logs/fhe") / challenge_name

    log_parent.mkdir(parents=True, exist_ok=True)
    run_name = generate_run_name(log_parent)
    log_dir = log_parent / run_name

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

    # 4. Configure OpenHands
    app_config = AppConfig()
    config_path = Path(__file__).parent / "config.toml"
    if config_path.exists():
        load_from_toml(app_config, str(config_path))
    load_from_env(app_config, os.environ)
    finalize_config(app_config)

    # Override LLM config with CLI args
    llm_config = app_config.get_llm_config()
    llm_config.model = args.model
    if args.api_key:
        llm_config.api_key = args.api_key
    elif not llm_config.api_key:
        # Fall back to common env vars
        llm_config.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if args.base_url:
        llm_config.base_url = args.base_url
    llm_config.temperature = args.temperature

    # 5. Create LLM + Agent
    llm = LLM(config=llm_config)
    agent_config = AgentConfig()
    template_code = read_templates(spec)
    agent = FHEAgent(
        llm=llm,
        config=agent_config,
        challenge_spec=spec,
        template_code=template_code,
    )

    # 6. Create EventStream + State
    file_store = LocalFileStore(str(log_dir / "event_store"))
    event_stream = EventStream(sid="fhe_run", file_store=file_store)

    state = State(max_iterations=args.max_steps)
    state.history.set_event_stream(event_stream)
    state.start_id = 0
    state.history.start_id = 0

    # 7. Add initial user message to event stream
    # The system prompt and initial user prompt come from templates via PromptManager.
    # This event stream message is the actual task trigger.
    task_desc = spec.task or "unknown"
    scheme = spec.scheme.value if spec.scheme else "CKKS"
    depth = spec.constraints.depth if spec.constraints else "N/A"
    initial_msg = MessageAction(
        content=(
            f"Implement the eval() function body for the '{task_desc}' FHE challenge "
            f"using the {scheme} encryption scheme.\n"
            f"Multiplicative depth budget: {depth}\n"
            f"Wrap your code in <submit_code>...</submit_code> tags."
        )
    )
    event_stream.add_event(initial_msg, EventSource.USER)

    # 8. Main loop
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
        state.iteration = step
        logger.info(f"=== Step {step}/{args.max_steps} ===")

        # Agent generates code via LLM
        try:
            action = agent.step(state)
        except Exception as e:
            logger.error(f"Agent step failed: {e}")
            trajectory.append({"step": step, "error": str(e), "timestamp": time.time()})
            error_msg = MessageAction(content=f"[Error: {e}] The previous call failed. Please try again.")
            event_stream.add_event(error_msg, EventSource.USER)
            continue

        if isinstance(action, AgentFinishAction):
            logger.info(f"Agent finished: {action.thought}")
            trajectory.append({"step": step, "finished": True, "thought": action.thought, "timestamp": time.time()})
            break

        # Log agent's thought if present
        if isinstance(action, MessageAction) and action.thought:
            logger.info(f"THOUGHT: {action.thought[:500]}")

        # Add agent response to event stream
        event_stream.add_event(action, EventSource.AGENT)

        if not isinstance(action, MessageAction) or not action.content.strip():
            logger.warning("Empty or non-message action from agent")
            feedback = MessageAction(content="No response received. Please provide a code implementation wrapped in <submit_code> tags.")
            event_stream.add_event(feedback, EventSource.USER)
            continue

        # Extract code from MessageAction
        # With the new parser, action.content already contains the code from <submit_code> tags.
        # extract_code_from_response handles both raw code and legacy ```cpp blocks.
        code = extract_code_from_response(action.content)
        if not code:
            logger.warning("No code found in agent response")
            feedback = MessageAction(content="ERROR: No code block found. Please wrap your code in <submit_code>...</submit_code> tags.")
            event_stream.add_event(feedback, EventSource.USER)
            trajectory.append({"step": step, "error": "no_code_extracted", "timestamp": time.time()})
            continue

        # Execute via interpreter
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

        # Feed back result as user message via EventStream
        feedback_text = result.get_feedback()
        remaining = args.max_steps - step - 1
        feedback_msg = MessageAction(
            content=f"Execution result:\n{feedback_text}\n\n[Remaining attempts: {remaining}] Please provide an improved implementation."
        )
        event_stream.add_event(feedback_msg, EventSource.USER)

        # Log
        thought = action.thought if isinstance(action, MessageAction) else ""
        step_info = {
            "step": step,
            "timestamp": time.time(),
            "thought": thought[:500] if thought else "",
            "code": code[:500] + "..." if len(code) > 500 else code,
            "build_success": result.build_success,
            "run_success": result.run_success,
            "accuracy": result.accuracy,
            "feedback": feedback_text[:200],
            "exec_time": time.time() - step_start,
            "is_best": is_best,
        }
        trajectory.append(step_info)

        save_step(log_dir, step, code, result, feedback_text)
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

    # 9. Final output
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

    # Save LLM metrics
    try:
        llm_metrics = {
            "model": args.model,
            "accumulated_cost": llm.metrics.accumulated_cost if hasattr(llm.metrics, 'accumulated_cost') else None,
            "total_tokens": getattr(llm.metrics, 'total_tokens', None),
        }
        (log_dir / "llm_metrics.json").write_text(json.dumps(llm_metrics, indent=2, default=str))
    except Exception:
        pass

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
