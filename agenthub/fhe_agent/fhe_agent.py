"""
FHEAgent for OpenHands.

Generates eval() function body code for FHE challenges.
Uses OpenHands' LLM class (litellm) for model calls.
The runner handles code injection, Docker execution, and feedback.
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.events.action import Action

from openhands.controller.agent import Agent
from openhands.events.action import MessageAction, AgentFinishAction
from openhands.llm.llm import LLM
from openhands.runtime.plugins import PluginRequirement

logger = logging.getLogger("openhands.fhe")


class FHEAgent(Agent):
    """Agent that generates eval() function body for FHE challenges.

    This agent does NOT interact with a runtime/sandbox.
    Instead, the runner extracts code from its responses, injects it
    into templates, runs Docker judges, and feeds back results.

    The agent's step() returns MessageAction containing C++ (or Python)
    code in a ```cpp code block. The runner extracts this code.
    """

    sandbox_plugins: list[PluginRequirement] = []

    def __init__(
        self,
        llm: LLM,
        config: 'AgentConfig',
        challenge_spec=None,
        template_code: dict[str, str] | None = None,
    ):
        super().__init__(llm, config)
        self.challenge_spec = challenge_spec
        self.template_code = template_code or {}
        self._system_prompt = None

    def _get_system_prompt(self) -> str:
        """Build the system prompt with challenge context."""
        if self._system_prompt is not None:
            return self._system_prompt

        spec = self.challenge_spec
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

            # Add useful links
            if spec.useful_links:
                lines.append("Useful Resources:")
                for link in spec.useful_links:
                    lines.append(f"  - {link['name']}: {link['url']}")
                lines.append("")

        # Add template code context
        if self.template_code:
            lines.append("=== TEMPLATE FILES ===")
            for name, content in self.template_code.items():
                # Truncate large files
                if len(content) > 3000:
                    content = content[:2500] + "\n\n// ... (truncated) ...\n\n" + content[-500:]
                lines.append(f"\n--- {name} ---")
                lines.append(content)
            lines.append("")

        # Add variable extraction hints
        if self.template_code:
            lines.append("=== VARIABLE NAMES ===")
            lines.extend(self._extract_variable_hints())
            lines.append("")

        # ML inference specific
        if spec and spec.challenge_type.value == "ml_inference":
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

        # White box config format
        if spec and spec.challenge_type.value == "white_box_openfhe":
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

        self._system_prompt = "\n".join(lines)
        return self._system_prompt

    def _extract_variable_hints(self) -> list[str]:
        """Extract variable names from template header files."""
        lines = []

        # Check for C++ header
        header = self.template_code.get("yourSolution.h", "")
        if not header:
            for name, content in self.template_code.items():
                if name.endswith(".h"):
                    header = content
                    break

        if header:
            # Extract Ciphertext member variables
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

        # Check for Python template
        py_template = self.template_code.get("app.py", "")
        if py_template:
            solve_match = re.search(r'def\s+solve\s*\(([^)]*)\)', py_template)
            if solve_match:
                lines.append(f"solve() function signature: def solve({solve_match.group(1)})")
                lines.append("Return the encrypted result from solve().")

        return lines

    def step(self, state: 'State') -> 'Action':
        """Generate next action based on current state.

        Builds conversation from state history and calls LLM.
        Returns MessageAction with code in ```cpp block.
        """
        messages = self._build_messages(state)

        try:
            response = self.llm.completion(messages=messages)
            content = response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return AgentFinishAction(
                outputs={"error": str(e)},
                thought="LLM call failed",
            )

        if not content or not content.strip():
            return AgentFinishAction(
                outputs={"error": "Empty response from LLM"},
                thought="LLM returned empty response",
            )

        return MessageAction(content=content)

    def _build_messages(self, state: 'State') -> list[dict]:
        """Build messages list for LLM completion.

        Structure:
        - System message with challenge context
        - History of previous attempts and feedback
        - Current request
        """
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
        ]

        # Build conversation from state history
        has_history = False
        if state and hasattr(state, 'history'):
            for event in state.history.get_events():
                from openhands.events.action import MessageAction as MA
                from openhands.events.observation import (
                    CmdOutputObservation,
                    ErrorObservation,
                )

                if isinstance(event, MA):
                    # Agent's previous response
                    if hasattr(event, '_source') and event._source == 'agent':
                        messages.append({"role": "assistant", "content": event.content})
                    else:
                        # User/system feedback
                        messages.append({"role": "user", "content": event.content})
                    has_history = True

        # If no history, add initial request
        if not has_history:
            messages.append({
                "role": "user",
                "content": self._get_initial_prompt(),
            })

        # Add remaining turns reminder
        remaining = state.max_iterations - state.iteration if state else "unknown"
        if has_history:
            messages.append({
                "role": "user",
                "content": f"[Remaining attempts: {remaining}] Please provide an improved implementation.",
            })

        return messages

    def _get_initial_prompt(self) -> str:
        """Get the initial prompt for the first step."""
        spec = self.challenge_spec
        if not spec:
            return "Please implement the eval() function body. Wrap your code in ```cpp block."

        task = spec.task or "unknown"
        scheme = spec.scheme.value if spec.scheme else "CKKS"

        return (
            f"Implement the eval() function body for the '{task}' FHE challenge "
            f"using the {scheme} encryption scheme.\n\n"
            f"Remember:\n"
            f"- Output ONLY the function body (not the full function definition)\n"
            f"- Use the exact member variable names from the template\n"
            f"- Stay within the multiplicative depth budget of {spec.constraints.depth}\n"
            f"- Wrap your code in a ```cpp code block\n"
        )


def extract_code_from_response(response: str) -> str | None:
    """Extract code from agent's response.

    Looks for code blocks in order of preference:
    1. ```cpp ... ```
    2. ```c++ ... ```
    3. ```python ... ``` (for ML inference)
    4. ``` ... ``` (generic code block)
    5. Raw text if it looks like code

    Also handles ### CONFIG ### / ### CODE ### sections.
    """
    # Check for multi-section format (CONFIG + CODE)
    if "### CONFIG ###" in response or "### TRAINING CODE ###" in response:
        # Return everything from the first ### marker onwards
        for marker in ["### CONFIG ###", "### TRAINING CODE ###", "### INFERENCE CODE ###", "### CODE ###"]:
            idx = response.find(marker)
            if idx >= 0:
                return response[idx:].strip()

    # Try specific language code blocks
    for lang in ['cpp', 'c\\+\\+', 'python', 'swift']:
        pattern = rf'```{lang}\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try generic code block
    pattern = r'```\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Heuristic: if response looks like code (has semicolons, braces)
    lines = response.strip().split('\n')
    code_indicators = sum(1 for l in lines if any(c in l for c in [';', '{', '}', 'auto ', 'int ', 'double ']))
    if code_indicators > len(lines) * 0.3:
        return response.strip()

    return None
