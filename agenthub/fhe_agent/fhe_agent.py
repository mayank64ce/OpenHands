"""
FHEAgent for OpenHands.

Extends CodeActAgent to generate eval() function body code for FHE challenges.
Reuses CodeActAgent's message building, prompt caching, and response parsing.
The runner handles code injection, Docker execution, and feedback.
"""

import os
import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.events.action import Action

from agenthub.fhe_agent.action_parser import FHEResponseParser
from openhands.controller.agent import Agent
from openhands.core.config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
    MessageAction,
)
from openhands.events.observation.observation import Observation
from openhands.llm.llm import LLM
from openhands.runtime.plugins import PluginRequirement
from openhands.utils.prompt import PromptManager

logger = logging.getLogger("openhands.fhe")


class FHEAgent(Agent):
    """Agent that generates eval() function body for FHE challenges.

    Extends the base Agent class, reusing CodeActAgent patterns for
    message building, prompt caching, and response parsing via
    <submit_code> tags.

    The agent's step() returns MessageAction with:
    - content: the extracted code (from <submit_code> tags)
    - thought: the agent's reasoning (everything outside the tags)
    """

    sandbox_plugins: list[PluginRequirement] = []

    action_parser = FHEResponseParser()

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
        challenge_spec=None,
        template_code: dict[str, str] | None = None,
    ):
        super().__init__(llm, config)
        self.reset()
        self.challenge_spec = challenge_spec
        self.template_code = template_code or {}

        # Build challenge context for the system prompt template
        challenge_context = self._build_challenge_context()

        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__)),
            agent_skills_docs=challenge_context,
        )

    def _build_challenge_context(self) -> str:
        """Build challenge-specific context injected into the system prompt template."""
        spec = self.challenge_spec
        lines = []

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
        if self.template_code:
            lines.append("=== TEMPLATE FILES ===")
            for name, content in self.template_code.items():
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

        return "\n".join(lines)

    def _extract_variable_hints(self) -> list[str]:
        """Extract variable names from template header files."""
        lines = []

        header = self.template_code.get("yourSolution.h", "")
        if not header:
            for name, content in self.template_code.items():
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

        py_template = self.template_code.get("app.py", "")
        if py_template:
            solve_match = re.search(r'def\s+solve\s*\(([^)]*)\)', py_template)
            if solve_match:
                lines.append(f"solve() function signature: def solve({solve_match.group(1)})")
                lines.append("Return the encrypted result from solve().")

        return lines

    def action_to_str(self, action: Action) -> str:
        """Convert an action to string for conversation history."""
        if isinstance(action, MessageAction):
            if action.thought:
                return f'{action.thought}\n<submit_code>\n{action.content}\n</submit_code>'
            return action.content
        elif isinstance(action, AgentFinishAction) and action.source == 'agent':
            return action.thought
        return ''

    def get_action_message(self, action: Action) -> Message | None:
        """Convert an action to a Message for LLM context."""
        if isinstance(action, (MessageAction, AgentFinishAction)):
            content = [TextContent(text=self.action_to_str(action))]
            return Message(
                role='user' if action.source == 'user' else 'assistant',
                content=content,
            )
        return None

    def get_observation_message(self, obs: Observation) -> Message | None:
        """Convert an observation to a Message for LLM context.

        FHE feedback comes as MessageAction from USER source, so observations
        are not typically used. Return None for unhandled observation types.
        """
        return None

    def step(self, state: 'State') -> 'Action':
        """Generate next action based on current state.

        Builds conversation from state history, calls LLM, and parses response
        using FHEResponseParser to separate thought from code.
        """
        # Check for exit
        latest_user_message = state.history.get_last_user_message()
        if latest_user_message and latest_user_message.strip() == '/exit':
            return AgentFinishAction()

        messages = self._get_messages(state)

        params: dict = {
            'messages': [message.model_dump() for message in messages],
        }
        if self.llm.supports_temperature:
            params['temperature'] = 0.0
        if self.llm.supports_stop_sequences:
            params['stop'] = ['</submit_code>']

        if self.llm.supports_prompt_caching:
            params['extra_headers'] = {
                'anthropic-beta': 'prompt-caching-2024-07-31',
            }

        try:
            response = self.llm.completion(**params)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return AgentFinishAction(
                outputs={"error": str(e)},
                thought="LLM call failed",
            )

        return self.action_parser.parse(response)

    def _get_messages(self, state: 'State') -> list[Message]:
        """Build messages list for LLM completion.

        Follows CodeActAgent pattern: system prompt, initial user message,
        then event history, with prompt caching and environment reminders.
        """
        messages: list[Message] = [
            Message(
                role='system',
                content=[
                    TextContent(
                        text=self.prompt_manager.system_message,
                        cache_prompt=self.llm.supports_prompt_caching,
                    )
                ],
            ),
            Message(
                role='user',
                content=[
                    TextContent(
                        text=self.prompt_manager.initial_user_message,
                        cache_prompt=self.llm.supports_prompt_caching,
                    )
                ],
            ),
        ]

        for event in state.history.get_events():
            if isinstance(event, Action):
                message = self.get_action_message(event)
            elif isinstance(event, Observation):
                message = self.get_observation_message(event)
            else:
                continue

            if message:
                # Avoid consecutive messages with the same role
                if messages and messages[-1].role == message.role:
                    messages[-1].content.extend(message.content)
                else:
                    messages.append(message)

        # Add prompt caching to last 2 user messages
        if self.llm.supports_prompt_caching:
            user_turns_processed = 0
            for message in reversed(messages):
                if message.role == 'user' and user_turns_processed < 2:
                    message.content[-1].cache_prompt = True
                    user_turns_processed += 1

        # Add environment reminder to latest user message
        latest_user_message = next(
            (
                m
                for m in reversed(messages)
                if m.role == 'user'
                and any(isinstance(c, TextContent) for c in m.content)
            ),
            None,
        )
        if latest_user_message:
            reminder_text = f'\n\nENVIRONMENT REMINDER: You have {state.max_iterations - state.iteration} turns left to complete the task. When finished reply with <finish></finish>.'
            latest_user_message.content.append(TextContent(text=reminder_text))

        return messages


def extract_code_from_response(response: str) -> str | None:
    """Extract code from agent's response.

    Handles both new format (<submit_code> tags, where content is already
    the extracted code) and legacy format (```cpp code blocks).

    Also handles ### CONFIG ### / ### CODE ### sections.
    """
    # Check for multi-section format (CONFIG + CODE)
    if "### CONFIG ###" in response or "### TRAINING CODE ###" in response:
        for marker in ["### CONFIG ###", "### TRAINING CODE ###", "### INFERENCE CODE ###", "### CODE ###"]:
            idx = response.find(marker)
            if idx >= 0:
                return response[idx:].strip()

    # Try specific language code blocks (legacy format)
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

    # If no code blocks found, the content itself may be code
    # (from submit_code parser which already extracted the code)
    lines = response.strip().split('\n')
    code_indicators = sum(1 for l in lines if any(c in l for c in [';', '{', '}', 'auto ', 'int ', 'double ']))
    if code_indicators > len(lines) * 0.3:
        return response.strip()

    return None
