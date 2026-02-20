import re

from openhands.controller.action_parser import ActionParser, ResponseParser
from openhands.events.action import (
    Action,
    AgentFinishAction,
    MessageAction,
)


class FHEResponseParser(ResponseParser):
    """Parser for FHE agent responses.

    Parses:
    - <submit_code>...</submit_code> -> MessageAction(content=code, thought=thought)
    - <finish>...</finish> -> AgentFinishAction
    - Fallback: raw message -> MessageAction
    """

    def __init__(self):
        super().__init__()
        self.action_parsers = [
            FHEActionParserFinish(),
            FHEActionParserSubmitCode(),
        ]
        self.default_parser = FHEActionParserMessage()

    def parse(self, response) -> Action:
        action_str = self.parse_response(response)
        return self.parse_action(action_str)

    def parse_response(self, response) -> str:
        action = response.choices[0].message.content
        if action is None:
            return ''
        # Auto-close unclosed submit_code tag (LLM stopped at stop token)
        if '<submit_code>' in action and '</submit_code>' not in action:
            action += '</submit_code>'
        # Truncate after </submit_code> — simulates stop sequence for models
        # that don't support the stop parameter (e.g. gpt-5-mini)
        end_tag = '</submit_code>'
        if end_tag in action:
            action = action[:action.index(end_tag) + len(end_tag)]
        return action

    def parse_action(self, action_str: str) -> Action:
        for action_parser in self.action_parsers:
            if action_parser.check_condition(action_str):
                return action_parser.parse(action_str)
        return self.default_parser.parse(action_str)


class FHEActionParserFinish(ActionParser):
    """Parses <finish>...</finish> into AgentFinishAction."""

    def __init__(self):
        self.finish_command = None

    def check_condition(self, action_str: str) -> bool:
        self.finish_command = re.search(r'<finish>.*</finish>', action_str, re.DOTALL)
        return self.finish_command is not None

    def parse(self, action_str: str) -> Action:
        assert self.finish_command is not None
        thought = action_str.replace(self.finish_command.group(0), '').strip()
        return AgentFinishAction(thought=thought)


class FHEActionParserSubmitCode(ActionParser):
    """Parses <submit_code>...</submit_code> into MessageAction with thought.

    Everything outside the tag becomes the thought.
    Everything inside the tag becomes the content (code).
    """

    def __init__(self):
        self.submit_code = None

    def check_condition(self, action_str: str) -> bool:
        self.submit_code = re.search(
            r'<submit_code>(.*?)</submit_code>', action_str, re.DOTALL
        )
        return self.submit_code is not None

    def parse(self, action_str: str) -> Action:
        assert self.submit_code is not None
        thought = action_str.replace(self.submit_code.group(0), '').strip()
        code = self.submit_code.group(1).strip()
        return MessageAction(content=code, thought=thought)


class FHEActionParserMessage(ActionParser):
    """Fallback parser: treats entire response as a message."""

    def __init__(self):
        pass

    def check_condition(self, action_str: str) -> bool:
        return True

    def parse(self, action_str: str) -> Action:
        return MessageAction(content=action_str, wait_for_response=True)
