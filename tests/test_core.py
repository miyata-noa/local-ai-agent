"""Agentクラスのテスト。"""
import pytest

from agent.core import Agent, AgentConfig, AgentState
from agent.llm.base import Message
from agent.llm.mock import MockLLMClient


class TestAgentConfig:
    def test_defaults(self) -> None:
        config = AgentConfig()
        assert "assistant" in config.system_prompt.lower()
        assert config.max_history == 10

    def test_custom_values(self) -> None:
        config = AgentConfig(system_prompt="Be brief.", max_history=5)
        assert config.system_prompt == "Be brief."
        assert config.max_history == 5


class TestAgentState:
    def test_add_message(self) -> None:
        state = AgentState()
        state.add("user", "hello")
        assert len(state.history) == 1
        assert state.history[0].role == "user"

    def test_clear(self) -> None:
        state = AgentState()
        state.add("user", "hello")
        state.clear()
        assert state.history == []


class TestAgent:
    def _make_agent(self, response: str = "ok") -> Agent:
        return Agent(llm=MockLLMClient(response=response))

    def test_run_returns_response(self) -> None:
        agent = self._make_agent("hello back")
        result = agent.run("hello")
        assert result == "hello back"

    def test_run_updates_history(self) -> None:
        agent = self._make_agent()
        agent.run("first")
        agent.run("second")
        history = agent.history
        assert len(history) == 4  # user, assistant, user, assistant
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_run_empty_input_raises(self) -> None:
        agent = self._make_agent()
        with pytest.raises(ValueError, match="must not be empty"):
            agent.run("")

    def test_run_whitespace_only_raises(self) -> None:
        agent = self._make_agent()
        with pytest.raises(ValueError):
            agent.run("   ")

    def test_reset_clears_history(self) -> None:
        agent = self._make_agent()
        agent.run("hello")
        agent.reset()
        assert agent.history == []

    def test_history_is_copy(self) -> None:
        """historyプロパティが内部状態への参照を返さないことを確認。"""
        agent = self._make_agent()
        agent.run("hello")
        h = agent.history
        h.clear()
        assert len(agent.history) == 2  # 内部状態は変わらない

    def test_system_prompt_included_in_llm_call(self) -> None:
        llm = MockLLMClient()
        config = AgentConfig(system_prompt="Custom prompt.")
        agent = Agent(llm=llm, config=config)
        agent.run("hi")
        first_msg = llm.last_messages[0]
        assert first_msg.role == "system"
        assert first_msg.content == "Custom prompt."

    def test_history_trimmed_when_exceeds_max(self) -> None:
        llm = MockLLMClient()
        config = AgentConfig(max_history=2)
        agent = Agent(llm=llm, config=config)
        for i in range(5):
            agent.run(f"message {i}")
        # max_history=2 → 最大4メッセージ（2往復）
        assert len(agent.history) <= 4
