"""LLMクライアント（base / mock）のテスト。"""

from agent.llm.base import BaseLLMClient, LLMResponse, Message
from agent.llm.mock import MockLLMClient


class TestMessage:
    def test_message_fields(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"


class TestLLMResponse:
    def test_response_defaults(self) -> None:
        res = LLMResponse(content="hi", model="mock")
        assert res.finish_reason == "stop"

    def test_response_custom_finish_reason(self) -> None:
        res = LLMResponse(content="hi", model="mock", finish_reason="length")
        assert res.finish_reason == "length"


class TestMockLLMClient:
    def test_returns_configured_response(self) -> None:
        client = MockLLMClient(response="test reply")
        result = client.complete([Message(role="user", content="hi")])
        assert result.content == "test reply"
        assert result.model == "mock"

    def test_call_count_increments(self) -> None:
        client = MockLLMClient()
        client.complete([Message(role="user", content="a")])
        client.complete([Message(role="user", content="b")])
        assert client.call_count == 2

    def test_last_messages_recorded(self) -> None:
        client = MockLLMClient()
        msgs = [Message(role="user", content="hello")]
        client.complete(msgs)
        assert client.last_messages == msgs

    def test_chat_interface(self) -> None:
        client = MockLLMClient(response="chat reply")
        result = client.chat("hello")
        assert result == "chat reply"

    def test_chat_with_system_prompt(self) -> None:
        client = MockLLMClient()
        client.chat("hello", system_prompt="You are helpful.")
        messages = client.last_messages
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful."
        assert messages[1].role == "user"

    def test_is_subclass_of_base(self) -> None:
        assert issubclass(MockLLMClient, BaseLLMClient)
