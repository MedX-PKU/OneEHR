from oneehr.agent.client import AgentClientError, OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec, AgentResponse, ParsedPrediction
from oneehr.agent.schema import build_response_format, parse_prediction_response, schema_prompt_text
from oneehr.agent.templates import (
    PromptTemplateSpec,
    describe_prompt_template,
    get_prompt_template,
    list_prompt_templates,
    safe_case_slug,
    select_events,
)

__all__ = [
    "AgentClientError",
    "AgentRequestSpec",
    "AgentResponse",
    "OpenAICompatibleAgentClient",
    "ParsedPrediction",
    "PromptTemplateSpec",
    "build_response_format",
    "describe_prompt_template",
    "get_prompt_template",
    "list_prompt_templates",
    "parse_prediction_response",
    "safe_case_slug",
    "schema_prompt_text",
    "select_events",
]
