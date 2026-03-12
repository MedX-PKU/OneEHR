from oneehr.agent.client import AgentClientError, OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec, AgentResponse, ParsedPrediction
from oneehr.agent.instances import (
    MaterializedAgentInstances,
    agent_instance_path,
    ensure_agent_predict_splits,
    materialize_agent_instances,
    validate_agent_predict_setup,
)
from oneehr.agent.predict_eval import summarize_prediction_rows
from oneehr.agent.render import render_prompt
from oneehr.agent.review_eval import summarize_review_rows
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
    "MaterializedAgentInstances",
    "OpenAICompatibleAgentClient",
    "ParsedPrediction",
    "PromptTemplateSpec",
    "describe_prompt_template",
    "get_prompt_template",
    "list_prompt_templates",
    "safe_case_slug",
    "select_events",
    "agent_instance_path",
    "ensure_agent_predict_splits",
    "materialize_agent_instances",
    "render_prompt",
    "summarize_prediction_rows",
    "summarize_review_rows",
    "validate_agent_predict_setup",
]
