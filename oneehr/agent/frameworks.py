"""LLM agent framework adapters for clinical prediction.

Each framework adapter takes a client, system config, and patient context,
then returns a ParsedPrediction via the shared schema parser.

Frameworks are lightweight adapters inspired by MedAgentAudit:
- single_llm: Direct single-model inference
- medagent: Multi-specialty MDT consultation
- colacare: Collaborative doctor analysis
- healthcareagent: Inquiry-based with safety review
- mac: Multi-agent conversation with supervisor
- mdagents: Complexity-adaptive routing
- reconcile: Confidence-weighted consensus
"""

from __future__ import annotations

from typing import Any, Callable

from oneehr.agent.client import OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec, ParsedPrediction
from oneehr.agent.runtime import execute_agent_request
from oneehr.agent.schema import parse_prediction_response, schema_prompt_text
from oneehr.config.schema import SystemConfig


FrameworkRunner = Callable[
    [OpenAICompatibleAgentClient, SystemConfig, str, str],
    ParsedPrediction | None,
]

_FRAMEWORK_REGISTRY: dict[str, FrameworkRunner] = {}


def register_framework(name: str) -> Callable[[FrameworkRunner], FrameworkRunner]:
    def decorator(fn: FrameworkRunner) -> FrameworkRunner:
        _FRAMEWORK_REGISTRY[name] = fn
        return fn
    return decorator


def get_framework(name: str) -> FrameworkRunner:
    runner = _FRAMEWORK_REGISTRY.get(name)
    if runner is None:
        available = ", ".join(sorted(_FRAMEWORK_REGISTRY))
        raise KeyError(f"Unknown framework {name!r}. Available: {available}")
    return runner


def list_frameworks() -> list[str]:
    return sorted(_FRAMEWORK_REGISTRY)


def _make_request(
    system_cfg: SystemConfig,
    prompt: str,
    system_prompt: str | None = None,
) -> AgentRequestSpec:
    """Build an AgentRequestSpec from SystemConfig + prompt."""
    params = system_cfg.params or {}

    # Determine base_url from backend
    backend = system_cfg.backend.lower()
    base_urls = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "together": "https://api.together.xyz/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    base_url = str(params.get("base_url", base_urls.get(backend, "https://api.openai.com/v1")))

    return AgentRequestSpec(
        backend_name=backend,
        provider_model=system_cfg.model,
        base_url=base_url,
        api_key_env=system_cfg.api_key_env,
        prompt=prompt,
        system_prompt=system_prompt,
        response_format=None,
        temperature=float(params.get("temperature", 0.3)),
        top_p=float(params.get("top_p", 1.0)),
        timeout_seconds=float(params.get("timeout_seconds", 120.0)),
        max_retries=int(params.get("max_retries", 2)),
        seed=int(params["seed"]) if "seed" in params else None,
    )


def _call_llm(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    prompt: str,
    task_kind: str,
    system_prompt: str | None = None,
) -> ParsedPrediction | None:
    """Make a single LLM call and parse the prediction response."""
    request = _make_request(system_cfg, prompt, system_prompt)
    result = execute_agent_request(
        client=client,
        request=request,
        parse_response=lambda text: parse_prediction_response(
            text,
            task_kind=task_kind,
            include_explanation=True,
            include_confidence=True,
        ),
    )
    return result.parsed


def _schema_text(task_kind: str) -> str:
    return schema_prompt_text(
        task_kind=task_kind,
        include_explanation=True,
        include_confidence=True,
    )


# ---------------------------------------------------------------------------
# Framework implementations
# ---------------------------------------------------------------------------


@register_framework("single_llm")
def run_single_llm(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    patient_context: str,
    task_kind: str,
) -> ParsedPrediction | None:
    """Direct single-model inference."""
    schema_text = _schema_text(task_kind)
    prompt = (
        f"{patient_context}\n\n"
        f"Output Schema:\n{schema_text}\n\n"
        "Return exactly one JSON object matching the schema above."
    )
    return _call_llm(client, system_cfg, prompt, task_kind,
                     system_prompt="You are a clinical prediction model. Analyze the patient data and make a prediction.")


@register_framework("medagent")
def run_medagent(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    patient_context: str,
    task_kind: str,
) -> ParsedPrediction | None:
    """Multi-disciplinary team (MDT) consultation.

    Steps: Expert gathering → Doctor opinions → Meta-synthesis → Decision.
    """
    params = system_cfg.params or {}
    max_rounds = int(params.get("max_rounds", 2))
    schema_text = _schema_text(task_kind)

    # Step 1: Gather relevant specialties
    gather_prompt = (
        f"Given this patient case, list 3 relevant medical specialties "
        f"that should be consulted (one per line, just the specialty name):\n\n{patient_context}"
    )
    gather_req = _make_request(system_cfg, gather_prompt,
                               "You are a medical expert coordinator.")
    gather_result = execute_agent_request(
        client=client, request=gather_req,
        parse_response=lambda t: t,
    )
    specialties = [s.strip("- ").strip() for s in (gather_result.raw_response or "").strip().split("\n") if s.strip()][:3]
    if not specialties:
        specialties = ["Internal Medicine", "Critical Care", "Emergency Medicine"]

    # Step 2: Get doctor opinions
    opinions = []
    for specialty in specialties:
        doc_prompt = (
            f"You are a {specialty} specialist. Analyze this patient case and provide "
            f"your clinical assessment and prediction.\n\n{patient_context}\n\n"
            f"Provide your assessment in 2-3 sentences."
        )
        doc_req = _make_request(system_cfg, doc_prompt,
                                f"You are a {specialty} specialist.")
        doc_result = execute_agent_request(
            client=client, request=doc_req, parse_response=lambda t: t,
        )
        opinions.append(f"{specialty}: {doc_result.raw_response}")

    # Step 3: Synthesize and decide
    synthesis_prompt = (
        f"Based on the following specialist opinions about a patient case, "
        f"synthesize a final prediction.\n\n"
        f"Patient Case:\n{patient_context}\n\n"
        f"Specialist Opinions:\n" + "\n".join(opinions) + "\n\n"
        f"Output Schema:\n{schema_text}\n\n"
        f"Return exactly one JSON object matching the schema above."
    )
    return _call_llm(client, system_cfg, synthesis_prompt, task_kind,
                     system_prompt="You are a medical decision synthesizer. Combine specialist opinions into a final prediction.")


@register_framework("colacare")
def run_colacare(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    patient_context: str,
    task_kind: str,
) -> ParsedPrediction | None:
    """Collaborative care: multiple doctors discuss and reach consensus."""
    params = system_cfg.params or {}
    n_doctors = int(params.get("n_doctors", 3))
    schema_text = _schema_text(task_kind)

    # Get initial analyses
    analyses = []
    roles = ["attending physician", "specialist consultant", "senior resident"][:n_doctors]
    for role in roles:
        prompt = (
            f"You are a {role}. Analyze this patient case and provide "
            f"your initial assessment.\n\n{patient_context}\n\n"
            f"Provide a brief clinical analysis (2-3 sentences)."
        )
        req = _make_request(system_cfg, prompt, f"You are a {role}.")
        result = execute_agent_request(client=client, request=req, parse_response=lambda t: t)
        analyses.append(f"{role}: {result.raw_response}")

    # Collaborative synthesis
    collab_prompt = (
        f"The following clinical team has analyzed a patient case. "
        f"Synthesize their analyses into a final prediction.\n\n"
        f"Patient Case:\n{patient_context}\n\n"
        f"Team Analyses:\n" + "\n".join(analyses) + "\n\n"
        f"Output Schema:\n{schema_text}\n\n"
        f"Return exactly one JSON object matching the schema above."
    )
    return _call_llm(client, system_cfg, collab_prompt, task_kind,
                     system_prompt="You are a collaborative care coordinator. Synthesize team opinions.")


@register_framework("healthcareagent")
def run_healthcareagent(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    patient_context: str,
    task_kind: str,
) -> ParsedPrediction | None:
    """Healthcare agent: inquiry → analysis → safety review → decision."""
    schema_text = _schema_text(task_kind)

    # Step 1: Preliminary analysis
    analysis_prompt = (
        f"Analyze this patient case and identify key clinical indicators, "
        f"risk factors, and relevant patterns.\n\n{patient_context}\n\n"
        f"Provide a structured clinical analysis."
    )
    analysis_req = _make_request(system_cfg, analysis_prompt,
                                  "You are a clinical analyst.")
    analysis_result = execute_agent_request(
        client=client, request=analysis_req, parse_response=lambda t: t,
    )

    # Step 2: Safety review
    safety_prompt = (
        f"Review this clinical analysis for safety concerns, potential errors, "
        f"and ethical considerations.\n\nAnalysis:\n{analysis_result.raw_response}\n\n"
        f"Provide a brief safety assessment."
    )
    safety_req = _make_request(system_cfg, safety_prompt,
                                "You are a clinical safety reviewer.")
    safety_result = execute_agent_request(
        client=client, request=safety_req, parse_response=lambda t: t,
    )

    # Step 3: Final decision
    decision_prompt = (
        f"Based on the clinical analysis and safety review, make a final prediction.\n\n"
        f"Patient Case:\n{patient_context}\n\n"
        f"Clinical Analysis:\n{analysis_result.raw_response}\n\n"
        f"Safety Review:\n{safety_result.raw_response}\n\n"
        f"Output Schema:\n{schema_text}\n\n"
        f"Return exactly one JSON object matching the schema above."
    )
    return _call_llm(client, system_cfg, decision_prompt, task_kind,
                     system_prompt="You are a clinical decision maker with safety awareness.")


@register_framework("mac")
def run_mac(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    patient_context: str,
    task_kind: str,
) -> ParsedPrediction | None:
    """Multi-Agent Conversation: agents discuss under supervisor."""
    params = system_cfg.params or {}
    n_rounds = int(params.get("n_rounds", 2))
    schema_text = _schema_text(task_kind)

    agents = ["Doctor A", "Doctor B", "Doctor C"]
    discussion: list[str] = []

    for round_num in range(n_rounds):
        for agent_name in agents:
            context = f"Patient Case:\n{patient_context}"
            if discussion:
                context += "\n\nPrior Discussion:\n" + "\n".join(discussion[-6:])

            prompt = (
                f"You are {agent_name} in a medical team discussion (round {round_num + 1}). "
                f"Provide your clinical opinion about this case.\n\n{context}\n\n"
                f"Give a brief opinion (2-3 sentences)."
            )
            req = _make_request(system_cfg, prompt, f"You are {agent_name}.")
            result = execute_agent_request(client=client, request=req, parse_response=lambda t: t)
            discussion.append(f"{agent_name} (round {round_num + 1}): {result.raw_response}")

    # Supervisor synthesis
    supervisor_prompt = (
        f"As the supervising physician, synthesize this team discussion into a final prediction.\n\n"
        f"Patient Case:\n{patient_context}\n\n"
        f"Discussion:\n" + "\n".join(discussion) + "\n\n"
        f"Output Schema:\n{schema_text}\n\n"
        f"Return exactly one JSON object matching the schema above."
    )
    return _call_llm(client, system_cfg, supervisor_prompt, task_kind,
                     system_prompt="You are the supervising physician. Make a final clinical prediction.")


@register_framework("mdagents")
def run_mdagents(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    patient_context: str,
    task_kind: str,
) -> ParsedPrediction | None:
    """MDAgents: complexity-adaptive routing."""
    schema_text = _schema_text(task_kind)

    # Step 1: Assess complexity
    complexity_prompt = (
        f"Assess the clinical complexity of this case. "
        f"Reply with exactly one word: basic, intermediate, or advanced.\n\n{patient_context}"
    )
    cx_req = _make_request(system_cfg, complexity_prompt,
                            "You are a clinical complexity assessor.")
    cx_result = execute_agent_request(client=client, request=cx_req, parse_response=lambda t: t)
    complexity = (cx_result.raw_response or "").strip().lower()

    if "advanced" in complexity:
        # Multi-team approach
        return run_medagent(client, system_cfg, patient_context, task_kind)
    elif "intermediate" in complexity:
        # Collaborative approach
        return run_colacare(client, system_cfg, patient_context, task_kind)
    else:
        # Single model (basic)
        return run_single_llm(client, system_cfg, patient_context, task_kind)


@register_framework("reconcile")
def run_reconcile(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    patient_context: str,
    task_kind: str,
) -> ParsedPrediction | None:
    """ReConcile: confidence-weighted consensus rounds."""
    params = system_cfg.params or {}
    n_rounds = int(params.get("n_rounds", 2))
    n_agents = int(params.get("n_agents", 3))
    schema_text = _schema_text(task_kind)

    agents = [f"Agent {i+1}" for i in range(n_agents)]
    predictions: list[str] = []

    for round_num in range(n_rounds):
        round_preds = []
        for agent_name in agents:
            context = f"Patient Case:\n{patient_context}"
            if predictions:
                context += "\n\nPrevious round predictions:\n" + "\n".join(predictions[-n_agents:])

            prompt = (
                f"You are {agent_name} (round {round_num + 1}). "
                f"Analyze this case and provide your prediction with a confidence score [0-1].\n\n"
                f"{context}\n\n"
                f"Output Schema:\n{schema_text}\n\n"
                f"Return exactly one JSON object matching the schema above."
            )
            pred = _call_llm(client, system_cfg, prompt, task_kind,
                             system_prompt=f"You are {agent_name}, a clinical prediction agent.")
            if pred and pred.parsed_ok:
                conf = pred.confidence if pred.confidence is not None else 0.5
                round_preds.append(f"{agent_name}: prediction={pred.prediction}, confidence={conf:.2f}")
            else:
                round_preds.append(f"{agent_name}: failed to parse")
        predictions.extend(round_preds)

    # Final consensus
    consensus_prompt = (
        f"Based on multi-round agent predictions, determine the final consensus prediction.\n\n"
        f"Patient Case:\n{patient_context}\n\n"
        f"All Predictions:\n" + "\n".join(predictions) + "\n\n"
        f"Output Schema:\n{schema_text}\n\n"
        f"Return exactly one JSON object matching the schema above. "
        f"Weight predictions by confidence scores."
    )
    return _call_llm(client, system_cfg, consensus_prompt, task_kind,
                     system_prompt="You are a consensus builder. Combine confidence-weighted predictions.")
