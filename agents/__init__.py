import os
import json
import requests
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# Define standard data schemas
class AgentReport(BaseModel):
    agent_name: str = Field(description="Name of the agent generating this audit section")
    score: int = Field(description="Audit score between 0 and 100", ge=0, le=100)
    status: str = Field(description="Status of this audit: PASS, WARNING, or FAIL")
    findings: List[str] = Field(description="List of key observations or validation findings (exactly 3-5 items)")
    recommendations: List[str] = Field(description="List of actionable recommendations (exactly 3-5 items)")
    metadata: Dict[str, Any] = Field(default={}, description="Optional key-value details for this section")

class FinalSummarizedReport(BaseModel):
    overall_status: str = Field(description="Final deployment status: APPROVED, APPROVED WITH CONDITIONS, or REJECTED")
    overall_risk_level: str = Field(description="Overall risk level: LOW, MEDIUM, or HIGH")
    overall_risk_score: int = Field(description="Synthesized overall risk score between 0 and 100", ge=0, le=100)
    executive_summary: str = Field(description="High-level narrative summarizing the entire audit outcome")
    key_critical_findings: List[str] = Field(description="Synthesized list of critical findings across all audits")
    remediation_plan: List[str] = Field(description="Prioritized roadmap of actions to address vulnerabilities")
    deployment_readiness_reasons: List[str] = Field(description="Explanation of why this deployment decision was reached")

# Try to import CrewAI
try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew as CrewClass, Process as CrewProcess, LLM as CrewLLM
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

# Fallback classes if CrewAI is not available
class MockLLM:
    def __init__(self, model: str, api_key: str, temperature: float = 0.1, max_tokens: int = 3000, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

class MockAgent:
    def __init__(self, role: str, goal: str, backstory: str, verbose: bool = True, allow_delegation: bool = False, llm: Any = None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.llm = llm

class MockTaskOutput:
    def __init__(self, pydantic_obj: Any):
        self.pydantic = pydantic_obj

class MockTask:
    def __init__(self, description: str, expected_output: str, agent: MockAgent, output_pydantic: Any):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.output_pydantic = output_pydantic
        self.output = None

class MockCrew:
    def __init__(self, agents: List[MockAgent], tasks: List[MockTask], process: Any = None, verbose: bool = True):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose

    def kickoff(self) -> Any:
        print("[Fallback Orchestrator] Executing tasks sequentially using direct LLM API calls...")
        
        AGENT_REPORT_SCHEMA = {
            "type": "OBJECT",
            "properties": {
                "agent_name": {"type": "STRING", "description": "Name of the agent generating this audit section"},
                "score": {"type": "INTEGER", "description": "Audit score between 0 and 100"},
                "status": {"type": "STRING", "description": "Status of this audit: PASS, WARNING, or FAIL"},
                "findings": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of key observations or validation findings (exactly 3-5 items)"},
                "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of actionable recommendations (exactly 3-5 items)"},
                "metadata": {"type": "OBJECT", "description": "Optional key-value details for this section"}
            },
            "required": ["agent_name", "score", "status", "findings", "recommendations"]
        }

        FINAL_SUMMARIZED_REPORT_SCHEMA = {
            "type": "OBJECT",
            "properties": {
                "overall_status": {"type": "STRING", "description": "Final deployment status: APPROVED, APPROVED WITH CONDITIONS, or REJECTED"},
                "overall_risk_level": {"type": "STRING", "description": "Overall risk level: LOW, MEDIUM, or HIGH"},
                "overall_risk_score": {"type": "INTEGER", "description": "Synthesized overall risk score between 0 and 100"},
                "executive_summary": {"type": "STRING", "description": "High-level narrative summarizing the entire audit outcome"},
                "key_critical_findings": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Synthesized list of critical findings across all audits"},
                "remediation_plan": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Prioritized roadmap of actions to address vulnerabilities"},
                "deployment_readiness_reasons": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Explanation of why this deployment decision was reached"}
            },
            "required": ["overall_status", "overall_risk_level", "overall_risk_score", "executive_summary", "key_critical_findings", "remediation_plan", "deployment_readiness_reasons"]
        }

        previous_outputs = {}

        for idx, task in enumerate(self.tasks):
            print(f"[Fallback Orchestrator] Running Task {idx + 1}/{len(self.tasks)}: {task.agent.role}...")
            
            prompt = f"""You are the following AI Governance Agent:
Role: {task.agent.role}
Goal: {task.agent.goal}
Backstory: {task.agent.backstory}

Your current task instructions:
{task.description}

Expected Output format:
{task.expected_output}
"""
            if previous_outputs:
                prompt += "\n\nBelow are the outputs from the preceding governance audits. Use them to synthesize your results:\n"
                for name, out in previous_outputs.items():
                    prompt += f"--- {name} Results ---\n{json.dumps(out, indent=2)}\n"

            is_final_task = (idx == len(self.tasks) - 1)
            schema = FINAL_SUMMARIZED_REPORT_SCHEMA if is_final_task else AGENT_REPORT_SCHEMA
            
            llm = task.agent.llm
            result_dict = self._call_llm_api(llm, prompt, schema, task.output_pydantic)
            
            pydantic_obj = task.output_pydantic(**result_dict)
            task.output = MockTaskOutput(pydantic_obj)
            
            agent_label = task.agent.role.replace(" ", "_").lower()
            previous_outputs[agent_label] = result_dict

        return self.tasks[-1].output.pydantic

    def _call_llm_api(self, llm: Any, prompt: str, schema: dict, schema_pydantic: Any) -> dict:
        model_name = llm.model
        api_key = llm.api_key
        
        if "gemini" in model_name.lower():
            clean_model = model_name.split("/")[-1]
            if clean_model == "gemini":
                clean_model = "gemini-2.5-flash"
                
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 8192,
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                    "thinkingConfig": {
                        "thinkingBudget": 0
                    }
                }
            }
            
            resp = requests.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")
                
            result = resp.json()
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Failed to parse Gemini response: {result}. Error: {e}")
                
        else:
            clean_model = model_name
            if clean_model.startswith("openrouter/"):
                clean_model = clean_model[len("openrouter/"):]
                
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/venu/ml-governance-pipeline",
                "X-Title": "AI Governance Platform"
            }
            
            prompt_with_schema = prompt + f"\n\nIMPORTANT: You must return a valid JSON object that conforms strictly to the following Pydantic schema:\n{json.dumps(schema_pydantic.model_json_schema())}"
            
            openrouter_max_tokens = min(llm.max_tokens, int(os.getenv("OPENROUTER_MAX_TOKENS", "2048")))

            payload = {
                "model": clean_model,
                "messages": [{"role": "user", "content": prompt_with_schema}],
                "temperature": 0.1,
                "max_tokens": openrouter_max_tokens,
                "response_format": {"type": "json_object"}
            }
            
            resp = requests.post(url, headers=headers, json=payload)
            if resp.status_code == 402 and openrouter_max_tokens > 1024:
                payload["max_tokens"] = 1024
                resp = requests.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API error ({resp.status_code}): {resp.text}")
                
            result = resp.json()
            try:
                text = result['choices'][0]['message']['content']
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return json.loads(text)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Failed to parse OpenRouter response: {result}. Error: {e}")

# Define API wrappers
if HAS_CREWAI:
    LLM = CrewLLM
    Agent = CrewAgent
    Task = CrewTask
    Crew = CrewClass
    Process = CrewProcess
else:
    LLM = MockLLM
    Agent = MockAgent
    Task = MockTask
    Crew = MockCrew
    Process = type('MockProcess', (), {'sequential': 'sequential'})()

def get_llm(provider: str, api_key: str) -> Any:
    """
    Returns the appropriate CrewAI LLM wrapper or fallback wrapper.
    """
    if provider == "Gemini":
        return LLM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.1,
            max_tokens=8192
        )
    elif provider == "DeepSeek (OpenRouter)":
        return LLM(
            model="openrouter/deepseek/deepseek-r1",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.1,
            max_tokens=2048
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def create_governance_crew(
    llm: Any,
    model_metadata: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    explainability_report: Dict[str, Any],
    fairness_report: Optional[Dict[str, Any]],
    drift_report: Optional[Dict[str, Any]],
    documentation_text: str
) -> Crew:
    """
    Constructs and configures the multi-agent Crew (or fallback Crew) for model governance auditing.
    """
    
    # Define Agents
    gov_agent = Agent(
        role="Lead AI Governance & Corporate Auditor",
        goal="Strictly validate corporate policy, version traceability, and documentation integrity across model assets.",
        backstory=(
            "You are a rigorous lead auditor specializing in financial model risk management frameworks (like SR 11-7). "
            "Your job is to check for clear ownership, complete version tracking, thorough documentation, and clean data lineage."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    perf_agent = Agent(
        role="ML Model Performance Validator",
        goal="Rigorously evaluate accuracy, precision, recall, and other performance metrics against corporate thresholds.",
        backstory=(
            "You are a senior quantitative validator. You examine performance scores, classification reports, "
            "and compare metric scores against mandatory thresholds to verify model stability and accuracy."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    risk_agent = Agent(
        role="Model Risk & Explainability Analyst",
        goal="Audit explainability artifacts (SHAP importances) and quantify risks, security vulnerability, and explainability compliance.",
        backstory=(
            "You are an expert in explainable AI (XAI) and adversarial ML defense. "
            "You analyze feature importances to check for unexpected proxy features, vulnerability, or feature over-reliance."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    compliance_agent = Agent(
        role="AI Regulatory Compliance Counsel",
        goal="Evaluate model and documentation compliance against global standards like EU AI Act, RBI AI Guidelines, and NIST AI RMF.",
        backstory=(
            "You are an expert AI compliance officer with a deep legal background in AI safety, data privacy, and global policies. "
            "You map the model's design, explanation, and lineage to regulatory checks."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    deployment_agent = Agent(
        role="Model Release Gatekeeper",
        goal="Verify if the model satisfies all technical release conditions, including validation coverage and monitoring setups.",
        backstory=(
            "You are the operations head of model release. You verify if monitoring is ready, metadata is complete, "
            "versioning is established, and testing is thorough before signing off."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    summarizer_agent = Agent(
        role="Principal AI Governance Officer",
        goal="Compile the final synthesized report, risk assessment score, and remediation plan based on all agent audits.",
        backstory=(
            "You are the Chief AI Officer who reviews all technical audits and creates a summary for regulators and executive boards. "
            "You weigh findings to construct a final score (0-100) and recommendation status."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Input Payload
    payload = {
        "metadata": model_metadata,
        "performance_metrics": performance_metrics,
        "explainability_report": explainability_report,
        "fairness_report": fairness_report or {"status": "NOT_PERFORMED", "warning": "No fairness metrics calculated"},
        "drift_report": drift_report or {"status": "NOT_PERFORMED", "warning": "No drift detection performed"},
        "documentation_text": documentation_text[:10000]
    }
    payload_str = json.dumps(payload, indent=2)

    # Define Tasks
    task_gov = Task(
        description=f"Evaluate corporate governance, metadata completeness, and documentation lineage from inputs:\n{payload_str}\n"
                    "Calculate a governance score (0-100) deducting points for missing ownership or documentation.",
        expected_output="An AgentReport containing governance audit score, status, findings, and recommendations.",
        agent=gov_agent,
        output_pydantic=AgentReport
    )
    
    task_perf = Task(
        description=f"Inspect performance metrics and threshold violations from inputs:\n{payload_str}\n"
                    "Check thresholds: Accuracy >= 0.85, Precision >= 0.80, Recall >= 0.80. Assign a score (0-100) based on threshold fulfillment.",
        expected_output="An AgentReport containing performance audit score, status, findings, and recommendations.",
        agent=perf_agent,
        output_pydantic=AgentReport
    )
    
    task_risk = Task(
        description=f"Analyze explainability, feature dependencies, and SHAP reports from inputs:\n{payload_str}\n"
                    "Check if features show over-reliance on sensitive items. Determine risk score (0-100).",
        expected_output="An AgentReport containing risk analysis score, status, findings, and recommendations.",
        agent=risk_agent,
        output_pydantic=AgentReport
    )
    
    task_comp = Task(
        description=f"Evaluate regulatory alignment with EU AI Act, RBI guidelines, and NIST AI RMF using documentation and metrics:\n{payload_str}\n"
                    "Grade compliance (0-100).",
        expected_output="An AgentReport containing compliance score, status, findings, and recommendations.",
        agent=compliance_agent,
        output_pydantic=AgentReport
    )
    
    task_deploy = Task(
        description=f"Assess model monitoring setups, version status, and deployment metadata from inputs:\n{payload_str}\n"
                    "Rate release readiness (0-100).",
        expected_output="An AgentReport containing deployment score, status, findings, and recommendations.",
        agent=deployment_agent,
        output_pydantic=AgentReport
    )
    
    task_summarize = Task(
        description="Synthesize the audit reports from the 5 preceding tasks (Governance, Performance, Risk, Compliance, Deployment) "
                    "into a unified executive sign-off report. Calculate a weighted overall risk score (0-100) and issue a deployment recommendation.",
        expected_output="A FinalSummarizedReport containing final deployment status, overall risk level, overall risk score, executive summary, critical findings, remediation plan, and readiness reasons.",
        agent=summarizer_agent,
        output_pydantic=FinalSummarizedReport
    )
    
    # Create Crew
    crew = Crew(
        agents=[gov_agent, perf_agent, risk_agent, compliance_agent, deployment_agent, summarizer_agent],
        tasks=[task_gov, task_perf, task_risk, task_comp, task_deploy, task_summarize],
        process=Process.sequential,
        verbose=True
    )
    
    return crew

# Expose legacy functions for backward compatibility
from .governance   import run_governance_agent
from .performance import run_performance_agent
from .risk        import run_risk_agent
from .summarizer  import run_summarizer_agent

__all__ = [
    "run_governance_agent",
    "run_performance_agent",
    "run_risk_agent",
    "run_summarizer_agent",
    "get_llm",
    "create_governance_crew",
    "HAS_CREWAI",
    "AgentReport",
    "FinalSummarizedReport"
]