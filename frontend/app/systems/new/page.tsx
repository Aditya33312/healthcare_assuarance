"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const steps = [
  "Basic information", "Intended use", "Input / output", "AI characteristics",
  "Endpoint", "Authentication scope", "Request mapping", "Response mapping",
  "Connection-test status", "Test-selection behavior", "Policy handoff", "Save"
];

const initialForm = {
  name: "", description: "", version: "0.1.0", application_type: "CLINICAL_SCRIBE",
  intended_use: "", clinical_specialty: "", input_type: "CLINICAL_TRANSCRIPT",
  output_type: "CLINICAL_NOTE", patient_facing: false, clinician_facing: true,
  uses_rag: false, uses_tools: false, uses_llm: true, uses_predictive_model: false,
  environment: "STAGING", risk_tier: "HIGH", endpoint_url: "",
  authentication_type: "API_KEY", request_schema: "{}", response_schema: "{}"
};

export default function NewSystemPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState(initialForm);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const update = (name: string, value: string | boolean) => setForm((current) => ({...current, [name]: value}));

  async function save() {
    setSaving(true); setError(null);
    try {
      const response = await fetch("/api/gatekeeper/systems", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...form, request_schema: JSON.parse(form.request_schema), response_schema: JSON.parse(form.response_schema), endpoint_url: form.endpoint_url || null})
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body?.error?.message ?? "Registration failed.");
      }
      router.push("/systems"); router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Registration failed.");
    } finally { setSaving(false); }
  }

  return <>
    <div className="page-header"><div><span className="eyebrow">Registration wizard</span><h1>Add an AI application</h1><p>The profile determines applicable tests; an administrator must still confirm the generated plan.</p></div></div>
    <div className="wizard-shell">
      <aside className="wizard-steps" aria-label="Registration steps">{steps.map((label, index) => <div className={index === step ? "wizard-step active" : "wizard-step"} key={label}><span>{index + 1}</span>{label}</div>)}</aside>
      <section className="panel"><div className="panel-header"><h2>{steps[step]}</h2><span className="muted">Step {step + 1} of {steps.length}</span></div><div className="panel-body">
        {step === 0 ? <div className="form-grid"><Field label="Application name" name="name" value={form.name} onChange={update} /><Field label="Version" name="version" value={form.version} onChange={update} /><Field full label="Description" name="description" value={form.description} onChange={update} textarea /><Select label="Application type" name="application_type" value={form.application_type} onChange={update} options={["CLINICAL_SCRIBE","CLINICAL_SUMMARIZER","CLINICAL_CHATBOT","PATIENT_ASSISTANT","RAG_ASSISTANT","CLINICAL_DECISION_SUPPORT","CLINICAL_LETTER_GENERATOR","HEALTHCARE_ADMIN_AI","AGENTIC_AI","OTHER"]} /></div> : null}
        {step === 1 ? <div className="form-grid"><Field full label="Intended use" name="intended_use" value={form.intended_use} onChange={update} textarea /><Field label="Clinical specialty" name="clinical_specialty" value={form.clinical_specialty} onChange={update} /><Select label="Risk tier" name="risk_tier" value={form.risk_tier} onChange={update} options={["LOW","MODERATE","HIGH","CRITICAL"]} /><Check label="Patient-facing" name="patient_facing" checked={form.patient_facing} onChange={update} /><Check label="Clinician-facing" name="clinician_facing" checked={form.clinician_facing} onChange={update} /></div> : null}
        {step === 2 ? <div className="form-grid"><Select label="Input type" name="input_type" value={form.input_type} onChange={update} options={["CLINICAL_TEXT","CLINICAL_TRANSCRIPT","QUESTION","DOCUMENT","STRUCTURED_DATA","AUDIO","IMAGE","MULTIMODAL","OTHER"]} /><Select label="Output type" name="output_type" value={form.output_type} onChange={update} options={["CLINICAL_TEXT","CLINICAL_NOTE","CLINICAL_SUMMARY","CLINICAL_LETTER","ANSWER","RECOMMENDATION","STRUCTURED_DATA","TOOL_ACTION","MULTIMODAL","OTHER"]} /></div> : null}
        {step === 3 ? <div className="form-grid"><Check label="Uses an LLM" name="uses_llm" checked={form.uses_llm} onChange={update} /><Check label="Uses RAG" name="uses_rag" checked={form.uses_rag} onChange={update} /><Check label="Uses tools / actions" name="uses_tools" checked={form.uses_tools} onChange={update} /><Check label="Uses predictive model" name="uses_predictive_model" checked={form.uses_predictive_model} onChange={update} /></div> : null}
        {step === 4 ? <div className="form-grid"><Field full label="Staging endpoint URL" name="endpoint_url" value={form.endpoint_url} onChange={update} /><Select label="Environment" name="environment" value={form.environment} onChange={update} options={["TEST","STAGING","PRODUCTION"]} /><div className="callout warning">Use a controlled staging endpoint. Phase 1 must not target uncontrolled production data.</div></div> : null}
        {step === 5 ? <div className="form-grid"><Select label="Authentication type" name="authentication_type" value={form.authentication_type} onChange={update} options={["NONE","API_KEY","BEARER_TOKEN","BASIC","OAUTH2","MTLS","CUSTOM"]} /><div className="callout warning">This portal records the authentication type only. Credential authoring is not implemented here; configure encrypted credentials through the scoped API before running an endpoint-backed suite.</div></div> : null}
        {step === 6 ? <div className="form-grid"><Field full label="Request JSON schema / field mapping" name="request_schema" value={form.request_schema} onChange={update} textarea mono /></div> : null}
        {step === 7 ? <div className="form-grid"><Field full label="Response JSON schema / output path" name="response_schema" value={form.response_schema} onChange={update} textarea mono /></div> : null}
        {step === 8 ? <div className="callout warning">No connection test is executed by this registration wizard. Verify the endpoint and encrypted credentials through an approved operational workflow before creating a run.</div> : null}
        {step === 9 ? <div className="stack"><div className="callout">The backend records the deterministic 16-family selection after registration. Review the generated classification when drafting the test plan; this wizard does not approve it.</div><p className="muted">RAG, patient-facing, tool-using, predictive, and high-risk flags promote specific tests without an unconstrained LLM.</p></div> : null}
        {step === 10 ? <div className="form-grid"><div className="callout full">Policy selection happens later, when a plan is drafted. Critical findings can block independently of an average score; missing required coverage can withhold a decision.</div></div> : null}
        {step === 11 ? <div className="stack"><div className="callout">Review the application profile, save it, then explicitly confirm the generated test plan before execution.</div><dl><dt>Name</dt><dd>{form.name || "Not provided"}</dd><dt>Type</dt><dd>{form.application_type}</dd><dt>Risk</dt><dd>{form.risk_tier}</dd><dt>Endpoint</dt><dd>{form.endpoint_url || "Not configured"}</dd></dl>{error ? <div className="callout warning">{error}</div> : null}</div> : null}
        <div className="wizard-actions"><button className="button secondary" disabled={step === 0 || saving} onClick={() => setStep((value) => value - 1)}>Back</button>{step < steps.length - 1 ? <button className="button primary" onClick={() => setStep((value) => value + 1)}>Continue</button> : <button className="button primary" disabled={saving || !form.name || !form.intended_use} onClick={save}>{saving ? "Saving…" : "Save application"}</button>}</div>
      </div></section>
    </div>
  </>;
}

function Field({label,name,value,onChange,full,textarea,mono}: {label:string;name:string;value:string;onChange:(name:string,value:string)=>void;full?:boolean;textarea?:boolean;mono?:boolean}) {
  return <div className={`field ${full ? "full" : ""}`}><label htmlFor={name}>{label}</label>{textarea ? <textarea className={mono ? "mono" : ""} id={name} value={value} onChange={(event) => onChange(name,event.target.value)} /> : <input id={name} value={value} onChange={(event) => onChange(name,event.target.value)} />}</div>;
}
function Select({label,name,value,onChange,options}: {label:string;name:string;value:string;onChange:(name:string,value:string)=>void;options:string[]}) {
  return <div className="field"><label htmlFor={name}>{label}</label><select id={name} value={value} onChange={(event) => onChange(name,event.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select></div>;
}
function Check({label,name,checked,onChange}: {label:string;name:string;checked:boolean;onChange:(name:string,value:boolean)=>void}) {
  return <label className="check-row"><input type="checkbox" checked={checked} onChange={(event) => onChange(name,event.target.checked)} />{label}</label>;
}
