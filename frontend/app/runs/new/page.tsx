"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

export default function NewRunPage() {
  const router = useRouter();
  const [form, setForm] = useState({system_id:"",system_version:"",environment:"staging",test_suite_id:"",test_suite_version:"",test_plan_id:""});
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const attemptId = useRef<string | null>(null);
  useEffect(() => {
    const search = new URLSearchParams(window.location.search);
    const requestedEnvironment = search.get("environment")?.trim();
    const environment = requestedEnvironment &&
      ["test", "staging", "production"].includes(requestedEnvironment)
      ? requestedEnvironment
      : null;
    setForm((current) => ({
      ...current,
      system_id: search.get("system_id")?.trim() || current.system_id,
      environment: environment || current.environment,
      test_suite_id: search.get("test_suite_id")?.trim() || current.test_suite_id,
      test_suite_version: search.get("test_suite_version")?.trim() || current.test_suite_version,
      test_plan_id: search.get("test_plan_id")?.trim() || current.test_plan_id
    }));
  }, []);
  const update = (name:string,value:string) => {
    attemptId.current = null;
    setForm((current) => ({...current,[name]:value}));
  };
  async function submit() {
    setSubmitting(true); setError(null);
    try {
      attemptId.current ??= crypto.randomUUID();
      const response = await fetch("/api/gatekeeper/test-runs", {method:"POST",headers:{"Content-Type":"application/json","X-Gatekeeper-Portal-Attempt-ID":attemptId.current},body:JSON.stringify({...form,source:{trigger:"portal"}})});
      const body = await response.json();
      if (!response.ok) throw new Error(body?.error?.message ?? "Unable to queue the test run.");
      router.push(`/runs/${body.run_id ?? body.id}`); router.refresh();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Unable to queue the test run. Retry will reuse the same safe attempt identifier."); }
    finally { setSubmitting(false); }
  }
  return <>
    <div className="page-header"><div><span className="eyebrow">Assurance execution</span><h1>Run test suite</h1><p>Only confirmed test plans should be executed against a controlled staging endpoint.</p></div></div>
    <section className="panel"><div className="panel-header"><h2>Run configuration</h2></div><div className="panel-body">
      <div className="form-grid">
        <RunField label="System ID" name="system_id" value={form.system_id} onChange={update} />
        <RunField label="System version" name="system_version" value={form.system_version} onChange={update} />
        <RunField label="Test suite ID" name="test_suite_id" value={form.test_suite_id} onChange={update} />
        <RunField label="Test suite version" name="test_suite_version" value={form.test_suite_version} onChange={update} />
        <RunField label="Confirmed test plan ID" name="test_plan_id" value={form.test_plan_id} onChange={update} />
        <div className="field"><label htmlFor="environment">Environment</label><select id="environment" value={form.environment} onChange={(event)=>update("environment",event.target.value)}><option value="test">Test</option><option value="staging">Staging</option><option value="production">Production (LIVE key required)</option></select></div>
        <div className="callout warning">Gatekeeper limits concurrency and retries only transient endpoint failures. A 401 or schema error stops execution.</div>
      </div>
      {error ? <div className="callout warning" style={{marginTop:16}}>{error}</div> : null}
      <div className="wizard-actions"><span className="muted">The API returns 202 QUEUED; that is not a release decision.</span><button className="button primary" disabled={submitting || Object.values(form).some((value)=>!value)} onClick={submit}>{submitting ? "Queueing…" : "Queue run"}</button></div>
    </div></section>
  </>;
}

function RunField({label,name,value,onChange}:{label:string;name:string;value:string;onChange:(name:string,value:string)=>void}) {
  return <div className="field"><label htmlFor={name}>{label}</label><input id={name} value={value} onChange={(event)=>onChange(name,event.target.value)} /></div>;
}
