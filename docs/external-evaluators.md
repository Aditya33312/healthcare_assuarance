# MedHELM and LangTest integration

Gatekeeper keeps external evaluation engines behind a small JSON runner
boundary.  Without a configured provider runner, Gatekeeper uses its local
deterministic contract evaluator; it does not claim that a provider benchmark
ran.

## Supported families

| Engine | Gatekeeper families |
| --- | --- |
| MedHELM | `clinical_note_generation`, `clinical_task_performance`, `medication_safety` |
| LangTest | `robustness`, `demographic_bias`, `prompt_injection_jailbreak` |

`clinical_faithfulness` continues to use Gatekeeper's local `clinical_compare`
evaluator.  The mapping is deliberately explicit and can be expanded after a
provider-specific validation. All other families have a local contract
evaluator that requires explicit assertions in the test case.

## Optional dependencies

Install the lightweight LangTest integration in the Python 3.12+ environment:

```bash
python -m pip install -e ".[evaluation]"
```

MedHELM has a heavier PyTorch dependency. Install it separately in a
Python/torch-compatible environment:

```bash
python -m pip install -e ".[medhelm]"
```

On Intel macOS with Python 3.13, PyTorch/MedHELM may not have a compatible
wheel. In that case run the MedHELM wrapper in a separate Python 3.12 or Linux
worker and keep the Gatekeeper JSON runner command pointed at that worker.

MedHELM and LangTest do not expose one identical API for every application
endpoint.  A small wrapper should translate the Gatekeeper payload to the
provider's benchmark/harness API and emit one JSON result on stdout.

Input on stdin:

```json
{
  "case_id": "case-1",
  "test_family": "clinical_task_performance",
  "input_data": {"question": "..."},
  "generated_output": "...",
  "evaluator_context": {"reference": "..."}
}
```

Output on stdout:

```json
{
  "score": 92.0,
  "passed": true,
  "metrics": {"accuracy": 0.92},
  "findings": []
}
```

Configure the wrappers as commands.  Gatekeeper passes JSON on stdin and never
uses a shell:

```bash
export GATEKEEPER_MEDHELM_COMMAND="python scripts/medhelm_gatekeeper_runner.py"
export GATEKEEPER_LANGTEST_COMMAND="python scripts/langtest_gatekeeper_runner.py"
```

The command must return full coverage.  A missing command, timeout, non-zero
exit, invalid JSON, or reduced coverage is recorded as an evaluator failure or
`UNASSESSED`; it cannot become a successful assessment.
