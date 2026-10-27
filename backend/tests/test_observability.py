import pytest

from backend.app.observability import MetricName, OperationalMetrics


def test_metrics_accept_operational_labels_and_reject_identifiers_or_phi() -> None:
    metrics = OperationalMetrics()
    metrics.increment(
        MetricName.EVALUATIONS,
        execution_status="COMPLETED",
        test_family="clinical_faithfulness",
    )
    metrics.observe(MetricName.EVALUATION_LATENCY, 0.5, outcome="success")

    snapshot = metrics.snapshot()

    assert snapshot["counters"]
    assert snapshot["observations"]
    with pytest.raises(ValueError):
        metrics.increment(MetricName.EVALUATIONS, patient_id="patient-1")
    with pytest.raises(ValueError):
        metrics.increment(MetricName.EVALUATIONS, clinical_content="knee pain")
