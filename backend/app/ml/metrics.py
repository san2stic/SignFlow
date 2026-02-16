"""Inference/runtime metrics with Prometheus-compatible export."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict

import structlog

logger = structlog.get_logger(__name__)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except Exception:  # noqa: BLE001
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    PROMETHEUS_AVAILABLE = False


class InferenceMetricsCollector:
    """Collect and expose inference metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._fallback_inference_total: DefaultDict[tuple[str, str], int] = defaultdict(int)
        self._fallback_latency_sum: DefaultDict[str, float] = defaultdict(float)
        self._fallback_latency_count: DefaultDict[str, int] = defaultdict(int)
        self._fallback_confidence: DefaultDict[tuple[str, str], float] = defaultdict(float)
        self._fallback_torchserve_errors: DefaultDict[str, int] = defaultdict(int)
        self._fallback_drift_alerts: DefaultDict[str, int] = defaultdict(int)
        self._fallback_routing_total: DefaultDict[str, int] = defaultdict(int)
        self._fallback_shadow_total: DefaultDict[tuple[str, str], int] = defaultdict(int)

        self._registry = None
        self._inference_total = None
        self._inference_latency = None
        self._prediction_confidence = None
        self._torchserve_errors = None
        self._drift_alerts = None
        self._routing_total = None
        self._shadow_total = None

        if PROMETHEUS_AVAILABLE:
            registry = CollectorRegistry()
            self._registry = registry
            self._inference_total = Counter(
                "signflow_inference_total",
                "Total number of completed inferences.",
                labelnames=("model_version", "sign"),
                registry=registry,
            )
            self._inference_latency = Histogram(
                "signflow_inference_latency_seconds",
                "Inference latency in seconds.",
                labelnames=("model_version",),
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
                registry=registry,
            )
            self._prediction_confidence = Gauge(
                "signflow_prediction_confidence",
                "Latest prediction confidence by sign.",
                labelnames=("model_version", "sign"),
                registry=registry,
            )
            self._torchserve_errors = Counter(
                "signflow_torchserve_errors_total",
                "TorchServe errors observed by fallback path.",
                labelnames=("reason",),
                registry=registry,
            )
            self._drift_alerts = Counter(
                "signflow_drift_alerts_total",
                "Detected confidence-drift alerts.",
                labelnames=("kind",),
                registry=registry,
            )
            self._routing_total = Counter(
                "signflow_routing_total",
                "Total websocket sessions routed by traffic policy.",
                labelnames=("route",),
                registry=registry,
            )
            self._shadow_total = Counter(
                "signflow_shadow_comparisons_total",
                "Shadow comparison outcomes.",
                labelnames=("route", "result"),
                registry=registry,
            )

    @staticmethod
    def _normalize_model_version(model_version: str | None) -> str:
        value = str(model_version or "unknown").strip()
        return value or "unknown"

    @staticmethod
    def _normalize_sign(sign: str | None) -> str:
        value = str(sign or "NONE").strip()
        return value or "NONE"

    def record_inference(
        self,
        *,
        model_version: str | None,
        sign: str | None,
        confidence: float,
        latency_seconds: float,
    ) -> None:
        """Record one completed inference event."""
        model = self._normalize_model_version(model_version)
        label = self._normalize_sign(sign)
        latency = max(0.0, float(latency_seconds))
        conf = max(0.0, min(1.0, float(confidence)))

        if PROMETHEUS_AVAILABLE and self._inference_total is not None:
            self._inference_total.labels(model_version=model, sign=label).inc()
            self._inference_latency.labels(model_version=model).observe(latency)
            self._prediction_confidence.labels(model_version=model, sign=label).set(conf)
            return

        with self._lock:
            self._fallback_inference_total[(model, label)] += 1
            self._fallback_latency_sum[model] += latency
            self._fallback_latency_count[model] += 1
            self._fallback_confidence[(model, label)] = conf

    def record_torchserve_error(self, *, reason: str) -> None:
        """Record one TorchServe error event."""
        normalized_reason = str(reason or "unknown").strip() or "unknown"
        if PROMETHEUS_AVAILABLE and self._torchserve_errors is not None:
            self._torchserve_errors.labels(reason=normalized_reason).inc()
            return
        with self._lock:
            self._fallback_torchserve_errors[normalized_reason] += 1

    def record_drift_alert(self, *, kind: str = "confidence") -> None:
        """Record one drift alert event."""
        normalized_kind = str(kind or "confidence").strip() or "confidence"
        if PROMETHEUS_AVAILABLE and self._drift_alerts is not None:
            self._drift_alerts.labels(kind=normalized_kind).inc()
            return
        with self._lock:
            self._fallback_drift_alerts[normalized_kind] += 1

    def record_routing_decision(self, *, route: str) -> None:
        """Record one routing decision for a websocket session."""
        normalized_route = str(route or "production").strip() or "production"
        if PROMETHEUS_AVAILABLE and self._routing_total is not None:
            self._routing_total.labels(route=normalized_route).inc()
            return
        with self._lock:
            self._fallback_routing_total[normalized_route] += 1

    def record_shadow_comparison(self, *, route: str, disagreed: bool) -> None:
        """Record one shadow comparison outcome."""
        normalized_route = str(route or "production").strip() or "production"
        result = "disagree" if disagreed else "agree"
        if PROMETHEUS_AVAILABLE and self._shadow_total is not None:
            self._shadow_total.labels(route=normalized_route, result=result).inc()
            return
        with self._lock:
            self._fallback_shadow_total[(normalized_route, result)] += 1

    def render_latest(self) -> tuple[bytes, str]:
        """Render all metrics in Prometheus text exposition format."""
        if PROMETHEUS_AVAILABLE and self._registry is not None:
            return generate_latest(self._registry), CONTENT_TYPE_LATEST

        with self._lock:
            lines: list[str] = [
                "# HELP signflow_inference_total Total number of completed inferences.",
                "# TYPE signflow_inference_total counter",
            ]
            for (model_version, sign), count in self._fallback_inference_total.items():
                lines.append(
                    f'signflow_inference_total{{model_version="{model_version}",sign="{sign}"}} {count}'
                )

            lines.extend(
                [
                    "# HELP signflow_inference_latency_seconds_sum Sum of inference latency in seconds.",
                    "# TYPE signflow_inference_latency_seconds_sum gauge",
                ]
            )
            for model_version, value in self._fallback_latency_sum.items():
                lines.append(
                    f'signflow_inference_latency_seconds_sum{{model_version="{model_version}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP signflow_prediction_confidence Latest prediction confidence by sign.",
                    "# TYPE signflow_prediction_confidence gauge",
                ]
            )
            for (model_version, sign), value in self._fallback_confidence.items():
                lines.append(
                    f'signflow_prediction_confidence{{model_version="{model_version}",sign="{sign}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP signflow_torchserve_errors_total TorchServe errors observed by fallback path.",
                    "# TYPE signflow_torchserve_errors_total counter",
                ]
            )
            for reason, count in self._fallback_torchserve_errors.items():
                lines.append(f'signflow_torchserve_errors_total{{reason="{reason}"}} {count}')

            lines.extend(
                [
                    "# HELP signflow_drift_alerts_total Detected confidence-drift alerts.",
                    "# TYPE signflow_drift_alerts_total counter",
                ]
            )
            for kind, count in self._fallback_drift_alerts.items():
                lines.append(f'signflow_drift_alerts_total{{kind="{kind}"}} {count}')

            lines.extend(
                [
                    "# HELP signflow_routing_total Total websocket sessions routed by traffic policy.",
                    "# TYPE signflow_routing_total counter",
                ]
            )
            for route, count in self._fallback_routing_total.items():
                lines.append(f'signflow_routing_total{{route="{route}"}} {count}')

            lines.extend(
                [
                    "# HELP signflow_shadow_comparisons_total Shadow comparison outcomes.",
                    "# TYPE signflow_shadow_comparisons_total counter",
                ]
            )
            for (route, result), count in self._fallback_shadow_total.items():
                lines.append(
                    f'signflow_shadow_comparisons_total{{route="{route}",result="{result}"}} {count}'
                )

            payload = ("\n".join(lines) + "\n").encode("utf-8")
            return payload, CONTENT_TYPE_LATEST


_global_metrics_collector: InferenceMetricsCollector | None = None


def get_metrics_collector() -> InferenceMetricsCollector:
    """Return shared metrics collector singleton."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = InferenceMetricsCollector()
        logger.info("metrics_collector_initialized", prometheus_enabled=PROMETHEUS_AVAILABLE)
    return _global_metrics_collector
