import time
import json
import os
import hashlib
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TRACE_FILES = 1000
TRACE_LOG_DIR = "logs/traces"


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------
def _hash_query(query: str) -> str:
    """Return a truncated SHA-256 of the query for safe logging."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def _redact_paths(params: Dict[str, Any]) -> Dict[str, Any]:
    """Replace absolute file paths with basenames + hash suffix."""
    redacted = {}
    for k, v in params.items():
        if isinstance(v, str) and os.path.isabs(v):
            basename = os.path.basename(v)
            short_hash = hashlib.sha256(v.encode()).hexdigest()[:8]
            redacted[k] = f"{basename} (hash:{short_hash})"
        elif isinstance(v, list):
            redacted[k] = [_redact_paths({"_": item})["_"] for item in v]
        else:
            redacted[k] = v
    return redacted


# ---------------------------------------------------------------------------
# Trace rotation
# ---------------------------------------------------------------------------
_trace_lock = threading.Lock()


def _rotate_traces(log_dir: str) -> None:
    """Delete oldest traces until under MAX_TRACE_FILES limit."""
    with _trace_lock:
        try:
            files = sorted(
                (os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".json")),
                key=os.path.getmtime,
            )
            while len(files) >= MAX_TRACE_FILES:
                oldest = files.pop(0)
                try:
                    os.remove(oldest)
                except OSError:
                    pass
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ExecutionStep:
    step_name: str
    tool_name: str
    parameters: Dict[str, Any]
    start_time: str
    end_time: str
    duration_ms: float
    status: str
    confidence: Optional[float] = None
    output_summary: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    query_hash: str                      # redacted: SHA-256 of query
    task_mode: str
    start_time: str
    end_time: Optional[str] = None
    total_duration_ms: float = 0.0
    steps: List[ExecutionStep] = field(default_factory=list)
    selected_tool: Optional[str] = None
    final_status: str = "pending"
    confidence_score: Optional[float] = None

    def add_step(self, step: ExecutionStep):
        self.steps.append(step)

    def finalize(self, status: str = "completed", confidence: Optional[float] = None):
        self.end_time = datetime.now().isoformat()
        self.final_status = status
        self.confidence_score = confidence
        if self.steps:
            self.total_duration_ms = sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "task_mode": self.task_mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "selected_tool": self.selected_tool,
            "final_status": self.final_status,
            "confidence_score": self.confidence_score,
            "steps": [asdict(s) for s in self.steps],
        }

    def to_summary(self) -> str:
        lines = [
            f"**Execution Trace Summary**",
            f"- **Query Hash:** {self.query_hash}",
            f"- **Task Mode:** {self.task_mode}",
            f"- **Selected Tool:** {self.selected_tool or 'N/A'}",
            f"- **Status:** {self.final_status}",
            f"- **Total Duration:** {self.total_duration_ms:.1f}ms",
        ]
        if self.confidence_score is not None:
            lines.append(f"- **Confidence:** {self.confidence_score:.1%}")
        if self.steps:
            lines.append(f"- **Steps Executed:** {len(self.steps)}")
            for i, step in enumerate(self.steps, 1):
                lines.append(f"  {i}. {step.step_name} ({step.duration_ms:.1f}ms) - {step.status}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TraceLogger
# ---------------------------------------------------------------------------
class TraceLogger:
    def __init__(self, log_dir: str = TRACE_LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._current_trace: Optional[ExecutionTrace] = None

    def start_trace(self, query: str, task_mode: str) -> ExecutionTrace:
        with self._lock:
            trace = ExecutionTrace(
                query_hash=_hash_query(query),
                task_mode=task_mode,
                start_time=datetime.now().isoformat(),
            )
            self._current_trace = trace
            return trace

    def log_step(
        self,
        step_name: str,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str = "completed",
        confidence: Optional[float] = None,
        output_summary: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ExecutionStep:
        with self._lock:
            if self._current_trace is None:
                raise RuntimeError("No active trace. Call start_trace() first.")

            t0 = time.perf_counter()
            step_start_iso = datetime.now().isoformat()
            # Redact absolute paths before storing
            safe_params = _redact_paths(dict(parameters))
            step_end_iso = datetime.now().isoformat()
            step_duration_ms = (time.perf_counter() - t0) * 1000
            step = ExecutionStep(
                step_name=step_name,
                tool_name=tool_name,
                parameters=safe_params,
                start_time=step_start_iso,
                end_time=step_end_iso,
                duration_ms=step_duration_ms,
                status=status,
                confidence=confidence,
                output_summary=output_summary,
                error=error,
            )
            self._current_trace.add_step(step)
            return step

    def finalize_trace(
        self,
        status: str = "completed",
        confidence: Optional[float] = None,
    ) -> Optional[ExecutionTrace]:
        with self._lock:
            if self._current_trace is None:
                return None
            self._current_trace.finalize(status=status, confidence=confidence)
            trace = self._current_trace
            self._current_trace = None
            self._save_trace(trace)
            return trace

    def _save_trace(self, trace: ExecutionTrace) -> None:
        _rotate_traces(self.log_dir)
        # UUID filename prevents collision under concurrent load
        filename = f"trace_{uuid.uuid4().hex}.json"
        filepath = os.path.join(self.log_dir, filename)
        with _trace_lock:
            with open(filepath, "w") as f:
                json.dump(trace.to_dict(), f, indent=2)

    def get_current_trace(self) -> Optional[ExecutionTrace]:
        with self._lock:
            return self._current_trace
