"""
Core data models for the Browser Automation Framework (BRAF).

These models define the structure for profiles, tasks, fingerprints, and compliance logging.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class ActionType(str, Enum):
    """Types of automation actions that can be performed."""
    
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    EXTRACT = "extract"
    SCROLL = "scroll"
    HOVER = "hover"
    SELECT = "select"
    UPLOAD = "upload"
    SCREENSHOT = "screenshot"


class TaskStatus(str, Enum):
    """Status values for automation tasks."""
    
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority levels for automation tasks."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BrowserFingerprint:
    """Browser fingerprint configuration for detection evasion."""
    
    user_agent: str
    screen_resolution: Tuple[int, int]
    timezone: str
    webgl_vendor: str
    webgl_renderer: str
    canvas_hash: str
    audio_context_hash: str
    fonts: List[str]
    plugins: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en-US", "en"])
    platform: str = "Win32"
    hardware_concurrency: int = 4
    device_memory: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary for serialization."""
        return {
            "user_agent": self.user_agent,
            "screen_resolution": self.screen_resolution,
            "timezone": self.timezone,
            "webgl_vendor": self.webgl_vendor,
            "webgl_renderer": self.webgl_renderer,
            "canvas_hash": self.canvas_hash,
            "audio_context_hash": self.audio_context_hash,
            "fonts": self.fonts,
            "plugins": self.plugins,
            "languages": self.languages,
            "platform": self.platform,
            "hardware_concurrency": self.hardware_concurrency,
            "device_memory": self.device_memory,
        }


@dataclass
class ProxyConfig:
    """Proxy configuration for IP rotation."""
    
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    proxy_type: str = "http"  # http, https, socks4, socks5
    
    def to_url(self) -> str:
        """Convert proxy config to URL format."""
        if self.username and self.password:
            return f"{self.proxy_type}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.proxy_type}://{self.host}:{self.port}"


@dataclass
class Profile:
    """User profile for consistent automation sessions."""
    
    id: str
    fingerprint_id: str
    proxy_config: Optional[ProxyConfig]
    created_at: datetime
    last_used: Optional[datetime] = None
    session_count: int = 0
    detection_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure ID is a valid UUID string."""
        if not self.id:
            self.id = str(uuid.uuid4())


class BehavioralConfig(BaseModel):
    """Configuration for behavioral emulation parameters."""
    
    mouse_speed_factor: float = Field(default=1.0, ge=0.1, le=5.0)
    typing_speed_wpm: int = Field(default=60, ge=20, le=120)
    error_rate: float = Field(default=0.02, ge=0.0, le=0.1)
    delay_variance: float = Field(default=0.3, ge=0.0, le=1.0)
    human_like_pauses: bool = True


class TaskConstraints(BaseModel):
    """Constraints and limits for task execution."""
    
    max_duration: int = Field(default=300, ge=10, le=3600)  # seconds
    max_pages: int = Field(default=10, ge=1, le=100)
    respect_robots_txt: bool = True
    max_requests_per_minute: int = Field(default=10, ge=1, le=60)
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None


class AutomationAction(BaseModel):
    """Individual action within an automation task."""
    
    type: ActionType
    selector: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None
    timeout: int = Field(default=30, ge=1, le=300)
    behavioral_config: Optional[BehavioralConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('selector')
    def validate_selector_required(cls, v, values):
        """Validate that selector is provided for actions that require it."""
        action_type = values.get('type')
        if action_type in [ActionType.CLICK, ActionType.TYPE, ActionType.EXTRACT, 
                          ActionType.HOVER, ActionType.SELECT]:
            if not v:
                raise ValueError(f"Selector is required for {action_type} actions")
        return v
    
    @validator('url')
    def validate_url_required(cls, v, values):
        """Validate that URL is provided for navigate actions."""
        if values.get('type') == ActionType.NAVIGATE and not v:
            raise ValueError("URL is required for navigate actions")
        return v


class AutomationTask(BaseModel):
    """Complete automation task with actions and configuration."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str
    target_url: Optional[str] = None
    actions: List[AutomationAction]
    constraints: Optional[TaskConstraints] = None
    priority: Optional[TaskPriority] = TaskPriority.NORMAL
    timeout: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('actions')
    def validate_actions_not_empty(cls, v):
        """Ensure task has at least one action."""
        if not v:
            raise ValueError("Task must have at least one action")
        return v


@dataclass
class ComplianceLog:
    """Log entry for compliance and audit tracking."""
    
    id: str
    timestamp: datetime
    action_type: str
    target_url: Optional[str]
    profile_id: str
    worker_id: str
    detection_score: float
    ethical_check_passed: bool
    authorization_token: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure ID is a valid UUID string."""
        if not self.id:
            self.id = str(uuid.uuid4())


class WorkerStatus(BaseModel):
    """Status information for worker nodes."""
    
    worker_id: str
    status: str  # online, offline, busy, error
    current_tasks: int = 0
    max_tasks: int = 5
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    version: str = "0.1.0"
    capabilities: List[str] = Field(default_factory=list)


class QueueMetrics(BaseModel):
    """Metrics for task queue monitoring."""
    
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    queue_length: int = 0
    worker_count: int = 0
    active_workers: int = 0


class AnalyticsReport(BaseModel):
    """Analytics report for system performance."""
    
    time_range_start: datetime
    time_range_end: datetime
    total_tasks: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    detection_rate: float = 0.0
    top_domains: List[Tuple[str, int]] = Field(default_factory=list)
    worker_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    compliance_violations: int = 0


class DetectionScore(BaseModel):
    """Detection risk assessment for automation sessions."""
    
    score: float = Field(ge=0.0, le=1.0)
    factors: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recommendations: List[str] = Field(default_factory=list)
    
    @property
    def risk_level(self) -> str:
        """Get human-readable risk level."""
        if self.score < 0.3:
            return "low"
        elif self.score < 0.7:
            return "medium"
        else:
            return "high"


# Type aliases for common data structures
Point = Tuple[float, float]
TimeRange = Tuple[datetime, datetime]
TaskID = str
ProfileID = str
WorkerID = str


@dataclass
class TaskResult:
    """Result of task execution."""
    
    task_id: str
    success: bool
    execution_time: float
    actions_completed: int
    error: Optional[str] = None
    action_results: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ViolationType(str, Enum):
    """Types of compliance violations."""
    
    RATE_LIMIT = "rate_limit"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    POLICY_VIOLATION = "policy_violation"
    DETECTION_TRIGGERED = "detection_triggered"


class SeverityLevel(str, Enum):
    """Severity levels for violations and events."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""
    
    id: str
    type: ViolationType
    severity: SeverityLevel
    description: str
    profile_id: Optional[str]
    task_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    metadata: Optional[Dict[str, Any]] = None