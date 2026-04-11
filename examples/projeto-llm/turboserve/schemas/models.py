"""Pydantic schemas for TurboServe API."""

from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    model_name: Optional[str] = None
    device: Optional[str] = None


class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="HuggingFace model name or path")
    device: str = Field("auto", description="Device: auto, cpu, cuda, mps")
    quant_type: str = Field("none", description="Quantization: none, int8, int4")
    cache_strategy: str = Field("standard", description="KV cache: standard, turboquant, none")
    cache_bits: int = Field(16, description="Cache compression bits (3-16)")
    max_ctx: int = Field(2048, description="Maximum context length")
    max_batch: int = Field(8, description="Maximum batch size")


class LoadModelResponse(BaseModel):
    status: str = "ok"
    model_name: str = ""
    device: str = ""
    parameters: str = ""
    message: str = ""


class CompileRequest(BaseModel):
    source: str = Field(..., description="TurboLang .tl source code")


class CompileResponse(BaseModel):
    status: str = "ok"
    plan: dict = {}
    ir: dict = {}
    warnings: list[str] = []


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(128, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling p")
    top_k: int = Field(0, ge=0, description="Top-k sampling (0=disabled)")
    stream: bool = Field(False, description="Stream tokens via SSE")


class GenerateResponse(BaseModel):
    text: str = ""
    tokens_generated: int = 0
    prompt_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_sec: float = 0.0


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(0, ge=0)


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str = ""
    tokens_generated: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_sec: float = 0.0


class MetricsResponse(BaseModel):
    ready: bool = False
    model: Optional[str] = None
    device: Optional[str] = None
    kv_cache: Optional[dict] = None
    scheduler: Optional[dict] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str = ""
