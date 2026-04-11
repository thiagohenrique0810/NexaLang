"""TurboServe — FastAPI HTTP server for TurboLang LLM Runtime."""

import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from turboserve.schemas.models import (
    HealthResponse, LoadModelRequest, LoadModelResponse,
    CompileRequest, CompileResponse,
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse,
    MetricsResponse, ErrorResponse,
)
from turboruntime.core.engine import InferenceEngine, EngineConfig

logging.basicConfig(
    level=os.environ.get("TURBO_LOG_LEVEL", "info").upper(),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("turboserve")

engine = InferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_model = os.environ.get("TURBO_DEFAULT_MODEL")
    if default_model:
        logger.info(f"Auto-loading model: {default_model}")
        try:
            device = os.environ.get("TURBO_DEVICE", "auto")
            if device == "auto":
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                except ImportError:
                    device = "cpu"
            config = EngineConfig(model_name=default_model, device=device)
            engine.init(config)
        except Exception as e:
            logger.error(f"Failed to auto-load model: {e}")
    yield
    engine.shutdown()


app = FastAPI(
    title="TurboServe",
    description="TurboLang LLM Runtime — High-performance inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=engine.ready,
        model_name=engine.config.model_name if engine.config else None,
        device=engine.config.device if engine.config else None,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    return MetricsResponse(**engine.get_metrics())


@app.post("/load-model", response_model=LoadModelResponse)
async def load_model(req: LoadModelRequest):
    try:
        device = req.device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        config = EngineConfig(
            model_name=req.model_name,
            device=device,
            quant_type=req.quant_type,
            cache_strategy=req.cache_strategy,
            cache_bits=req.cache_bits,
            max_ctx=req.max_ctx,
            max_batch=req.max_batch,
        )
        engine.init(config)

        param_count = sum(p.numel() for p in engine.model.model.parameters())
        return LoadModelResponse(
            status="ok",
            model_name=req.model_name,
            device=device,
            parameters=f"{param_count/1e6:.1f}M",
            message=f"Model loaded successfully on {device}",
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compile", response_model=CompileResponse)
async def compile_tl(req: CompileRequest):
    try:
        from turbolang.parser.lexer import Lexer
        from turbolang.parser.parser import Parser
        from turbolang.validator.validator import Validator
        from turboir.ir.nodes import IRGenerator
        from turboir.planner.planner import ExecutionPlanner

        tokens = Lexer(req.source).tokenize()
        ast = Parser(tokens).parse()
        validator = Validator()
        warnings = validator.validate(ast)
        ir_program = IRGenerator().generate(ast)
        plans = ExecutionPlanner().plan(ir_program)

        return CompileResponse(
            status="ok",
            plan={"steps": [
                {"name": s.name, "action": s.action, "params": s.params}
                for s in plans[0].steps
            ]} if plans else {},
            ir={
                "models": [m.to_dict() for m in ir_program.models]
            },
            warnings=[str(w) for w in warnings],
        )
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="No model loaded. POST /load-model first.")

    if req.stream:
        async def stream_tokens():
            for token_text in engine.generate_stream(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            ):
                yield f"data: {token_text}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    try:
        result = engine.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )
        return GenerateResponse(
            text=result.text,
            tokens_generated=result.tokens_generated,
            prompt_tokens=result.prompt_tokens,
            ttft_ms=result.ttft_ms,
            total_ms=result.total_ms,
            tokens_per_sec=result.tokens_per_sec,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="No model loaded. POST /load-model first.")

    # Format messages into a prompt
    prompt_parts = []
    for msg in req.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    try:
        result = engine.generate(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )
        return ChatResponse(
            role="assistant",
            content=result.text,
            tokens_generated=result.tokens_generated,
            ttft_ms=result.ttft_ms,
            total_ms=result.total_ms,
            tokens_per_sec=result.tokens_per_sec,
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TURBO_HOST", "0.0.0.0")
    port = int(os.environ.get("TURBO_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
