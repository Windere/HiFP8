#!/usr/bin/env python3
"""
Start vLLM OpenAI-compatible API server with HiFP8 fake quantization.

This script:
1. Loads a BF16 model exported by export_bf16_for_vllm
2. Applies HiFP8 fake quantization via our plugin
3. Starts vLLM OpenAI-compatible API server
4. Compatible with evalscope and other OpenAI API clients

Usage:
    python scripts/start_vllm_hifp8_server.py \\
        --model /home/data/quantized_qwen3_30b_moe \\
        --host 0.0.0.0 \\
        --port 8000
"""

import argparse
import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ao"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_plugin import apply_hifp8_fake_quant_to_vllm_model


def start_server(args):
    """Start vLLM API server with HiFP8 quantization."""

    print("="*80)
    print("Starting vLLM API Server with HiFP8 Fake Quantization")
    print("="*80)

    # 1. Load model
    print(f"\n[1/3] Loading model from: {args.model}")
    print("   This may take a while for large models...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model loaded ({total_params / 1e9:.2f}B parameters)")

    # 2. Apply HiFP8 quantization
    print(f"\n[2/3] Applying HiFP8 fake quantization...")
    apply_hifp8_fake_quant_to_vllm_model(model, args.model)

    from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
    num_quantized = sum(1 for m in model.modules()
                       if isinstance(m, HiFP8FakeQuantizedLinear))
    print(f"   ✓ Applied HiFP8 quantization to {num_quantized} layers")

    # 3. Start API server using uvicorn + FastAPI
    print(f"\n[3/3] Starting API server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   API endpoint: http://{args.host}:{args.port}/v1")

    # Create FastAPI app
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List, Optional
    import uvicorn
    import time

    app = FastAPI(title="HiFP8 vLLM Server")

    # Store model and tokenizer in app state
    app.state.model = model
    app.state.tokenizer = tokenizer

    # OpenAI-compatible API schemas
    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: float = 0.7
        max_tokens: Optional[int] = 512
        top_p: float = 1.0
        stream: bool = False

    class CompletionRequest(BaseModel):
        model: str
        prompt: str
        temperature: float = 0.7
        max_tokens: Optional[int] = 512
        top_p: float = 1.0

    # API endpoints
    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI compatible)."""
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hifp8",
                }
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completion endpoint (OpenAI compatible)."""
        try:
            # Convert messages to proper format using chat template
            # This is CRITICAL for model performance - each model has its own chat format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            # Use tokenizer's chat template (e.g., ChatML for Qwen3)
            # This ensures proper formatting with special tokens like <|im_start|>, <|im_end|>
            if hasattr(app.state.tokenizer, 'apply_chat_template'):
                prompt = app.state.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        prompt += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n"
                prompt += "Assistant:"

            # Generate
            inputs = app.state.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = app.state.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                )

            response_text = app.state.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                    "total_tokens": outputs.shape[1],
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """Text completion endpoint (OpenAI compatible)."""
        try:
            inputs = app.state.tokenizer(request.prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = app.state.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                )

            response_text = app.state.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "text": response_text,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                    "total_tokens": outputs.shape[1],
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "model": args.model_name}

    # Start server
    print(f"\n{'='*80}")
    print(f"✅ Server ready!")
    print(f"{'='*80}")
    print(f"")
    print(f"API Base URL: http://{args.host}:{args.port}/v1")
    print(f"")
    print(f"Test with curl:")
    print(f'  curl http://{args.host}:{args.port}/v1/models')
    print(f"")
    print(f"Use with evalscope:")
    print(f'  evalscope eval --model {args.model_name} \\')
    print(f'    --api-base http://{args.host}:{args.port}/v1 \\')
    print(f'    --datasets mmlu ceval')
    print(f"")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*80}\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM API server with HiFP8 quantization"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to HiFP8-quantized model (exported by export_bf16_for_vllm)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for API (default: use --model path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )

    args = parser.parse_args()

    # Set default model name
    if args.model_name is None:
        args.model_name = Path(args.model).name

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    try:
        start_server(args)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
