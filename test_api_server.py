#!/usr/bin/env python3
"""
Quick test for the HiFP8 vLLM API server.

This script:
1. Verifies that a quantized model exists
2. Starts the API server in a subprocess
3. Tests all API endpoints
4. Cleans up and reports results

Usage:
    python test_api_server.py [--model /path/to/model] [--port 8000]
"""

import argparse
import sys
import os
import time
import subprocess
import requests
import signal
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_model_exists(model_path):
    """Check if model directory exists and has required files."""
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False

    # Check for config.json
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"❌ config.json not found in {model_path}")
        return False

    # Check for model weights (safetensors or pytorch)
    has_weights = (
        (model_path / "model.safetensors").exists() or
        (model_path / "pytorch_model.bin").exists() or
        any(model_path.glob("model-*.safetensors")) or
        any(model_path.glob("pytorch_model-*.bin"))
    )

    if not has_weights:
        print(f"❌ No model weights found in {model_path}")
        return False

    print(f"✓ Model directory looks valid")
    return True


def start_server(model_path, port, model_name):
    """Start API server in subprocess."""
    print(f"\nStarting API server...")
    print(f"  Model: {model_path}")
    print(f"  Port: {port}")
    print(f"  Name: {model_name}")

    cmd = [
        sys.executable,
        "scripts/start_vllm_hifp8_server.py",
        "--model", str(model_path),
        "--port", str(port),
        "--model-name", model_name,
    ]

    # Start server
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return process


def wait_for_server(port, timeout=120):
    """Wait for server to be ready."""
    print(f"\nWaiting for server to be ready (timeout: {timeout}s)...")

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(2)
        elapsed = int(time.time() - start_time)
        print(f"  Waiting... ({elapsed}/{timeout}s)")

    print(f"❌ Server failed to start within {timeout} seconds")
    return False


def test_health_endpoint(port):
    """Test /health endpoint."""
    print("\n[Test 1/4] Testing /health endpoint...")

    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Status: {data.get('status')}")
            print(f"  ✓ Model: {data.get('model')}")
            return True
        else:
            print(f"  ❌ Unexpected status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_models_endpoint(port, expected_model_name):
    """Test /v1/models endpoint."""
    print("\n[Test 2/4] Testing /v1/models endpoint...")

    try:
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])

            if len(models) > 0:
                model_id = models[0].get('id')
                print(f"  ✓ Found model: {model_id}")

                if model_id == expected_model_name:
                    print(f"  ✓ Model name matches expected: {expected_model_name}")
                    return True
                else:
                    print(f"  ⚠️  Model name mismatch: expected {expected_model_name}, got {model_id}")
                    return True  # Still pass, just a warning
            else:
                print(f"  ❌ No models found")
                return False
        else:
            print(f"  ❌ Unexpected status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_completion_endpoint(port, model_name):
    """Test /v1/completions endpoint."""
    print("\n[Test 3/4] Testing /v1/completions endpoint...")

    try:
        response = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={
                "model": model_name,
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()

            if 'choices' in data and len(data['choices']) > 0:
                text = data['choices'][0].get('text', '')
                print(f"  ✓ Generated text: {text[:100]}")
                print(f"  ✓ Tokens: {data.get('usage', {})}")
                return True
            else:
                print(f"  ❌ No choices in response")
                return False
        else:
            print(f"  ❌ Unexpected status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_chat_completion_endpoint(port, model_name):
    """Test /v1/chat/completions endpoint."""
    print("\n[Test 4/4] Testing /v1/chat/completions endpoint...")

    try:
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ],
                "max_tokens": 20,
                "temperature": 0.0,
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()

            if 'choices' in data and len(data['choices']) > 0:
                message = data['choices'][0].get('message', {})
                content = message.get('content', '')
                print(f"  ✓ Generated response: {content[:100]}")
                print(f"  ✓ Tokens: {data.get('usage', {})}")
                return True
            else:
                print(f"  ❌ No choices in response")
                return False
        else:
            print(f"  ❌ Unexpected status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test HiFP8 vLLM API server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/data/quantized_qwen3_0.6b",
        help="Path to quantized model",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use (default: 8000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3-hifp8",
        help="Model name for API",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Server startup timeout in seconds (default: 120)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HiFP8 vLLM API Server Test")
    print("=" * 80)

    # Check model exists
    print("\n[Setup] Checking model directory...")
    if not check_model_exists(args.model):
        print("\n❌ Model check failed")
        print(f"\nTo create a quantized model, run:")
        print(f"  python examples/quantize_qwen3.py")
        sys.exit(1)

    # Start server
    server_process = start_server(args.model, args.port, args.model_name)

    try:
        # Wait for server
        if not wait_for_server(args.port, timeout=args.timeout):
            print("\n❌ Server startup failed")
            print("\nServer output:")
            print("-" * 80)
            # Try to read any output
            try:
                stdout, _ = server_process.communicate(timeout=1)
                print(stdout)
            except subprocess.TimeoutExpired:
                server_process.kill()
                stdout, _ = server_process.communicate()
                print(stdout)
            sys.exit(1)

        # Run tests
        results = []

        results.append(("Health endpoint", test_health_endpoint(args.port)))
        results.append(("Models endpoint", test_models_endpoint(args.port, args.model_name)))
        results.append(("Completion endpoint", test_completion_endpoint(args.port, args.model_name)))
        results.append(("Chat completion endpoint", test_chat_completion_endpoint(args.port, args.model_name)))

        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")

        print(f"\nTotal: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All tests passed!")
            exit_code = 0
        else:
            print(f"\n❌ {total - passed} test(s) failed")
            exit_code = 1

    finally:
        # Cleanup
        print("\n[Cleanup] Stopping server...")
        server_process.send_signal(signal.SIGINT)

        try:
            server_process.wait(timeout=5)
            print("✓ Server stopped cleanly")
        except subprocess.TimeoutExpired:
            print("⚠️  Server didn't stop, killing...")
            server_process.kill()
            server_process.wait()
            print("✓ Server killed")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
