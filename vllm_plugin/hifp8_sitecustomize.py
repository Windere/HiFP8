"""
Auto-install HiFP8 hook in all processes (including TP workers).
Activated by setting HIFP8_MODEL_PATH env var.
"""
import os

model_path = os.environ.get("HIFP8_MODEL_PATH")
if model_path:
    try:
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        ao_path = os.path.join(project_root, "ao")
        if os.path.isdir(ao_path) and ao_path not in sys.path:
            sys.path.insert(0, ao_path)

        from vllm.model_executor.model_loader import default_loader

        _original_load = default_loader.DefaultModelLoader.load_model

        def _hooked_load(self, *args, **kwargs):
            model = _original_load(self, *args, **kwargs)
            try:
                from vllm_plugin.hifp8_vllm_patcher import (
                    patch_vllm_linear_layers,
                    print_hifp8_vllm_integration_summary,
                )
                print_hifp8_vllm_integration_summary(model_path)
                model = patch_vllm_linear_layers(model, model_path)
                print("[HiFP8] Patching complete (via sitecustomize)")
            except Exception as e:
                print(f"[HiFP8] Patching failed: {e}")
                import traceback
                traceback.print_exc()
            return model

        default_loader.DefaultModelLoader.load_model = _hooked_load
    except Exception:
        pass  # silently skip if vllm not available yet
