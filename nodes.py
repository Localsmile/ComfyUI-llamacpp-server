import os
import re
import json
import threading
import base64
import io
import requests
import numpy as np
import PIL.Image
from aiohttp import web
from server import PromptServer

# Keep Alive Manager to handle background model unloading
class KeepAliveManager:
    _timers = {} # base_url -> timer
    _lock = threading.Lock()

    @classmethod
    def schedule_unload(cls, base_url, model_name, keep_alive_seconds):
        with cls._lock:
            if base_url in cls._timers:
                cls._timers[base_url].cancel()
                del cls._timers[base_url]
            
            if keep_alive_seconds == 0:
                threading.Thread(target=cls._unload, args=(base_url, model_name), daemon=True).start()
            else:
                t = threading.Timer(keep_alive_seconds, cls._unload, args=(base_url, model_name))
                cls._timers[base_url] = t
                t.start()

    @classmethod
    def cancel_unload(cls, base_url):
        with cls._lock:
            if base_url in cls._timers:
                cls._timers[base_url].cancel()
                del cls._timers[base_url]

    @classmethod
    def _unload(cls, base_url, model_name):
        try:
            payload = {"model": model_name}
            headers = {"Content-Type": "application/json"}
            
            # Try POST /model/unload
            try:
                r = requests.post(f"{base_url}/model/unload", json=payload, headers=headers, timeout=5)
                if r.status_code == 200:
                    print(f"[Llama.cpp Node] Successfully unloaded model '{model_name}' via /model/unload")
                    return
            except Exception:
                pass
                
            # Try POST /models/unload
            try:
                r = requests.post(f"{base_url}/models/unload", json=payload, headers=headers, timeout=5)
                if r.status_code == 200:
                    print(f"[Llama.cpp Node] Successfully unloaded model '{model_name}' via /models/unload")
                    return
            except Exception:
                pass
                
            print(f"[Llama.cpp Node] Failed to unload model '{model_name}' (endpoints did not return success)")
        except Exception as e:
            print(f"[Llama.cpp Node] Exception while unloading model: {e}")

# Register custom API route in ComfyUI backend for model synchronization
@PromptServer.instance.routes.post("/llama-cpp/models")
async def get_models(request):
    try:
        data = await request.json()
        url = data.get("url", "").rstrip('/')
        if not url:
            return web.json_response({"error": "No URL provided"}, status=400)
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            models = []
            active_model = None
            
            # 1. Try /v1/models (standard OpenAI-like endpoint)
            try:
                async with session.get(f"{url}/v1/models", timeout=5) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        if "data" in res_json:
                            models = [m["id"] for m in res_json["data"]]
                            if models:
                                active_model = models[0]
            except Exception:
                pass
                
            # 2. Try /models (custom llama.cpp router endpoint)
            if not models:
                try:
                    async with session.get(f"{url}/models", timeout=5) as resp:
                        if resp.status == 200:
                            res_json = await resp.json()
                            if isinstance(res_json, list):
                                models = [m.get("id") or m.get("model") or m.get("name") for m in res_json if m.get("id") or m.get("model") or m.get("name")]
                            elif isinstance(res_json, dict) and "models" in res_json:
                                models = [m.get("id") or m.get("model") for m in res_json["models"]]
                except Exception:
                    pass
            
            # 3. Try /slots
            if not models:
                try:
                    async with session.get(f"{url}/slots", timeout=5) as resp:
                        if resp.status == 200:
                            slots = await resp.json()
                            if isinstance(slots, list) and len(slots) > 0:
                                active_model = slots[0].get("model")
                                if active_model:
                                    models = [active_model]
                except Exception:
                    pass
            
            if not models and active_model:
                models = [active_model]
                
            return web.json_response({
                "models": models,
                "active_model": active_model
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

class LlamaCppServerNode:
    sessions = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "model": (["Auto (Detect Active Model)"],),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant."}),
                "user_request": ("STRING", {"multiline": True, "default": "Hello!"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "keep_alive": ("INT", {"default": 0, "min": 0, "max": 43200, "step": 1}),
                "keep_alive_unit": (["minutes", "hours"],),
                "reset_session": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "images": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000, "step": 1}),
                "max_tokens": ("INT", {"default": 2048, "min": 0, "max": 32768, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "thinking")
    FUNCTION = "process"
    CATEGORY = "Llama-cpp"

    def process(self, url, model, system_prompt, user_request, seed, keep_alive, keep_alive_unit, reset_session, 
                images=None, temperature=1.0, top_p=0.95, top_k=64, max_tokens=2048, repeat_penalty=1.0, unique_id=None):
        base_url = url.rstrip('/')
        KeepAliveManager.cancel_unload(base_url)
        
        if reset_session or unique_id not in self.sessions:
            self.sessions[unique_id] = []
            
        session_history = self.sessions[unique_id]
        
        if not session_history and system_prompt.strip():
            session_history.append({"role": "system", "content": system_prompt})
            
        content_list = []
        if images is not None:
            print(f"[Llama.cpp Node] Image tensor received. Shape: {images.shape}")
            for i in range(images.shape[0]):
                img_tensor = images[i]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img_pil = PIL.Image.fromarray(img_np)
                
                buffered = io.BytesIO()
                img_pil.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
        else:
            print("[Llama.cpp Node] No image tensor connected (None).")
        
        content_list.append({
            "type": "text",
            "text": user_request
        })
        
        session_history.append({
            "role": "user",
            "content": content_list if len(content_list) > 1 else user_request
        })
        
        # Resolve active model (fallback to querying server if default/error/Auto values are set)
        active_model = model
        if not active_model or any(k in active_model for k in ["Auto", "None", "Error", "Connecting"]):
            try:
                r = requests.get(f"{base_url}/v1/models", timeout=5)
                if r.status_code == 200:
                    r_json = r.json()
                    if "data" in r_json and len(r_json["data"]) > 0:
                        active_model = r_json["data"][0]["id"]
            except Exception:
                pass
            
            if not active_model or any(k in active_model for k in ["Auto", "None", "Error", "Connecting"]):
                try:
                    r = requests.get(f"{base_url}/slots", timeout=5)
                    if r.status_code == 200:
                        slots = r.json()
                        if isinstance(slots, list) and len(slots) > 0:
                            active_model = slots[0].get("model")
                except Exception:
                    pass

        payload = {
            "messages": session_history,
            "stream": False,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "repeat_penalty": repeat_penalty
        }
        
        if active_model and not any(k in active_model for k in ["Auto", "None", "Error", "Connecting"]):
            payload["model"] = active_model
            
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            res_json = response.json()
            
            choice = res_json["choices"][0]["message"]
            assistant_content = choice.get("content", "")
            
            session_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            thinking = ""
            result = assistant_content
            
            if "reasoning_content" in choice:
                thinking = choice["reasoning_content"]
            elif "<think>" in assistant_content:
                think_match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    result = re.sub(r'<think>.*?</think>', '', assistant_content, flags=re.DOTALL).strip()
            
        except Exception as e:
            if session_history and session_history[-1]["role"] == "user":
                session_history.pop()
            return (f"Error: {str(e)}", "")
            
        keep_alive_seconds = keep_alive
        if keep_alive_unit == "hours":
            keep_alive_seconds = keep_alive * 3600
        elif keep_alive_unit == "minutes":
            keep_alive_seconds = keep_alive * 60
            
        if not active_model or any(k in active_model for k in ["Auto", "None", "Error", "Connecting"]):
            active_model = res_json.get("model", "")
            
        if active_model and not any(k in active_model for k in ["Auto", "None", "Error", "Connecting"]):
            KeepAliveManager.schedule_unload(base_url, active_model, keep_alive_seconds)
            
        return (result, thinking)

NODE_CLASS_MAPPINGS = {
    "LlamaCppServerNode": LlamaCppServerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppServerNode": "Llama.cpp Server Node"
}
