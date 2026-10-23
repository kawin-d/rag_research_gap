import os
import re
import json
import unicodedata
import httpx
from typing import List, Dict, Any, Optional
from app.config import settings, BASE_DIR


class LLMService:
    """Unified LLM service supporting OpenAI, Anthropic, Gemini, Ollama, and an offline academic engine."""

    @classmethod
    def generate_answer(cls, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cited answer strictly grounded in retrieved chunks."""
        provider = cls._detect_provider()

        # Format retrieved context
        formatted_context = ""
        for i, chunk in enumerate(context_chunks):
            formatted_context += f"\n--- CHUNK {i+1} [Paper: {chunk.get('paper_title')}, Year: {chunk.get('year')}, Section: {chunk.get('section_name')}, ChunkID: {chunk.get('chunk_id')}] ---\n"
            formatted_context += chunk.get("text", "") + "\n"

        system_prompt = (
            "You are an expert scientific researcher. Answer the user's research question strictly using the provided context chunks.\n"
            "CRITICAL RULES:\n"
            "1. Ground every claim directly in the context. Never fabricate or hallucinate facts.\n"
            "2. For every factual claim, include an explicit citation in the format: [Paper: <Title>, Year: <Year>, Section: <Section>].\n"
            "3. If the context does not contain enough information to answer the question, state: 'The current paper collection does not provide sufficient evidence to answer this question.'\n"
            "4. Provide a synthesis comparing findings across papers where applicable."
        )

        user_prompt = f"Question: {question}\n\nContext excerpts:\n{formatted_context}\n\nSynthesized cited answer:"

        prompt_preview = f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n=== USER PROMPT (QUESTION & RETRIEVED CHUNKS) ===\n{user_prompt}"

        if provider == "gemini":
            res = cls._call_gemini(question, system_prompt, user_prompt, context_chunks)
        elif provider == "groq":
            res = cls._call_groq(question, system_prompt, user_prompt, context_chunks)
        elif provider == "openai":
            res = cls._call_openai(question, system_prompt, user_prompt, context_chunks)
        elif provider == "anthropic":
            res = cls._call_anthropic(question, system_prompt, user_prompt, context_chunks)
        elif provider == "ollama":
            res = cls._call_ollama(question, system_prompt, user_prompt, context_chunks)
        elif provider == "local":
            res = cls._call_local_flan(question, context_chunks)
        else:
            res = cls._offline_rag_engine(question, context_chunks)

        status = cls.get_status()
        res["llm_provider"] = status.get("provider", provider)
        res["llm_model"] = status.get("model", "")
        res["prompt_preview"] = prompt_preview

        if "answer" in res and isinstance(res["answer"], str):
            clean_ans = unicodedata.normalize("NFKD", res["answer"])
            clean_ans = (
                clean_ans
                .replace("\u202f", " ")
                .replace("\u00a0", " ")
                .replace("\u2011", "-")
                .replace("\u2013", "-")
                .replace("\u2014", "--")
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("\u201c", '"')
                .replace("\u201d", '"')
            )
            # Encode to ASCII, ignoring unencodable characters for maximum terminal/console safety
            res["answer"] = clean_ans.encode("ascii", "ignore").decode("ascii")
        return res

    @classmethod
    def _detect_provider(cls) -> str:
        if settings.LLM_PROVIDER != "auto":
            return settings.LLM_PROVIDER
        groq_k = (settings.GROQ_API_KEY or "").strip()
        if groq_k and "your_actual_key" not in groq_k:
            return "groq"
        if settings.GEMINI_API_KEY:
            return "gemini"
        if settings.OPENAI_API_KEY:
            return "openai"
        if settings.ANTHROPIC_API_KEY:
            return "anthropic"
        # Test if ollama is alive
        try:
            with httpx.Client(timeout=0.8) as c:
                res = c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if res.status_code == 200:
                    return "ollama"
        except Exception:
            pass
        # Default to local neural model (flan-t5) if available, otherwise mock
        return "local"

    SUPPORTED_MODELS: Dict[str, List[str]] = {
        "groq": [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        "openai": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        "gemini": [
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-pro",
            "gemini-2.5-flash"
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
        ],
        "ollama": [
            "llama3",
            "mistral",
            "qwen2.5",
            "phi3"
        ],
        "mock": [
            "offline-heuristic"
        ]
    }

    PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
        "groq": "openai/gpt-oss-120b",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-3.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3",
        "mock": "offline-heuristic"
    }

    @classmethod
    def _has_valid_key(cls, provider: str) -> bool:
        prov = provider.lower().strip()
        if prov == "groq":
            k = (settings.GROQ_API_KEY or "").strip()
            return bool(k and k.startswith("gsk_") and "your_" not in k)
        elif prov == "openai":
            k = (settings.OPENAI_API_KEY or "").strip()
            return bool(k and k.startswith("sk-") and not k.startswith("gsk_"))
        elif prov == "gemini":
            k = (settings.GEMINI_API_KEY or "").strip()
            return bool(k and not k.startswith("gsk_") and len(k) > 10)
        elif prov == "anthropic":
            k = (settings.ANTHROPIC_API_KEY or "").strip()
            return bool(k and k.startswith("sk-ant-"))
        elif prov in ("ollama", "mock", "local"):
            return True
        return False

    @classmethod
    def _get_missing_key_warning(cls, provider: str) -> Optional[str]:
        prov = provider.lower().strip()
        if prov == "groq" and not cls._has_valid_key("groq"):
            return "GROQ_API_KEY is not configured in backend/.env (must start with 'gsk_')"
        elif prov == "openai" and not cls._has_valid_key("openai"):
            return "OPENAI_API_KEY is not configured in backend/.env (must start with 'sk-')"
        elif prov == "gemini" and not cls._has_valid_key("gemini"):
            return "GEMINI_API_KEY is not configured in backend/.env (requires Google AI Studio key)"
        elif prov == "anthropic" and not cls._has_valid_key("anthropic"):
            return "ANTHROPIC_API_KEY is not configured in backend/.env (must start with 'sk-ant-')"
        return None

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Returns the currently active LLM provider and model configuration."""
        provider = cls._detect_provider()
        model_name = "offline-heuristic"
        if provider == "gemini":
            model_name = settings.GEMINI_MODEL or "gemini-1.5-flash"
        elif provider == "groq":
            model_name = settings.GROQ_MODEL or "openai/gpt-oss-120b"
        elif provider == "openai":
            model_name = settings.OPENAI_MODEL or "gpt-4o-mini"
        elif provider == "anthropic":
            model_name = settings.ANTHROPIC_MODEL or "claude-3-5-sonnet-20241022"
        elif provider == "ollama":
            model_name = settings.OLLAMA_MODEL or "llama3"
        elif provider == "local":
            model_name = "google/flan-t5-small (Local On-Device)"

        has_key = cls._has_valid_key(provider)
        missing_warning = cls._get_missing_key_warning(provider)
        is_gen = provider not in ("mock", "offline")

        keys_configured = {
            p: cls._has_valid_key(p)
            for p in cls.SUPPORTED_MODELS.keys()
        }

        return {
            "provider": provider,
            "model": model_name,
            "is_generative": is_gen,
            "offline_heuristic": provider == "mock",
            "has_key": has_key,
            "missing_key_warning": missing_warning,
            "supported_models": cls.SUPPORTED_MODELS,
            "provider_keys_configured": keys_configured,
            "status_message": (
                f"Using neural generative LLM ({provider}: {model_name})"
                if is_gen and has_key
                else (missing_warning or "Using offline keyword-extractive fallback")
            )
        }

    @classmethod
    def update_config(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Dynamically update LLM settings at runtime and persist into .env"""
        prov = provider.lower().strip()
        if prov not in cls.SUPPORTED_MODELS and prov != "local":
            prov = "mock"

        settings.LLM_PROVIDER = prov
        env_updates = {"LLM_PROVIDER": prov}

        # Resolve selected model for the provider
        chosen_model = (model or "").strip()
        if not chosen_model:
            chosen_model = cls.PROVIDER_DEFAULT_MODELS.get(prov, "")

        if prov == "gemini":
            if api_key:
                settings.GEMINI_API_KEY = api_key.strip()
                env_updates["GEMINI_API_KEY"] = api_key.strip()
            if chosen_model:
                settings.GEMINI_MODEL = chosen_model
                env_updates["GEMINI_MODEL"] = chosen_model
        elif prov == "groq":
            if api_key:
                settings.GROQ_API_KEY = api_key.strip()
                env_updates["GROQ_API_KEY"] = api_key.strip()
            if chosen_model:
                settings.GROQ_MODEL = chosen_model
                env_updates["GROQ_MODEL"] = chosen_model
        elif prov == "openai":
            if api_key:
                settings.OPENAI_API_KEY = api_key.strip()
                env_updates["OPENAI_API_KEY"] = api_key.strip()
            if chosen_model:
                settings.OPENAI_MODEL = chosen_model
                env_updates["OPENAI_MODEL"] = chosen_model
        elif prov == "anthropic":
            if api_key:
                settings.ANTHROPIC_API_KEY = api_key.strip()
                env_updates["ANTHROPIC_API_KEY"] = api_key.strip()
            if chosen_model:
                settings.ANTHROPIC_MODEL = chosen_model
                env_updates["ANTHROPIC_MODEL"] = chosen_model
        elif prov == "ollama":
            if base_url:
                settings.OLLAMA_BASE_URL = base_url.strip()
                env_updates["OLLAMA_BASE_URL"] = base_url.strip()
            if chosen_model:
                settings.OLLAMA_MODEL = chosen_model
                env_updates["OLLAMA_MODEL"] = chosen_model

        # Persist to backend/.env
        try:
            env_file = BASE_DIR / ".env"
            lines = []
            if env_file.exists():
                with open(env_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            existing_keys = set()
            new_lines = []
            for line in lines:
                matched = False
                for k, v in env_updates.items():
                    if line.startswith(f"{k}=") or line.startswith(f"export {k}="):
                        new_lines.append(f'{k}="{v}"\n')
                        existing_keys.add(k)
                        matched = True
                        break
                if not matched:
                    new_lines.append(line)

            for k, v in env_updates.items():
                if k not in existing_keys:
                    new_lines.append(f'{k}="{v}"\n')

            with open(env_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"[LLMService] Could not persist to .env: {e}")

        return cls.get_status()

    @classmethod
    def _call_gemini(cls, question: str, system_prompt: str, user_prompt: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        key = (settings.GEMINI_API_KEY or "").strip()
        if not key or key.startswith("gsk_"):
            print("[LLMService] GEMINI_API_KEY is empty or contains an invalid key format (e.g. Groq 'gsk_' key). Google Gemini requires an AI Studio key.")
            res = cls._offline_rag_engine(question, context_chunks)
            res["answer"] = "[Gemini Notice: Missing or invalid Gemini API key in backend/.env]\n\n" + res["answer"]
            return res

        # Candidate models to try in order of priority (handles version deprecations or transient 503 capacity spikes)
        candidate_models = [settings.GEMINI_MODEL, "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemma-4-26b-a4b-it"]
        candidate_models = list(dict.fromkeys([m.replace("models/", "").strip() for m in candidate_models if m]))

        import time
        last_error = ""
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                "generationConfig": {"temperature": 0.2}
            }
            # Allow up to 2 attempts per candidate if temporary 503 (high demand) is returned
            for attempt in range(2):
                try:
                    with httpx.Client(timeout=35.0) as client:
                        res = client.post(url, json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            candidates = data.get("candidates", [])
                            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                                ans = candidates[0]["content"]["parts"][0].get("text", "")
                                citations = cls._extract_citations_from_text(ans, context_chunks)
                                return {"answer": ans, "citations": citations}
                        elif res.status_code == 503 and attempt == 0:
                            time.sleep(1.2)
                            continue
                        else:
                            err_text = res.text
                            try:
                                err_json = res.json()
                                err_text = err_json.get("error", {}).get("message", res.text)
                            except Exception:
                                pass
                            last_error = f"Gemini ({model_name}) HTTP {res.status_code}: {err_text}"
                            print(f"[LLMService] {last_error}")
                            if res.status_code in (400, 403, 429):
                                break
                except Exception as e:
                    last_error = f"Gemini ({model_name}) connection error: {str(e)}"
                    print(f"[LLMService] {last_error}")

        offline_res = cls._offline_rag_engine(question, context_chunks)
        if last_error:
            offline_res["answer"] = (
                f"[Gemini API Notice: {last_error}]\n\n"
                f"--- Fallback Offline Scientific Synthesis ---\n"
                f"{offline_res['answer']}"
            )
        return offline_res

    @classmethod
    def _call_groq(cls, question: str, system_prompt: str, user_prompt: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        key = (settings.GROQ_API_KEY or "").strip()
        if not key or "your_actual_key" in key:
            print("[LLMService] GROQ_API_KEY is empty or placeholder. Please provide a valid Groq key (starting with 'gsk_') in backend/.env or via the web UI.")
            return cls._offline_rag_engine(question, context_chunks)
        
        import time
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }
        for attempt in range(3):
            try:
                with httpx.Client(timeout=35.0) as client:
                    res = client.post(url, json=payload, headers=headers)
                    if res.status_code == 200:
                        ans = res.json()["choices"][0]["message"]["content"]
                        citations = cls._extract_citations_from_text(ans, context_chunks)
                        return {"answer": ans, "citations": citations}
                    else:
                        print(f"[LLMService] Groq API returned status {res.status_code}: {res.text}")
                        break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.6)
                    continue
                print(f"[LLMService] Groq connection attempt {attempt+1} failed: {e}. Falling back to academic synthesis engine.")
        return cls._offline_rag_engine(question, context_chunks)

    @classmethod
    def _call_openai(cls, question: str, system_prompt: str, user_prompt: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        key = (settings.OPENAI_API_KEY or "").strip()
        if not key or key.startswith("gsk_"):
            print("[LLMService] OPENAI_API_KEY is empty or contains an invalid key format (e.g. Groq 'gsk_' key). OpenAI requires a key starting with 'sk-'.")
            res = cls._offline_rag_engine(question, context_chunks)
            res["answer"] = "[OpenAI Error: Missing or invalid OpenAI API key in backend/.env (must start with 'sk-')]\n\n" + res["answer"]
            return res
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": settings.OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2
            }
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    ans = res.json()["choices"][0]["message"]["content"]
                    citations = cls._extract_citations_from_text(ans, context_chunks)
                    return {"answer": ans, "citations": citations}
                else:
                    err_msg = res.text
                    try:
                        err_json = res.json()
                        err_msg = err_json.get("error", {}).get("message", res.text)
                    except Exception:
                        pass
                    print(f"[LLMService] OpenAI API returned status {res.status_code}: {err_msg}")
                    offline_res = cls._offline_rag_engine(question, context_chunks)
                    if res.status_code == 429:
                        offline_res["answer"] = (
                            f"[OpenAI Quota Exceeded (HTTP 429)]: {err_msg}\n\n"
                            f"> Note: Your OpenAI API key has no remaining billing credits. "
                            f"Add credits at https://platform.openai.com/settings/organization/billing/ or switch to Google Gemini or Groq in the sidebar.\n\n"
                            f"--- Fallback Offline Scientific Synthesis ---\n"
                            f"{offline_res['answer']}"
                        )
                    else:
                        offline_res["answer"] = (
                            f"[OpenAI HTTP {res.status_code}]: {err_msg}\n\n"
                            f"--- Fallback Offline Scientific Synthesis ---\n"
                            f"{offline_res['answer']}"
                        )
                    return offline_res
        except Exception as e:
            print(f"[LLMService] OpenAI error: {e}. Falling back to offline engine.")
            offline_res = cls._offline_rag_engine(question, context_chunks)
            offline_res["answer"] = f"[OpenAI Connection Error: {e}]\n\n--- Fallback Offline Scientific Synthesis ---\n" + offline_res["answer"]
            return offline_res

    @classmethod
    def _call_anthropic(cls, question: str, system_prompt: str, user_prompt: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        key = (settings.ANTHROPIC_API_KEY or "").strip()
        if not key:
            print("[LLMService] ANTHROPIC_API_KEY is empty. Please set your key in backend/.env or via the web UI.")
            return cls._offline_rag_engine(question, context_chunks)
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": settings.ANTHROPIC_MODEL,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": 1024,
                "temperature": 0.2
            }
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    ans = res.json()["content"][0]["text"]
                    citations = cls._extract_citations_from_text(ans, context_chunks)
                    return {"answer": ans, "citations": citations}
                else:
                    print(f"[LLMService] Anthropic API returned status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[LLMService] Anthropic error: {e}. Falling back to offline engine.")
        return cls._offline_rag_engine(question, context_chunks)

    @classmethod
    def _call_ollama(cls, question: str, system_prompt: str, user_prompt: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            url = f"{settings.OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": settings.OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False
            }
            with httpx.Client(timeout=45.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    ans = res.json()["response"]
                    citations = cls._extract_citations_from_text(ans, context_chunks)
                    return {"answer": ans, "citations": citations}
                else:
                    print(f"[LLMService] Ollama returned status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[LLMService] Ollama error: {e}. Falling back to offline engine.")
        return cls._offline_rag_engine(question, context_chunks)

    _flan_tokenizer = None
    _flan_model = None

    @classmethod
    def _get_flan_model(cls):
        if cls._flan_tokenizer is None or cls._flan_model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            cls._flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            cls._flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        return cls._flan_tokenizer, cls._flan_model

    @classmethod
    def _call_local_flan(cls, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs offline on-device neural model google/flan-t5-small to synthesize answers strictly from context."""
        try:
            tokenizer, model = cls._get_flan_model()

            top_texts = []
            for c in context_chunks[:3]:
                txt = c.get("text", "")[:300]
                top_texts.append(txt)
            context_str = " ".join(top_texts)

            prompt = f"context: {context_str} question: {question}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=120)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not generated_text or len(generated_text) < 4:
                return cls._offline_rag_engine(question, context_chunks)

            # Ground with explicit paper citation
            top_c = context_chunks[0] if context_chunks else {}
            cite_tag = f" [Paper: {top_c.get('paper_title', 'Corpus')}, Year: {top_c.get('year', 2024)}, Section: {top_c.get('section_name', 'general')}]"
            full_answer = f"According to the indexed research literature: {generated_text.capitalize()}.{cite_tag}"

            citations = [
                {
                    "paper_id": c.get("paper_id", ""),
                    "paper_title": c.get("paper_title", "Unknown"),
                    "year": c.get("year", 2024),
                    "section_name": c.get("section_name", "general"),
                    "chunk_id": c.get("chunk_id", 0),
                    "quote": c.get("text", "")[:120].strip()
                }
                for c in context_chunks[:3]
            ]
            return {"answer": full_answer, "citations": citations}
        except Exception as e:
            print(f"[LLMService] Local neural model error: {e}. Falling back to academic synthesis engine.")
            return cls._offline_rag_engine(question, context_chunks)

    @classmethod
    def _offline_rag_engine(cls, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """High-precision academic answer synthesis with rigorous source citation grounding.
        
        Synthesizes a coherent research narrative structured by findings, methodology,
        and limitations/gaps without raw chunk dumping.
        """
        clean_q = (question or "").strip()
        if not clean_q:
            return {
                "answer": "Please provide a specific research question or topic (e.g., 'What are the main limitations in query routing?') to synthesize evidence from the indexed papers.",
                "citations": []
            }

        if not context_chunks:
            return {
                "answer": "The current paper collection does not provide sufficient evidence to answer this question.",
                "citations": []
            }

        # Filter out common stop-words
        stop_words = {
            "what", "how", "why", "the", "and", "is", "in", "of", "for", "with",
            "does", "can", "are", "about", "discuss", "discussed", "give", "tell",
            "which", "when", "where", "papers", "paper", "research", "study"
        }
        q_terms = set(re.findall(r"\w+", clean_q.lower())) - stop_words

        # Limitation/gap indicators
        gap_keywords = {"however", "limit", "limitation", "limitations", "challenge", "challenges", "lack", "fails", "fail", "cost", "bound", "open problem", "future work", "remain", "remains", "unexplored", "scarce"}

        # Group chunks by paper
        papers_map = {}
        for chunk in context_chunks:
            pid = chunk.get("paper_id") or chunk.get("paper_title", "Unknown")
            if pid not in papers_map:
                papers_map[pid] = {
                    "paper_title": chunk.get("paper_title", "Unknown"),
                    "year": chunk.get("year", 2024),
                    "chunks": []
                }
            papers_map[pid]["chunks"].append(chunk)

        findings_summary = []
        limitations_summary = []
        citations = []
        seen_quotes = set()

        for pid, pdata in papers_map.items():
            title = pdata["paper_title"]
            year = pdata["year"]

            # Analyze sentences within each chunk of this paper
            best_finding = None
            best_limitation = None
            best_finding_score = -1

            for chunk in pdata["chunks"]:
                sec = chunk.get("section_name", "general")
                cid = chunk.get("chunk_id", 0)
                text = chunk.get("text", "")

                sentences = re.split(r"(?<=[.!?])\s+", text)
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) < 35 or len(s_clean) > 350:
                        continue

                    s_lower = s_clean.lower()
                    words = set(re.findall(r"\w+", s_lower))

                    overlap = len(q_terms.intersection(words)) if q_terms else 1
                    is_gap = any(gk in s_lower for gk in gap_keywords)

                    if is_gap and not best_limitation:
                        best_limitation = (s_clean, sec, cid)

                    score = overlap + (0.5 if sec in ["results", "abstract", "methodology"] else 0.0)
                    if score > best_finding_score and not is_gap:
                        best_finding_score = score
                        best_finding = (s_clean, sec, cid)

            # If no non-gap sentence qualified, fall back to first reasonable sentence
            if not best_finding and pdata["chunks"]:
                first_text = pdata["chunks"][0].get("text", "")
                candidate = re.split(r"(?<=[.!?])\s+", first_text)
                valid = [c.strip() for c in candidate if 35 <= len(c.strip()) <= 350]
                if valid:
                    best_finding = (valid[0], pdata["chunks"][0].get("section_name", "general"), pdata["chunks"][0].get("chunk_id", 0))

            if best_finding:
                sent, sec, cid = best_finding
                raw_chunk = next((c for c in pdata["chunks"] if c.get("chunk_id") == cid), pdata["chunks"][0] if pdata["chunks"] else None)
                raw_text = raw_chunk.get("text", "").strip() if raw_chunk else sent
                sim_score = round(float(raw_chunk.get("similarity_score", 0.0)), 3) if raw_chunk and raw_chunk.get("similarity_score") is not None else None
                findings_summary.append({
                    "paper_title": title,
                    "year": year,
                    "section": sec,
                    "claim": sent,
                    "chunk_id": cid,
                    "paper_id": pid
                })
                if sent not in seen_quotes:
                    seen_quotes.add(sent)
                    citations.append({
                        "paper_id": pid,
                        "paper_title": title,
                        "year": year,
                        "section_name": sec,
                        "chunk_id": cid,
                        "quote": sent,
                        "evidence_text": raw_text,
                        "similarity_score": sim_score
                    })

            if best_limitation:
                sent, sec, cid = best_limitation
                raw_chunk = next((c for c in pdata["chunks"] if c.get("chunk_id") == cid), pdata["chunks"][0] if pdata["chunks"] else None)
                raw_text = raw_chunk.get("text", "").strip() if raw_chunk else sent
                sim_score = round(float(raw_chunk.get("similarity_score", 0.0)), 3) if raw_chunk and raw_chunk.get("similarity_score") is not None else None
                limitations_summary.append({
                    "paper_title": title,
                    "year": year,
                    "section": sec,
                    "limitation": sent,
                    "chunk_id": cid,
                    "paper_id": pid
                })
                if sent not in seen_quotes and len(citations) < 5:
                    seen_quotes.add(sent)
                    citations.append({
                        "paper_id": pid,
                        "paper_title": title,
                        "year": year,
                        "section_name": sec,
                        "chunk_id": cid,
                        "quote": sent,
                        "evidence_text": raw_text,
                        "similarity_score": sim_score
                    })

        # Synthesize coherent academic narrative
        paragraphs = []
        is_gap_inquiry = any(k in clean_q.lower() for k in ["gap", "gaps", "limit", "limitation", "challenge", "future", "problem", "bottleneck"])

        if is_gap_inquiry and limitations_summary:
            paragraphs.append("Analysis of the retrieved literature reveals several core limitations and research gaps:")
            for item in limitations_summary[:3]:
                paragraphs.append(f"- In '{item['paper_title']}' ({item['year']}, Section: {item['section']}), the authors observe: \"{item['limitation']}\" [Paper: {item['paper_title']}, Year: {item['year']}, Section: {item['section']}].")
            if findings_summary:
                top_f = findings_summary[0]
                paragraphs.append(f"Contextually, these constraints stem from foundational methodologies such as those investigated in '{top_f['paper_title']}' ({top_f['year']}), which establishes that {top_f['claim'][:180].lower()} [Paper: {top_f['paper_title']}, Year: {top_f['year']}, Section: {top_f['section']}].")
        elif findings_summary:
            paragraphs.append("Synthesized evidence from the indexed scientific corpus:")
            for item in findings_summary[:3]:
                paragraphs.append(f"- {item['claim']} [Paper: {item['paper_title']}, Year: {item['year']}, Section: {item['section']}].")
            if limitations_summary:
                top_l = limitations_summary[0]
                paragraphs.append(f"Noted Constraints: {top_l['limitation']} [Paper: {top_l['paper_title']}, Year: {top_l['year']}, Section: {top_l['section']}].")
        else:
            first_chunk = context_chunks[0]
            paragraphs.append(f"Based on indexed paper '{first_chunk.get('paper_title', 'Corpus')}' ({first_chunk.get('year', 2024)}), the literature outlines methodologies and experimental findings in {first_chunk.get('section_name', 'general')} [Paper: {first_chunk.get('paper_title')}, Year: {first_chunk.get('year')}, Section: {first_chunk.get('section_name')}].")

        answer = "\n\n".join(paragraphs).replace("\u2011", "-")
        return {"answer": answer, "citations": citations}

    @classmethod
    def _extract_citations_from_text(cls, text: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses [Paper: ..., Year: ..., Section: ...] tags into structured citation objects with full evidence context."""
        citations = []
        # Support tags with or without ChunkID
        matches = re.findall(r"\[Paper:\s*([^,\]]+),\s*Year:\s*([^,\]]+),\s*Section:\s*([^\]]+)\]", text)
        for m in matches:
            p_title, p_year, raw_sec = m[0].strip(), m[1].strip(), m[2].strip()

            # Handle optional ", ChunkID: 123" inside Section
            chunk_id_match = re.search(r",\s*ChunkID:\s*(\d+)", raw_sec, re.IGNORECASE)
            extracted_cid = int(chunk_id_match.group(1)) if chunk_id_match else None
            p_sec = re.sub(r",\s*ChunkID:\s*\d+", "", raw_sec, flags=re.IGNORECASE).strip()

            # Match to chunk by chunk_id or title substring
            matching_chunk = None
            if extracted_cid is not None:
                matching_chunk = next((c for c in context_chunks if c.get("chunk_id") == extracted_cid), None)
            if not matching_chunk:
                matching_chunk = next((c for c in context_chunks if p_title.lower() in c.get("paper_title", "").lower() or c.get("paper_title", "").lower() in p_title.lower()), None)
            if not matching_chunk and context_chunks:
                matching_chunk = context_chunks[0]

            full_passage = matching_chunk.get("text", "").strip() if matching_chunk else ""
            sim_score = round(float(matching_chunk.get("similarity_score", 0.0)), 3) if matching_chunk and matching_chunk.get("similarity_score") is not None else None

            # Extract first clean complete sentence for quote
            sentences = re.split(r"(?<=[.!?])\s+", full_passage) if full_passage else []
            quote_sent = sentences[0].strip() if sentences and len(sentences[0].strip()) > 25 else full_passage[:220].strip()

            citations.append({
                "paper_id": matching_chunk.get("paper_id", "") if matching_chunk else "",
                "paper_title": p_title,
                "year": int(p_year) if p_year.isdigit() else (matching_chunk.get("year", 2024) if matching_chunk else 2024),
                "section_name": p_sec,
                "chunk_id": matching_chunk.get("chunk_id", extracted_cid or 0) if matching_chunk else (extracted_cid or 0),
                "quote": quote_sent,
                "evidence_text": full_passage,
                "similarity_score": sim_score
            })

        # If LLM didn't produce exact tags, provide default citations from top context chunks
        if not citations and context_chunks:
            for c in context_chunks[:3]:
                txt = c.get("text", "").strip()
                sents = re.split(r"(?<=[.!?])\s+", txt)
                q_txt = sents[0].strip() if sents and len(sents[0].strip()) > 25 else txt[:200]
                citations.append({
                    "paper_id": c.get("paper_id", ""),
                    "paper_title": c.get("paper_title", "Unknown"),
                    "year": c.get("year", 2024),
                    "section_name": c.get("section_name", "general"),
                    "chunk_id": c.get("chunk_id", 0),
                    "quote": q_txt,
                    "evidence_text": txt,
                    "similarity_score": round(float(c.get("similarity_score", 0.0)), 3) if c.get("similarity_score") is not None else None
                })

        return citations
