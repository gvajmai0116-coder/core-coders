import json
import re
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry

@registry.register_worker()
class Planner(BaseWorker, BaseLLMBackend):
    """Planner Worker that determines which specialized workers should run based on the user prompt."""
    
    llm: OpenaiGPTLLM

    def _run(self, user_prompt: str, *args, **kwargs):
        system_prompt = (
            "You are the Planner worker of an Image Processing Agent. "
            "Your task is to decide which specialized workers must execute to address the user's prompt.\n"
            "Decide for each of these three workers:\n"
            "1. run_ocr: Set to true if the user query asks to read text, extract text, transcribe, or analyze text-heavy elements like invoices, receipts, signs, or documents.\n"
            "2. run_detector: Set to true if the user query asks to count items, check count, find objects, detect logos, or locate items (e.g. people, cars, logos, objects).\n"
            "3. run_caption: Set to true if the user query asks for a general description, summary of the scene, what is happening in the image, or general layout.\n\n"
            "Respond ONLY with a JSON object containing exactly these fields (values as boolean):\n"
            "{\n"
            '  "run_ocr": true/false,\n'
            '  "run_detector": true/false,\n'
            '  "run_caption": true/false\n'
            "}\n"
            "Do not include any explanation, thinking process, or markdown block styling."
        )
        
        chat_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"User prompt: {user_prompt}")
        ]
        
        res = self.llm.generate(records=chat_messages)
        content = res["choices"][0]["message"]["content"].strip()
        
        # Clean markdown wrappers if returned
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        
        try:
            decision = json.loads(content)
            run_caption = bool(decision.get("run_caption", False))
        except Exception:
            prompt_lower = user_prompt.lower()
            run_caption = any(x in prompt_lower for x in ["describe", "what is", "summary", "summarize", "caption"])
        
        # Override to True as requested by user
        run_ocr = True
        run_detector = True
        
        return {
            "run_ocr": run_ocr,
            "run_detector": run_detector,
            "run_caption": run_caption
        }
