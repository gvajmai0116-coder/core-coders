import json
import re
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry
from tools.ocr import PaddleOCRTool
from tools.document_parser import parse_document

@registry.register_worker()
class OCRWorker(BaseWorker, BaseLLMBackend):
    """OCR Worker that extracts text using enhanced PaddleOCR and parses structured document fields."""
    
    llm: OpenaiGPTLLM
    _tool: PaddleOCRTool = None

    def _run(self, run_ocr: bool, *args, **kwargs):
        if not run_ocr:
            return {"text": "", "structured_data": {}}
        
        # Lazy initialization of the tool
        if self._tool is None:
            self._tool = PaddleOCRTool()
            
        # Retrieve cached image from short-term memory (STM)
        img_cache = self.stm(self.workflow_instance_id).get("image_cache")
        if not img_cache or "<image_0>" not in img_cache:
            return {"text": "", "structured_data": {}}
            
        img = img_cache["<image_0>"]
        try:
            ocr_res = self._tool.perform_ocr(img)
            text = ocr_res.get("text", "") if isinstance(ocr_res, dict) else str(ocr_res)
        except Exception as e:
            return {"text": f"Error running OCR: {str(e)}", "structured_data": {}}

        if not text.strip():
            return {"text": "", "structured_data": {}}

        # Step 1: Run fast & deterministic document parser (e.g. B. Braun DN template parser / generic regexes)
        structured_json = parse_document(text)

        # Step 2: Use LLM to complement/refine if additional context is needed
        system_prompt = (
            "You are a document understanding assistant.\n"
            "An automated document parser has already extracted the following structured data from multiline OCR text:\n"
            f"{json.dumps(structured_json, indent=2)}\n\n"
            "Analyze the multiline OCR text below and return a refined JSON matching this schema:\n"
            "{\n"
            '  "document_type": "Identified document type",\n'
            '  "title": "Document title/heading",\n'
            '  "organization": "Company or organization name",\n'
            '  "fields": {\n'
            '     "key_name": "value"\n'
            '  },\n'
            '  "dates": ["List of dates"],\n'
            '  "ids": ["List of IDs, reference numbers, SOP numbers, etc."],\n'
            '  "checked_options": ["List of checked options"],\n'
            '  "handwritten_entries": {\n'
            '     "field_name": "value"\n'
            '  },\n'
            '  "summary": "Brief summary",\n'
            '  "raw_text": "Full multiline OCR text"\n'
            "}\n"
            "Keep all pre-extracted non-null fields accurate and intact. Do not lose structure."
        )

        chat_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Multiline OCR text:\n{text}")
        ]

        try:
            res = self.llm.generate(records=chat_messages)
            content = res["choices"][0]["message"]["content"].strip()
            
            # Clean markdown wrappers if returned
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            
            llm_json = json.loads(content)
            
            # Merge deterministic parsed fields over LLM values to guarantee accuracy
            for key in ["document_type", "organization", "title"]:
                if structured_json.get(key):
                    llm_json[key] = structured_json[key]
            
            if structured_json.get("fields"):
                llm_fields = llm_json.get("fields", {})
                llm_fields.update({k: v for k, v in structured_json["fields"].items() if v is not None})
                llm_json["fields"] = llm_fields

            structured_json = llm_json
        except Exception:
            # Fallback to deterministic regex parsed JSON directly
            pass

        return {
            "text": text,
            "structured_data": structured_json
        }
