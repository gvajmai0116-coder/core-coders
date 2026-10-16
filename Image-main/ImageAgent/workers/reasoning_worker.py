import json
import re
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry

@registry.register_worker()
class ReasoningWorker(BaseWorker, BaseLLMBackend):
    """Reasoning Worker that synthesizes outputs from OCR, detection, and captioning into a cohesive response."""
    
    llm: OpenaiGPTLLM

    def _run(self, user_prompt: str, ocr_output: dict, detector_output: list, caption_output: str, *args, **kwargs):
        # Extract structured data and raw multiline text safely
        ocr_structured = {}
        ocr_text = ""
        if isinstance(ocr_output, dict):
            ocr_structured = ocr_output.get("structured_data", {})
            ocr_text = ocr_output.get("text", "")
        else:
            ocr_text = str(ocr_output) if ocr_output else ""

        ocr_structured_str = json.dumps(ocr_structured, indent=2) if ocr_structured else "(Skipped or no structured data)"

        system_prompt = (
            "You are the Reasoning worker of an Image Processing Agent.\n"
            "The specialized OCR worker has already pre-extracted structured data and preserved line breaks from the image.\n"
            "Do NOT attempt to guess or re-infer key values—simply use the pre-extracted structured OCR values, YOLO detections, and visual captions to answer the user's prompt directly and accurately.\n\n"
            "Inputs:\n"
            f"- User Prompt: {user_prompt}\n"
            f"- Structured OCR Data:\n{ocr_structured_str}\n\n"
            f"- Detected Objects: {detector_output if detector_output else '(Skipped)'}\n"
            f"- Image Description: {caption_output if caption_output else '(Skipped)'}\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "answer": "A clear, natural-language response directly answering the user prompt using the pre-extracted fields.",\n'
            '  "structured_data": {\n'
            '     "document_type": "Document type or category",\n'
            '     "organization": "Organization or vendor name",\n'
            '     "document_id": "Primary document ID / reference number or null",\n'
            '     "date": "Primary date or effective date or null",\n'
            '     "fields": {\n'
            '        "key_name": "value"\n'
            '     },\n'
            '     "objects": ["List of unique object types detected"],\n'
            '     "summary": "Concise summary of the document and image"\n'
            '  }\n'
            "}\n"
            "Ensure valid JSON without markdown wrapping or explanations outside the JSON."
        )
        
        chat_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content="Synthesize the pre-extracted data into the final JSON response.")
        ]
        
        try:
            res = self.llm.generate(records=chat_messages)
            content = res["choices"][0]["message"]["content"].strip()
            
            # Clean markdown code blocks
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            
            parsed = json.loads(content)
            self.callback.send_answer(self.workflow_instance_id, msg=parsed.get("answer", ""))
            return parsed
        except Exception:
            # Deterministic fallback directly using structured fields
            fields = ocr_structured.get("fields", {})
            doc_id = fields.get("document_id") or (ocr_structured.get("ids")[0] if (isinstance(ocr_structured.get("ids"), list) and ocr_structured["ids"]) else None)
            date_val = fields.get("effective_date") or fields.get("when") or (ocr_structured.get("dates")[0] if (isinstance(ocr_structured.get("dates"), list) and ocr_structured["dates"]) else None)
            org_val = ocr_structured.get("organization") or ocr_structured.get("title")

            fallback_answer = (
                f"Document Type: {ocr_structured.get('document_type', 'Generic Document')}. "
                f"Organization: {org_val or 'N/A'}. "
                f"Document ID: {doc_id or 'N/A'}. "
                f"Summary: {ocr_structured.get('summary', ocr_text[:100])}"
            )

            fallback = {
                "answer": fallback_answer,
                "structured_data": {
                    "document_type": ocr_structured.get("document_type"),
                    "organization": org_val,
                    "document_id": doc_id,
                    "date": date_val,
                    "fields": fields,
                    "objects": detector_output,
                    "summary": ocr_structured.get("summary") or caption_output or "Processed image"
                }
            }
            self.callback.send_answer(self.workflow_instance_id, msg=fallback["answer"])
            return fallback
