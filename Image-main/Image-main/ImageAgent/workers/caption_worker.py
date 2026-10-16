from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message, Content
from omagent_core.utils.general import encode_image
from omagent_core.utils.registry import registry

@registry.register_worker()
class CaptionWorker(BaseWorker, BaseLLMBackend):
    """Caption Worker that generates descriptions using local models or GPT-4o Vision fallback."""
    
    llm: OpenaiGPTLLM

    def _run(self, run_caption: bool, *args, **kwargs):
        if not run_caption:
            return {"caption": ""}
            
        # Retrieve cached image from short-term memory (STM)
        img_cache = self.stm(self.workflow_instance_id).get("image_cache")
        if not img_cache or "<image_0>" not in img_cache:
            return {"caption": ""}
            
        img = img_cache["<image_0>"]
        
        # Try local pipeline first
        try:
            from transformers import pipeline
            # Uses a small, fast local image-to-text model
            captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            res = captioner(img)
            caption = res[0].get("generated_text", "")
            if caption:
                return {"caption": caption}
        except Exception:
            pass
            
        # Fallback to GPT-4o Vision
        try:
            chat_messages = [
                Message(role="user", message_type="text", content="Describe this image in one concise sentence.")
            ]
            chat_messages.append(
                Message(
                    role="user",
                    message_type="image",
                    content=[
                        Content(
                            type="image_url",
                            image_url={
                                "url": f"data:image/jpeg;base64,{encode_image(img)}"
                            },
                        )
                    ],
                )
            )
            chat_complete_res = self.llm.generate(records=chat_messages)
            caption = chat_complete_res["choices"][0]["message"]["content"].strip()
            return {"caption": caption}
        except Exception as e:
            return {"caption": f"Failed to generate caption: {str(e)}"}
