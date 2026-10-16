import os
import sys
from pathlib import Path

CURRENT_PATH = Path(__file__).resolve().parent
if str(CURRENT_PATH) not in sys.path:
    sys.path.insert(0, str(CURRENT_PATH))

os.environ["OMAGENT_MODE"] = "lite"
os.environ["DEBUG"] = "false"  # Override system DEBUG=release which breaks pydantic bool parsing
if not os.environ.get("custom_openai_model"):
    os.environ["custom_openai_model"] = "qwen3:8b"
if not os.environ.get("custom_openai_endpoint"):
    os.environ["custom_openai_endpoint"] = "http://localhost:11434/v1"
if not os.environ.get("custom_openai_key"):
    os.environ["custom_openai_key"] = "ollama"
if not os.environ.get("openai_api_key"):
    os.environ["openai_api_key"] = "ollama"

from omagent_core.clients.devices.programmatic import ProgrammaticClient
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from workflow.image_agent import create_image_agent_workflow

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("workers"))

container.register_stm("SharedMemSTM")
container.from_config(CURRENT_PATH.joinpath("container.yaml"))

# Create and register workflow
workflow = create_image_agent_workflow()
workflow.register(overwrite=True)

# Initialize programmatic client
config_path = CURRENT_PATH.joinpath("configs")
client = ProgrammaticClient(processor=workflow, config_path=config_path)

if __name__ == "__main__":
    # Ensure uploads directory exists
    uploads_dir = CURRENT_PATH.joinpath("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Create a dummy image with text for testing
    dummy_img_path = uploads_dir.joinpath("test_dummy.png")
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (600, 250), color = (40, 44, 52))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
    except Exception:
        font = None
    d.text((40, 40), "Invoice INV-12345\nDate: 2026-07-27\nTotal: $150.00", fill=(255, 255, 0), font=font)
    img.save(dummy_img_path)
    print(f"Created/Updated test dummy image with Arial font at {dummy_img_path}")
        
    print("\n--- Running OCR Test ---")
    inputs = {
        "image_path": str(dummy_img_path.absolute()),
        "user_prompt": "What is the invoice number and total amount?"
    }
    
    # Configure OpenAI API keys in environment if not present
    if not os.getenv("openai_api_key") and not os.getenv("custom_openai_key"):
        print("WARNING: Neither openai_api_key nor custom_openai_key is set in environment.")
        
    results = client.start_batch_processor(workflow_input_list=[inputs])
    print("\nWorkflow Final Output:")
    import pprint
    pprint.pprint(results)
    
    print("\n--- Execution Timeline & Outputs ---")
    # Retrieve all task execution histories from executor
    for task_name, task_out in client._interactor._executor.local_executor.task_outputs.items():
        print(f"\nTask: {task_name}")
        pprint.pprint(task_out)
        
    client.stop_processor()
