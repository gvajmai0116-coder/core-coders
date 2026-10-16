from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task

def create_image_agent_workflow() -> ConductorWorkflow:
    """Builds and orchestrates the image agent workflow graph.
    
    Dataflow:
    Workflow Input (image_path, user_prompt) ->
      ImageLoader(image_path) ->
      Planner(user_prompt) -> outputs (run_ocr, run_detector, run_caption) ->
      OCRWorker(run_ocr) -> output (text) ->
      DetectorWorker(run_detector) -> output (objects) ->
      CaptionWorker(run_caption) -> output (caption) ->
      ReasoningWorker(user_prompt, OCR text, YOLO labels, Caption) -> Final Output.
    """
    workflow = ConductorWorkflow(name="image_agent_workflow")
    
    # 1. Load image and cache it in workflow STM
    loader_task = simple_task(
        task_def_name="ImageLoader",
        task_reference_name="image_loader_task",
        inputs={"image_path": workflow.input("image_path")}
    )
    
    # 2. Decide tool routing based on user prompt
    planner_task = simple_task(
        task_def_name="Planner",
        task_reference_name="planner_task",
        inputs={"user_prompt": workflow.input("user_prompt")}
    )
    
    # 3. Extract text (conditional)
    ocr_task = simple_task(
        task_def_name="OCRWorker",
        task_reference_name="ocr_task",
        inputs={"run_ocr": planner_task.output("run_ocr")}
    )
    
    # 4. Detect objects (conditional)
    detector_task = simple_task(
        task_def_name="DetectorWorker",
        task_reference_name="detector_task",
        inputs={"run_detector": planner_task.output("run_detector")}
    )
    
    # 5. Generate image description (conditional)
    caption_task = simple_task(
        task_def_name="CaptionWorker",
        task_reference_name="caption_task",
        inputs={"run_caption": planner_task.output("run_caption")}
    )
    
    # 6. Generate final natural language response and structured JSON metadata
    reasoning_task = simple_task(
        task_def_name="ReasoningWorker",
        task_reference_name="reasoning_task",
        inputs={
            "user_prompt": workflow.input("user_prompt"),
            "ocr_output": ocr_task.output(),
            "detector_output": detector_task.output("objects"),
            "caption_output": caption_task.output("caption")
        }
    )
    
    # Set execution order: sequential flow
    workflow >> loader_task >> planner_task >> ocr_task >> detector_task >> caption_task >> reasoning_task
    
    return workflow
