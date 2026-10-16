import os
import sys
import json
import uuid
import pprint
from pathlib import Path
from PIL import Image
import streamlit as st

CURRENT_PATH = Path(__file__).resolve().parent
if str(CURRENT_PATH) not in sys.path:
    sys.path.insert(0, str(CURRENT_PATH))

# Configure page metadata and layout first
st.set_page_config(
    page_title="Image Processing Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom headers */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 50%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #8A9A86;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: rgba(77, 150, 255, 0.06);
        border-left: 5px solid #4D96FF;
        padding: 15px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 12px;
    }
    
    /* Custom Badges */
    .badge-run {
        background-color: #2ECC71;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-skip {
        background-color: #7F8C8D;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Code styling */
    pre {
        background-color: #1E1E1E !important;
        border-radius: 10px !important;
        border: 1px solid #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Set environment variable to run in Lite mode (no conductor/redis required)
os.environ["OMAGENT_MODE"] = "lite"
os.environ["DEBUG"] = "false"  # Override system DEBUG=release which breaks pydantic bool parsing
os.environ["custom_openai_model"] = "qwen3:8b"
os.environ["custom_openai_endpoint"] = "http://localhost:11434/v1"
os.environ["custom_openai_key"] = "ollama"
os.environ["openai_api_key"] = "ollama"

from omagent_core.utils.logger import logging
logging.init_logger("omagent", "omagent", level="INFO")

from omagent_core.clients.devices.programmatic import ProgrammaticClient
from omagent_core.utils.container import container
from omagent_core.utils.registry import registry
from workflow.image_agent import create_image_agent_workflow

CURRENT_PATH = Path(__file__).parents[0]

# Ensure uploads folder exists
uploads_dir = CURRENT_PATH.joinpath("uploads")
uploads_dir.mkdir(exist_ok=True)

# Lazy initialization of OmAgent client and workflow
@st.cache_resource
def init_agent():
    # Register workers
    registry.import_module(project_path=CURRENT_PATH.joinpath("workers"))
    container.register_stm("SharedMemSTM")
    container.from_config(CURRENT_PATH.joinpath("container.yaml"))
    
    # Create and register workflow
    workflow = create_image_agent_workflow()
    workflow.register(overwrite=True)
    
    # Return instantiated programmatic client
    config_path = CURRENT_PATH.joinpath("configs")
    client = ProgrammaticClient(processor=workflow, config_path=config_path)
    return client

# Render Title
st.markdown("<h1 class='main-title'>OmAgent Multi-worker Image Processing Agent</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image and ask a question. The Planner worker will dynamically orchestrate specialized tools (OCR, Object Detection, Captioning) to compile a structured reasoning response.</p>", unsafe_allow_html=True)



# Layout split
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Input Section")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display image preview
        img_preview = Image.open(uploaded_file)
        st.image(img_preview, caption="Uploaded Image", use_container_width=True)
        
        # Save file to uploads folder
        file_ext = uploaded_file.name.split(".")[-1]
        temp_filename = f"upload_{uuid.uuid4().hex}.{file_ext}"
        temp_file_path = uploads_dir.joinpath(temp_filename)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["image_path"] = str(temp_file_path.absolute())
    
    st.markdown("---")
    
    # Prompt presets
    st.markdown("**💡 Quick Preset Prompts**")
    preset_cols = st.columns(2)
    with preset_cols[0]:
        preset_1 = st.button("🔍 Extract all text")
        preset_2 = st.button("💼 Describe this invoice")
    with preset_cols[1]:
        preset_3 = st.button("👥 Count the people")
        preset_4 = st.button("🖼️ What is in this image?")
        
    # Set input text depending on preset clicks
    prompt_value = "What is in this image?"
    if preset_1:
        prompt_value = "Extract all text."
    elif preset_2:
        prompt_value = "Describe this invoice details: invoice number, date, and vendor."
    elif preset_3:
        prompt_value = "Count the people."
    elif preset_4:
        prompt_value = "What is in this image? Provide a caption and summary."
        
    user_prompt = st.text_input("💬 Ask the Agent:", value=prompt_value)
    
    run_btn = st.button("🚀 Process Image with Agent", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Agent Execution Results")
    
    if run_btn:
        if uploaded_file is None:
            st.error("Please upload an image first!")
        else:
            with st.spinner("Initializing OmAgent Orchestration & loading models..."):
                try:
                    client = init_agent()
                except Exception as e:
                    st.error(f"Failed to initialize OmAgent: {e}")
                    client = None
                    
            if client is not None:
                with st.spinner("Agent is reasoning and executing workers..."):
                    # Prepare input parameters
                    inputs = {
                        "image_path": st.session_state["image_path"],
                        "user_prompt": user_prompt
                    }
                    
                    try:
                        # Start programmatic client run
                        results_list = client.start_batch_processor(workflow_input_list=[inputs])
                        
                        if results_list and len(results_list) > 0:
                            result = results_list[0]
                            st.success("Workflow completed successfully!")
                            
                            # Display natural language answer
                            st.markdown("#### 💬 Agent Response")
                            st.write(result.get("answer", "No answer generated."))
                            
                            # Display JSON Structured data
                            st.markdown("#### 📊 Extracted Structured Data")
                            st.json(result.get("structured_data", {}))
                            
                            st.markdown("---")
                            st.markdown("#### ⏳ Execution Trace & Specialized Worker Logs")
                            
                            # Display timeline steps based on task_outputs
                            task_outputs = client._interactor._executor.local_executor.task_outputs
                            
                            # 1. Planner Output
                            planner_out = task_outputs.get("planner_task", {}).get("output", {})
                            if planner_out:
                                run_ocr = planner_out.get("run_ocr", False)
                                run_detector = planner_out.get("run_detector", False)
                                run_caption = planner_out.get("run_caption", False)
                                
                                st.markdown("##### 📍 Planner Decisions")
                                st.markdown(f"""
                                - **OCR Worker**: {'<span class="badge-run">RUN</span>' if run_ocr else '<span class="badge-skip">SKIP</span>'}
                                - **Object Detector**: {'<span class="badge-run">RUN</span>' if run_detector else '<span class="badge-skip">SKIP</span>'}
                                - **Captioner**: {'<span class="badge-run">RUN</span>' if run_caption else '<span class="badge-skip">SKIP</span>'}
                                """, unsafe_allow_html=True)
                            
                            # 2. OCR Task details
                            ocr_out = task_outputs.get("ocr_task", {}).get("output", {})
                            if ocr_out and (ocr_out.get("text") or ocr_out.get("structured_data")):
                                with st.expander("📝 OCR Worker Output", expanded=False):
                                    # Existing output (UNCHANGED)
                                    if ocr_out.get("text"):
                                        st.text(ocr_out.get("text"))
                                        st.divider()

                                    # Retrieve structured JSON (using rich fields if present, fallback if not)
                                    if ocr_out.get("structured_data"):
                                        ocr_json = ocr_out.get("structured_data")
                                    else:
                                        ocr_json = {
                                            "ocr_text": ocr_out.get("text", "")
                                        }

                                    st.subheader("OCR JSON")
                                    st.json(ocr_json)

                                    st.download_button(
                                        label="📥 Download OCR JSON",
                                        data=json.dumps(
                                            ocr_json,
                                            indent=4,
                                            ensure_ascii=False
                                        ),
                                        file_name="ocr_output.json",
                                        mime="application/json"
                                    )
                            else:
                                st.caption("🔕 OCR Worker was skipped.")
                                
                            # 3. Detector Task details
                            detector_out = task_outputs.get("detector_task", {}).get("output", {})
                            if detector_out and detector_out.get("objects"):
                                with st.expander("🎯 Detector Worker Output", expanded=False):
                                    st.write(f"Detected Labels: `{detector_out.get('objects')}`")
                            else:
                                st.caption("🔕 Detector Worker was skipped.")
                                
                            # 4. Caption Task details
                            caption_out = task_outputs.get("caption_task", {}).get("output", {})
                            if caption_out and caption_out.get("caption"):
                                with st.expander("🖼️ Caption Worker Output", expanded=False):
                                    st.write(caption_out.get("caption"))
                            else:
                                st.caption("🔕 Caption Worker was skipped.")
                        else:
                            st.error("No results returned from workflow.")
                    except Exception as ex:
                        st.error(f"Error during workflow execution: {ex}")
    else:
        st.info("Upload an image and run the query to see the agent orchestration details.")
        
    st.markdown('</div>', unsafe_allow_html=True)
