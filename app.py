import os
import re
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI — Remote Sensing Intelligence Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --sq-bg: #0b0f14;
            --sq-panel: #111821;
            --sq-panel-soft: #0f151d;
            --sq-border: #233041;
            --sq-text: #eef3f8;
            --sq-muted: #8fa1b3;
            --sq-accent: #55a7ff;
            --sq-accent-soft: rgba(85, 167, 255, 0.10);
            --sq-success: #4fc38a;
            --sq-warning: #e6b85c;
            --sq-danger: #ef6b73;
        }

        .stApp {
            background: var(--sq-bg);
        }

        .block-container {
            max-width: 1520px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #0d131a;
        }

        .sq-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.25rem;
            padding: 0.35rem 0 1.25rem 0;
            border-bottom: 1px solid var(--sq-border);
            margin-bottom: 1.25rem;
        }

        .sq-title {
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 800;
            color: var(--sq-text);
            letter-spacing: -0.02em;
            margin: 0;
        }

        .sq-subtitle {
            color: var(--sq-muted);
            margin-top: 0.42rem;
            font-size: 0.98rem;
        }

        .sq-badges {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            justify-content: flex-end;
            padding-top: 0.1rem;
        }

        .sq-badge {
            border: 1px solid var(--sq-border);
            background: var(--sq-panel);
            color: #c8d4df;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            white-space: nowrap;
        }

        .sq-badge strong {
            color: var(--sq-text);
            font-weight: 700;
        }

        .sq-section-label {
            color: var(--sq-muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .sq-panel {
            background: var(--sq-panel);
            border: 1px solid var(--sq-border);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.75rem;
        }

        .sq-result {
            background: linear-gradient(
                180deg,
                rgba(85, 167, 255, 0.07),
                rgba(85, 167, 255, 0.02)
            );
            border: 1px solid rgba(85, 167, 255, 0.22);
            border-radius: 14px;
            padding: 1.2rem 1.25rem;
            margin-top: 0.35rem;
        }

        .sq-result-title {
            color: var(--sq-muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .sq-answer {
            color: var(--sq-text);
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 0;
        }

        .sq-meta {
            color: var(--sq-muted);
            font-size: 0.84rem;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed #33465b;
            border-radius: 12px;
            padding: 0.15rem 0.25rem;
            background: var(--sq-panel-soft);
        }

        div[data-testid="stMetric"] {
            background: var(--sq-panel);
            border: 1px solid var(--sq-border);
            padding: 0.85rem 1rem;
            border-radius: 12px;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--sq-muted);
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            min-height: 2.8rem;
            font-weight: 700;
            border-radius: 10px;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        .sq-footer {
            color: #6f8090;
            font-size: 0.78rem;
            padding-top: 1.1rem;
            border-top: 1px solid var(--sq-border);
            margin-top: 2rem;
        }

        @media (max-width: 900px) {
            .sq-header {
                flex-direction: column;
            }
            .sq-badges {
                justify-content: flex-start;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND IMPORT
# ============================================================

try:
    from satquery_ai.agent.controller import AgentController

    ENGINE_AVAILABLE = True
    ENGINE_IMPORT_ERROR = ""
except Exception as exc:
    ENGINE_AVAILABLE = False
    ENGINE_IMPORT_ERROR = str(exc)


# ============================================================
# CONSTANTS / MODE MAPPING
# ============================================================

MODE_BACKEND_MAP = {
    "Single Image": "Single-Scene Visual Question Answering (VQA)",
    "Change Detection": "Bi-Temporal Change Detection",
    "SAR + Optical": "Optical + SAR Joint Analysis",
}

PROMPTS = {
    "Single Image": [
        "Describe this remote-sensing scene.",
        "What is the dominant land-cover type?",
        "Is a large water body visible in this image?",
        "Identify visible agricultural areas.",
        "Describe the built-up regions visible in the image.",
    ],
    "Change Detection": [
        "Describe the major changes between these two images.",
        "Where did change occur?",
        "Estimate the changed area.",
        "Highlight the detected change regions.",
    ],
    "SAR + Optical": [
        "Analyze complementary information from the optical and SAR images.",
        "Compare structural information across the SAR and optical data.",
        "Summarize the fused observation from both modalities.",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def safe_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    return re.sub(r"[^\w\-.,;()_ ]+", "_", basename)


def persist_upload(uploaded_file, prefix: str) -> Optional[str]:
    if uploaded_file is None:
        return None

    os.makedirs("uploads", exist_ok=True)
    name = safe_filename(uploaded_file.name)
    output_path = os.path.abspath(
        os.path.join("uploads", f"{prefix}_{name}")
    )
    uploads_root = os.path.abspath("uploads")

    if not output_path.startswith(uploads_root):
        raise ValueError("Invalid uploaded filename.")

    with open(output_path, "wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())

    return output_path


def preview_uploaded_image(uploaded_file, caption: str) -> None:
    if uploaded_file is None:
        st.markdown(
            '<div class="sq-panel"><span class="sq-meta">'
            "No image uploaded yet."
            "</span></div>",
            unsafe_allow_html=True,
        )
        return

    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.image(image, caption=caption, width="stretch")
        st.caption(
            f"{uploaded_file.name} · "
            f"{image.width}×{image.height} · "
            f"{(uploaded_file.size / 1024):.1f} KB"
        )
        uploaded_file.seek(0)
    except Exception:
        st.info(
            f"{uploaded_file.name} uploaded. "
            "Preview is unavailable for this image format."
        )
        try:
            uploaded_file.seek(0)
        except Exception:
            pass


def normalize_remote_metadata(results: Dict[str, Any]) -> Dict[str, Any]:
    metadata = results.get("remote_vlm_metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def get_confidence_values(
    results: Dict[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    metadata = normalize_remote_metadata(results)

    raw_conf = metadata.get("confidence", results.get("confidence"))
    raw_percent = metadata.get(
        "confidence_percent",
        results.get("confidence_percent"),
    )

    model_trace = results.get("model_trace")
    model_trace = model_trace if isinstance(model_trace, dict) else {}

    confidence_type = (
        metadata.get("confidence_type")
        or model_trace.get("confidence_type")
        or results.get("confidence_type")
    )

    confidence = (
        float(raw_conf)
        if isinstance(raw_conf, (int, float))
        else None
    )

    confidence_percent = (
        float(raw_percent)
        if isinstance(raw_percent, (int, float))
        else None
    )

    if confidence_percent is None and confidence is not None:
        confidence_percent = confidence * 100.0

    return confidence, confidence_percent, confidence_type


def format_runtime_ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "Unavailable"

    milliseconds = float(value)
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.2f} s"
    return f"{milliseconds:.0f} ms"


def bool_label(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unavailable"


def render_spatial_evidence(results: Dict[str, Any]) -> None:
    evidence = results.get("spatial_evidence")
    if not isinstance(evidence, list) or not evidence:
        return

    st.markdown("#### Visual / Spatial Evidence")

    rows = []
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue

        confidence = item.get("confidence")
        confidence_display = (
            f"{float(confidence):.2f}"
            if isinstance(confidence, (int, float))
            else "Unavailable"
        )

        rows.append(
            {
                "Region": index,
                "Label": item.get("label", "Unlabeled"),
                "Specialist": item.get("specialist", "—"),
                "Coordinates": str(item.get("box", "—")),
                "Confidence": confidence_display,
            }
        )

    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)


def render_visual_evidence(
    results: Dict[str, Any],
    primary_path: str,
    secondary_path: Optional[str],
    mode: str,
) -> None:
    overlay_path = results.get("overlay_image_path")
    change_map = (
        results.get("change_map_path")
        or results.get("change_mask_path")
        or results.get("mask_path")
    )

    has_generated_visual = any(
        path and os.path.exists(str(path))
        for path in [overlay_path, change_map]
    )

    if not has_generated_visual and mode == "Single Image":
        return

    st.markdown("#### Visual Evidence")

    if mode in {"Change Detection", "SAR + Optical"} and secondary_path:
        left, right = st.columns(2)

        with left:
            st.caption(
                "T1 / Before"
                if mode == "Change Detection"
                else "Optical / Primary"
            )
            try:
                st.image(primary_path, width="stretch")
            except Exception:
                st.info("Primary image preview unavailable.")

        with right:
            st.caption(
                "T2 / After"
                if mode == "Change Detection"
                else "SAR / Secondary"
            )
            try:
                st.image(secondary_path, width="stretch")
            except Exception:
                st.info("Secondary image preview unavailable.")

    if change_map and os.path.exists(str(change_map)):
        st.caption("Detected change evidence")
        st.image(str(change_map), width="stretch")

    if overlay_path and os.path.exists(str(overlay_path)):
        st.caption("Generated spatial overlay")
        st.image(str(overlay_path), width="stretch")


def render_execution_summary(results: Dict[str, Any]) -> None:
    metadata = normalize_remote_metadata(results)
    model_trace = results.get("model_trace")
    model_trace = model_trace if isinstance(model_trace, dict) else {}

    task = (
        metadata.get("task")
        or model_trace.get("selected_task")
        or results.get("task")
        or "Unavailable"
    )
    specialist = (
        results.get("routed_tool")
        or model_trace.get("selected_specialist")
        or model_trace.get("tool")
        or "Unavailable"
    )
    model_name = (
        metadata.get("base_model")
        or model_trace.get("base_model")
        or model_trace.get("model")
        or "Unavailable"
    )
    adapter = (
        metadata.get("adapter_bucket")
        or model_trace.get("adapter_bucket")
        or "Unavailable"
    )
    adapter_source = (
        metadata.get("adapter_source")
        or model_trace.get("adapter_source")
        or "Unavailable"
    )
    lora_verified = (
        metadata.get("lora_verified")
        if "lora_verified" in metadata
        else model_trace.get("lora_verified")
    )
    remote_service = (
        model_trace.get("remote_service")
        or (
            "Hugging Face ZeroGPU"
            if metadata
            else "Local specialist"
        )
    )
    runtime = (
        metadata.get("execution_time_ms")
        or model_trace.get("execution_time_ms")
        or results.get("execution_time_ms")
    )
    _, _, confidence_type = get_confidence_values(results)

    with st.expander("Execution Summary", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Task**  \n{task}")
            st.markdown(f"**Specialist**  \n{specialist}")
            st.markdown(f"**Model**  \n{model_name}")
            st.markdown(f"**Execution**  \n{remote_service}")

        with col2:
            st.markdown(f"**Adapter**  \n{adapter}")
            st.markdown(f"**Adapter Source**  \n{adapter_source}")
            st.markdown(
                f"**LoRA Verified**  \n{bool_label(lora_verified)}"
            )
            st.markdown(
                f"**Runtime**  \n{format_runtime_ms(runtime)}"
            )

        st.markdown(
            f"**Confidence Type**  \n"
            f"{confidence_type or 'Unavailable'}"
        )

        trace_summary = results.get("trace_summary")
        if trace_summary:
            st.markdown("---")
            st.markdown("**Execution Trace Summary**")
            st.markdown(str(trace_summary))


def render_debug_details(results: Dict[str, Any]) -> None:
    with st.expander("Developer Details", expanded=False):
        st.json(results)


@st.cache_resource(show_spinner=False)
def get_controller() -> Optional[Any]:
    if not ENGINE_AVAILABLE:
        return None
    try:
        return AgentController()
    except Exception:
        return None


def execute_satquery(
    query: str,
    primary_path: Optional[str],
    secondary_path: Optional[str],
    mode: str,
) -> Dict[str, Any]:
    if not ENGINE_AVAILABLE:
        return {
            "error": (
                "SatQuery engine could not be loaded. "
                f"{ENGINE_IMPORT_ERROR or ''}"
            ).strip()
        }

    if not primary_path:
        return {"error": "Primary satellite image is required."}

    image_paths = [primary_path]
    if secondary_path:
        image_paths.append(secondary_path)

    controller = get_controller()
    if controller is None:
        controller = AgentController()

    process_fn = (
        getattr(controller, "process_query", None)
        or getattr(controller, "execute_query", None)
    )

    if not callable(process_fn):
        return {
            "error": (
                "AgentController does not expose process_query "
                "or execute_query."
            )
        }

    backend_mode = MODE_BACKEND_MAP[mode]

    response = process_fn(
        query=query,
        images=image_paths,
        task_mode=backend_mode,
    )

    if isinstance(response, dict):
        return response

    return {
        "error": "SatQuery returned an unexpected response type."
    }


def set_prompt(value: str) -> None:
    st.session_state["satquery_prompt"] = value


# ============================================================
# SESSION STATE
# ============================================================

if "satquery_prompt" not in st.session_state:
    st.session_state["satquery_prompt"] = PROMPTS["Single Image"][0]

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "last_context" not in st.session_state:
    st.session_state["last_context"] = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="sq-header">
        <div>
            <div class="sq-title">SatQuery AI</div>
            <div class="sq-subtitle">
                Remote Sensing Intelligence Console ·
                Agentic satellite image analysis with
                multimodal vision-language and specialist tools
            </div>
        </div>
        <div class="sq-badges">
            <div class="sq-badge">
                <strong>Remote VLM</strong> · HF ZeroGPU
            </div>
            <div class="sq-badge">
                <strong>Local Tools</strong> · Change + Fusion
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TASK MODE
# ============================================================

st.markdown(
    '<div class="sq-section-label">Analysis Mode</div>',
    unsafe_allow_html=True,
)

mode = st.radio(
    "Analysis Mode",
    options=list(MODE_BACKEND_MAP.keys()),
    horizontal=True,
    label_visibility="collapsed",
)

# Keep prompt sensible when switching modes for the first time.
mode_state_key = "last_selected_mode"
if st.session_state.get(mode_state_key) != mode:
    st.session_state[mode_state_key] = mode
    st.session_state["satquery_prompt"] = PROMPTS[mode][0]


# ============================================================
# WORKSPACE
# ============================================================

if mode == "Single Image":
    image_col, query_col = st.columns([1.15, 0.85], gap="large")

    with image_col:
        st.markdown("### Satellite Image")
        primary_file = st.file_uploader(
            "Upload GeoTIFF, TIFF, PNG or JPEG",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="single_primary",
        )
        preview_uploaded_image(primary_file, "Uploaded satellite image")

    with query_col:
        st.markdown("### Ask SatQuery")
        st.caption("Choose a prompt or write your own question.")

        quick_cols = st.columns(2)
        for index, prompt in enumerate(PROMPTS[mode][:4]):
            with quick_cols[index % 2]:
                st.button(
                    prompt,
                    key=f"single_prompt_{index}",
                    width="stretch",
                    on_click=set_prompt,
                    args=(prompt,),
                )

        user_query = st.text_area(
            "What would you like to know about this image?",
            key="satquery_prompt",
            height=150,
        )

        submit_btn = st.button(
            "Analyze Image",
            type="primary",
            width="stretch",
        )

    secondary_file = None

elif mode == "Change Detection":
    st.markdown("### Before / After Imagery")
    before_col, after_col = st.columns(2, gap="large")

    with before_col:
        primary_file = st.file_uploader(
            "T1 / Before image",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="change_primary",
        )
        preview_uploaded_image(primary_file, "T1 / Before")

    with after_col:
        secondary_file = st.file_uploader(
            "T2 / After image",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="change_secondary",
        )
        preview_uploaded_image(secondary_file, "T2 / After")

    st.markdown("### Change Analysis Query")

    prompt_cols = st.columns(4)
    for index, prompt in enumerate(PROMPTS[mode]):
        with prompt_cols[index]:
            st.button(
                prompt,
                key=f"change_prompt_{index}",
                width="stretch",
                on_click=set_prompt,
                args=(prompt,),
            )

    user_query = st.text_area(
        "What would you like to know about the change?",
        key="satquery_prompt",
        height=110,
    )

    submit_btn = st.button(
        "Analyze Changes",
        type="primary",
        width="stretch",
    )

else:
    st.markdown("### Multimodal Input")
    optical_col, sar_col = st.columns(2, gap="large")

    with optical_col:
        primary_file = st.file_uploader(
            "Optical image",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="fusion_primary",
        )
        preview_uploaded_image(primary_file, "Optical / Primary")

    with sar_col:
        secondary_file = st.file_uploader(
            "SAR image",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="fusion_secondary",
        )
        preview_uploaded_image(secondary_file, "SAR / Secondary")

    st.markdown("### Cross-Modal Analysis Query")

    prompt_cols = st.columns(3)
    for index, prompt in enumerate(PROMPTS[mode]):
        with prompt_cols[index]:
            st.button(
                prompt,
                key=f"fusion_prompt_{index}",
                width="stretch",
                on_click=set_prompt,
                args=(prompt,),
            )

    user_query = st.text_area(
        "What would you like SatQuery to analyze across both modalities?",
        key="satquery_prompt",
        height=110,
    )

    submit_btn = st.button(
        "Run Fusion Analysis",
        type="primary",
        width="stretch",
    )


# ============================================================
# EXECUTION
# ============================================================

if submit_btn:
    validation_error = None

    if primary_file is None:
        validation_error = "Upload a satellite image before running analysis."
    elif not user_query.strip():
        validation_error = "Enter a question or analysis prompt."
    elif mode in {"Change Detection", "SAR + Optical"} and secondary_file is None:
        validation_error = (
            "Change Detection requires both T1 and T2 images."
            if mode == "Change Detection"
            else "SAR + Optical analysis requires both images."
        )

    if validation_error:
        st.warning(validation_error)
    else:
        try:
            primary_path = persist_upload(primary_file, "primary")
            secondary_path = persist_upload(
                secondary_file,
                "secondary",
            )

            with st.status(
                "Running SatQuery analysis…",
                expanded=True,
            ) as status:
                st.write("Preparing satellite imagery…")
                st.write("Routing query to the appropriate specialist…")

                if mode == "Single Image":
                    st.write("Running remote vision-language analysis…")
                elif mode == "Change Detection":
                    st.write("Running local bi-temporal change analysis…")
                else:
                    st.write("Running local SAR-optical fusion analysis…")

                results = execute_satquery(
                    query=user_query,
                    primary_path=primary_path,
                    secondary_path=secondary_path,
                    mode=mode,
                )

                if results.get("error") or results.get("success") is False:
                    status.update(
                        label="SatQuery analysis failed",
                        state="error",
                        expanded=True,
                    )
                else:
                    status.update(
                        label="Analysis complete",
                        state="complete",
                        expanded=False,
                    )

            st.session_state["last_result"] = results
            st.session_state["last_context"] = {
                "mode": mode,
                "primary_path": primary_path,
                "secondary_path": secondary_path,
            }

        except Exception as exc:
            st.session_state["last_result"] = {
                "error": str(exc)
            }
            st.session_state["last_context"] = {
                "mode": mode,
                "primary_path": None,
                "secondary_path": None,
            }


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.get("last_result")
context = st.session_state.get("last_context")

if isinstance(results, dict):
    st.markdown("---")
    st.markdown(
        '<div class="sq-section-label">Analysis Result</div>',
        unsafe_allow_html=True,
    )

    # Check for hidden VLM errors that weren't propagated as top-level "error"
    vlm_metadata = results.get("remote_vlm_metadata") or {}
    vlm_error_hidden = vlm_metadata.get("error") if isinstance(vlm_metadata, dict) else None

    if results.get("error") or results.get("success") is False or vlm_error_hidden:
        if vlm_error_hidden and not results.get("error"):
            # VLM failed but controller didn't surface it as top-level error
            err_msg = vlm_error_hidden
        else:
            err_msg = results.get('error') or vlm_error_hidden or "An unknown error occurred during inference."
        st.error(
            "SatQuery could not complete this request. "
            f"{err_msg}"
        )

        # Show actionable diagnostics if it looks like auth/quota issue
        lower_err = err_msg.lower()
        if "zerogpu" in lower_err or "quota" in lower_err or "token" in lower_err or "auth" in lower_err:
            st.info(
                "**Quick checks to restore VLM service:**\n\n"
                "1. Ensure `HF_TOKEN` is set in your Streamlit Cloud app secrets\n"
                "   (create a token at https://huggingface.co/settings/tokens with **Write** scope).\n"
                "2. Confirm the token has **Write** scope (required for ZeroGPU).\n"
                "3. Restart your Streamlit app after updating the secret.\n"
                "4. Free quota is ~5 min/day; monitor usage on the Space page.\n"
                "5. If quota is exhausted, wait until the next day or switch to a paid GPU Space."
            )
    else:
        answer = (
            results.get("answer")
            or results.get("analysis")
            or "Analysis completed, but no textual answer was returned."
        )

        st.markdown(
            f"""
            <div class="sq-result">
                <div class="sq-result-title">SatQuery Analysis</div>
                <p class="sq-answer">{str(answer)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        confidence, confidence_percent, confidence_type = (
            get_confidence_values(results)
        )

        metadata = normalize_remote_metadata(results)
        runtime = (
            metadata.get("execution_time_ms")
            or results.get("execution_time_ms")
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            if confidence_percent is not None:
                st.metric(
                    "Generation Confidence",
                    f"{confidence_percent:.2f}%",
                )
            else:
                st.metric(
                    "Model Confidence",
                    "Unavailable",
                )

        with metric2:
            st.metric(
                "Runtime",
                format_runtime_ms(runtime),
            )

        with metric3:
            specialist = (
                results.get("routed_tool")
                or (
                    results.get("model_trace", {}).get(
                        "selected_specialist"
                    )
                    if isinstance(
                        results.get("model_trace"),
                        dict,
                    )
                    else None
                )
                or "—"
            )
            st.metric(
                "Specialist",
                str(specialist),
            )

        if confidence_type:
            st.caption(
                "Confidence source: "
                f"{confidence_type}. "
                "This is a model-generation confidence proxy, "
                "not an accuracy score."
            )

        render_spatial_evidence(results)

        if context:
            render_visual_evidence(
                results=results,
                primary_path=context.get("primary_path"),
                secondary_path=context.get("secondary_path"),
                mode=context.get("mode", "Single Image"),
            )

        render_execution_summary(results)

        with st.expander("Download / Developer Tools", expanded=False):
            debug_enabled = st.checkbox(
                "Show raw result JSON",
                value=False,
                key="debug_result_json",
            )

            if debug_enabled:
                st.json(results)

            try:
                from satquery_ai.utils.report_generator import ReportGenerator

                pdf_path = ReportGenerator.generate_pdf(
                    results,
                    output_pdf_path="SatQuery_AI_Report.pdf",
                )
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download Analysis Report (PDF)",
                        data=pdf_file.read(),
                        file_name="SatQuery_AI_Report.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )
            except Exception as pdf_error:
                st.caption(
                    f"PDF export unavailable: {pdf_error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="sq-footer">
        SatQuery AI prototype · Built for the ISRO / Smart India
        Hackathon remote-sensing problem statement.
        Heavy VLM inference runs remotely; change detection and
        SAR-optical fusion remain specialist local workflows.
    </div>
    """,
    unsafe_allow_html=True,
)