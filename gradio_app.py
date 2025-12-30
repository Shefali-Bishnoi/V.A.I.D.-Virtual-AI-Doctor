#VoiceBot UI with Gradio - Multi-tab Interface

import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_loaded = load_dotenv()
if env_loaded:
    print("✅ Loaded .env file successfully")
else:
    print("⚠️  No .env file found (will use system environment variables)")

from doctor_brain import encode_image, analyze_image_with_query
from patient_voice import transcribe_with_groq
from doctor_voice import text_to_speech_with_gtts, text_to_speech_with_elevenlabs
from mental_wellbeing import (
    analyze_mental_wellbeing,
    detect_emergency,
    get_emergency_response,
    reset_conversation,
    analyze_image_emotion,
)

# Check API keys on startup
print("=" * 60)
print("🔑 API Key Status Check")
print("=" * 60)
GROQ_KEY = os.environ.get("GROQ_API_KEY")
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY")

if GROQ_KEY:
    print(f"✅ GROQ_API_KEY: Found ({GROQ_KEY[:10]}...{GROQ_KEY[-5:] if len(GROQ_KEY) > 15 else '***'})")
else:
    print("❌ GROQ_API_KEY: NOT FOUND!")
    print("   Please set GROQ_API_KEY in your environment variables or .env file")
    print("   Get your free API key at: https://console.groq.com/")

if ELEVENLABS_KEY:
    print(f"✅ ELEVENLABS_API_KEY: Found ({ELEVENLABS_KEY[:10]}...{ELEVENLABS_KEY[-5:] if len(ELEVENLABS_KEY) > 15 else '***'})")
else:
    print("⚠️  ELEVENLABS_API_KEY: NOT FOUND!")
    print("   Audio responses will fall back to gTTS (free but lower quality)")
    print("   Get your API key at: https://elevenlabs.io/")

print("=" * 60)

# --- SYSTEM PROMPT FOR MEDICAL DIAGNOSIS ---
system_prompt_base = """You have to act as a professional doctor.
Analyze the image and the patient's query.
If you make a differential, suggest some remedies. 
Do not add any numbers or special characters.
Your response should be in one long paragraph.
Do not say 'In the image I see', just start with your assessment.
Keep your answer concise (max 2 sentences)."""

# --- MEDICAL DIAGNOSIS FUNCTIONS ---
def process_medical_inputs(image_filepath, audio_filepath, text_input, language):
    """Process inputs for medical diagnosis tab."""
    # 1. Map code to full language name
    lang_map = {
        "en": "English", 
        "hi": "Hindi", 
        "es": "Spanish", 
        "fr": "French", 
        "de": "German"
    }
    target_language = lang_map.get(language, "English")

    # 2. Handle Audio vs Text Input
    speech_to_text_output = ""
    if audio_filepath:
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
            audio_filepath=audio_filepath, 
            stt_model="whisper-large-v3",
            language=language
        )
    elif text_input:
        speech_to_text_output = text_input
    else:
        return "No input provided", "Please provide voice or text input.", None

    # 3. Analyze Image with Language Instruction
    if image_filepath:
        final_prompt = (
            f"{system_prompt_base}\n\n"
            f"Patient Query: {speech_to_text_output}\n"
            f"IMPORTANT: You must respond in the {target_language} language only."
        )
        
        doctor_response = analyze_image_with_query(
            query=final_prompt, 
            encoded_image=encode_image(image_filepath), 
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for me to analyze."

    # 4. Generate Audio Response
    voice_of_doctor = text_to_speech_with_elevenlabs(doctor_response, "final_medical.mp3")
    
    return speech_to_text_output, doctor_response, voice_of_doctor


# --- MENTAL WELLBEING FUNCTIONS ---
def process_mental_wellbeing(image_filepath, audio_filepath, text_input, language, conversation_history):
    """Process inputs for mental wellbeing tab (optional image + voice/text)."""
    # 1. Handle Audio vs Text Input
    user_input = ""
    if audio_filepath:
        try:
            user_input = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3",
                language=language,
            )
        except Exception as e:
            print(f"Transcription error: {e}")
            user_input = ""
    elif text_input:
        user_input = text_input.strip()

    # 2. Optional: Analyze uploaded image for emotional context
    image_context = ""
    if image_filepath:
        try:
            print(f"🖼️ Analyzing uploaded image for emotional context: {image_filepath}")
            image_context = analyze_image_emotion(image_path=image_filepath, language=language)
            if image_context:
                print(f"✅ Image emotion analysis: {image_context[:100]}...")
            else:
                print("⚠️ Image emotion analysis returned empty result")
        except Exception as e:
            print(f"❌ Image emotion analysis error: {e}")
            image_context = ""

    # If no text/voice AND no image context, ask for some input
    if not user_input and not image_context:
        return "", "Please share your thoughts with text, voice, or optionally a photo.", None, gr.update(visible=False), 0, []

    # 3. Build combined input for the therapeutic agent
    combined_input = user_input or ""
    if image_context:
        if combined_input:
            combined_input = (
                combined_input
                + "\n\nAdditional emotional context inferred from their photo "
                + "(do NOT mention the photo explicitly to the user): "
                + image_context
            )
        else:
            combined_input = (
                "The user did not provide any text, but from their photo they appear emotionally as follows: "
                + image_context
            )

    # What to show back to the user in the transcript box
    display_input = user_input if user_input else (image_context or "Image uploaded (no text provided)")

    # 4. Analyze mental wellbeing
    try:
        print(f"🔍 Calling analyze_mental_wellbeing with input: '{combined_input[:80]}...'")
        result = analyze_mental_wellbeing(
            user_input=combined_input,
            conversation_history=conversation_history if conversation_history else None,
            language=language,
        )
        print(f"📦 Result from analyze_mental_wellbeing: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")

        # Unpack result safely
        if isinstance(result, tuple) and len(result) >= 3:
            response, is_emergency, severity_score = result[0], result[1], result[2]
            print(f"✅ Unpacked: response='{str(response)[:50]}...', is_emergency={is_emergency}, severity={severity_score}")
        else:
            print(f"❌ Unexpected result format: {result}")
            raise ValueError(f"Unexpected result format from analyze_mental_wellbeing: {result}")

        # Ensure response is a string
        if not response or not isinstance(response, str):
            print(f"⚠️ Response is not a valid string: {response}")
            response = "I apologize, but I'm having trouble processing your message right now. Please try again or contact a mental health professional."

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        response = "I apologize, but I'm having trouble processing your message right now. Please try again or contact a mental health professional."
        is_emergency = False
        severity_score = 0

    # 5. Generate Audio Response
    voice_response = None
    if response:
        try:
            voice_response = text_to_speech_with_elevenlabs(response, "final_mental.mp3")
        except Exception as e:
            print(f"TTS error: {e}")

    # 6. Update conversation history
    try:
        updated_history = conversation_history.copy() if conversation_history else []
        updated_history.append({"role": "user", "content": display_input})
        updated_history.append({"role": "assistant", "content": str(response)})
    except Exception as e:
        print(f"❌ History update error: {e}")
        updated_history = conversation_history if conversation_history else []

    # 7. Show emergency alert if needed
    emergency_visible = is_emergency or severity_score >= 8
    emergency_msg = ""
    if emergency_visible:
        emergency_msg = get_emergency_response(language)

    # Ensure all return values are properly formatted
    try:
        return_value = (
            str(display_input) if display_input else "",
            str(response) if response else "I'm here to help. Can you tell me more?",
            voice_response,
            gr.update(value=emergency_msg, visible=emergency_visible),
            float(severity_score) if severity_score else 0.0,
            updated_history,
        )
        print("✅ Returning from process_mental_wellbeing")
        return return_value
    except Exception as e:
        print(f"❌ Return value error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback return
        return (
            str(display_input) if display_input else "",
            "I'm here to help. Can you tell me more?",
            None,
            gr.update(visible=False),
            0.0,
            [],
        )

def clear_mental_conversation():
    """Clear the mental wellbeing conversation history."""
    print("🔄 Clearing mental wellbeing conversation...")
    reset_conversation()  # Reset the conversation manager
    print("✅ Clear function called - returning empty values")
    return [], "", "", None, gr.update(value="", visible=False), 0

# --- CUSTOM THEME SETTINGS ---
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="#d4f1f4",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#b2dfdb",
    block_title_text_weight="600",
    block_label_text_weight="500",
    block_label_text_color="#0d9488",
    input_background_fill="#f0fdfa",
    button_primary_background_fill="#14b8a6",
    button_primary_background_fill_hover="#0d9488",
    button_primary_text_color="#ffffff",
)

# Custom CSS for better text visibility and styling
custom_css = """
/* Main title styling */
h1 {
    text-align: center;
    color: #0f766e !important;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-weight: 700;
}

/* Subtitle styling */
h3 {
    color: #115e59 !important;
    font-weight: 600;
}

/* Container background */
.gradio-container {
    background: linear-gradient(135deg, #d4f1f4 0%, #b2dfdb 100%);
}

/* All text elements should be dark and readable */
.block label, 
.block span,
.block p {
    color: #1f2937 !important;
    font-weight: 500;
}

/* Input fields */
textarea, 
input[type="text"] {
    color: #111827 !important;
    background-color: #f0fdfa !important;
    border: 1px solid #99f6e4 !important;
    font-size: 14px;
}

textarea::placeholder,
input::placeholder {
    color: #6b7280 !important;
}

/* Output textboxes */
.output-text textarea {
    color: #111827 !important;
    background-color: #ecfdf5 !important;
    font-size: 14px;
    line-height: 1.6;
}

/* Markdown content */
.markdown-text, 
.markdown p, 
.markdown span, 
.markdown div,
.prose {
    color: #1f2937 !important;
}

/* Tab labels - Multiple selectors to ensure it works */
.tab-nav button,
.tabs button,
.tabitem button,
button[role="tab"],
.gr-button.tab {
    color: #000000 !important;
    font-weight: 500 !important;
}

.tab-nav button[aria-selected="true"],
.tabs button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Target tab content text */
.tabitem span,
.tab-nav span,
.tabs span {
    color: #000000 !important;
}

/* Dropdown styling */
.dropdown-container label {
    color: #0d9488 !important;
    font-weight: 600;
}

/* Section headers */
.block-title {
    color: #0f766e !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Emergency alert styling */
.emergency-alert {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 2px solid #dc2626;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    color: #7f1d1d !important;
    font-weight: 600;
    box-shadow: 0 4px 6px rgba(220, 38, 38, 0.1);
}

.emergency-alert p,
.emergency-alert span {
    color: #7f1d1d !important;
}

/* Button styling */
button {
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
}

/* Info text styling */
.info {
    color: #4b5563 !important;
    font-size: 13px;
}

/* Slider labels */
.slider-container label {
    color: #374151 !important;
    font-weight: 500;
}

/* Footer disclaimer */
.disclaimer {
    color: #6b7280 !important;
    font-size: 13px;
    font-style: italic;
}

/* Audio player styling */
audio {
    border-radius: 8px;
}

/* Image upload area */
.image-container {
    border: 2px dashed #99f6e4 !important;
    border-radius: 12px !important;
    background-color: #f0fdfa !important;
}

/* Ensure all nested elements inherit proper text color */
* {
    color: inherit;
}

/* Force dark text in all interactive elements */
.gr-box *,
.gr-form *,
.gr-input * {
    color: #1f2937 !important;
}
"""

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=theme, css=custom_css, title="V.A.I.D. - Virtual AI Doctor") as demo:
    
    # Header Section
    gr.Markdown("# 🏥 V.A.I.D. (Virtual AI Doctor)")
    gr.Markdown("### *Intelligent Health Analysis with Vision & Voice*")
    
    # Main Tabs
    with gr.Tabs() as main_tabs:
        
        # ========== TAB 1: MEDICAL DIAGNOSIS ==========
        with gr.Tab("🩺 Medical Diagnosis"):
            with gr.Row():
                # --- LEFT COLUMN (Inputs) ---
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload Symptoms 📸")
                    medical_image_input = gr.Image(
                        type="filepath", 
                        label="Upload Image", 
                        sources=["upload", "webcam"], 
                        height=300,
                        elem_classes=["image-container"]
                    )
                    
                    gr.Markdown("### 2. Select Language 🌐")
                    medical_language_dropdown = gr.Dropdown(
                        choices=[
                            ("English", "en"), 
                            ("Hindi", "hi"), 
                            ("Spanish", "es"), 
                            ("French", "fr"), 
                            ("German", "de")
                        ], 
                        value="en", 
                        label="Language"
                    )
                    
                    gr.Markdown("### 3. Describe Problem 🗣️")
                    with gr.Tabs():
                        with gr.TabItem("🎤 Voice Input"):
                            medical_audio_input = gr.Audio(
                                sources=["microphone", "upload"], 
                                type="filepath", 
                                label="Record Voice"
                            )
                        with gr.TabItem("⌨️ Text Input"):
                            medical_text_input = gr.Textbox(
                                label="Type Symptoms", 
                                placeholder="E.g., I have a red rash on my arm...", 
                                lines=3
                            )
                    
                    # Big Submit Button
                    medical_submit_btn = gr.Button(
                        "👨‍⚕️ Consult V.A.I.D.", 
                        variant="primary", 
                        size="lg"
                    )

                # --- RIGHT COLUMN (Outputs) ---
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Patient Transcript")
                    medical_input_transcript = gr.Textbox(
                        label="You Said:", 
                        lines=2, 
                        interactive=False,
                        elem_classes=["output-text"]
                    )
                    
                    gr.Markdown("### 🩺 V.A.I.D. Diagnosis")
                    medical_doctor_text_output = gr.Textbox(
                        label="Analysis", 
                        lines=8, 
                        interactive=False,
                        show_label=False,
                        elem_classes=["output-text"]
                    )
                    
                    gr.Markdown("### 🔊 Audio Response")
                    medical_audio_output = gr.Audio(
                        label="Listen to Doctor", 
                        autoplay=True
                    )

            # Click Event for Medical Diagnosis
            medical_submit_btn.click(
                fn=process_medical_inputs,
                inputs=[
                    medical_image_input, 
                    medical_audio_input, 
                    medical_text_input, 
                    medical_language_dropdown
                ],
                outputs=[
                    medical_input_transcript, 
                    medical_doctor_text_output, 
                    medical_audio_output
                ]
            )
        
        # ========== TAB 2: MENTAL WELLBEING ==========
        with gr.Tab("🧠 Mental Well-being"):
            with gr.Row():
                # --- LEFT COLUMN (Inputs) ---
                with gr.Column(scale=1):
                    gr.Markdown("### 💭 Share Your Thoughts")
                    gr.Markdown("*I'm here to listen and support you. Share what's on your mind.*")
                    
                    gr.Markdown("### 1. Select Language 🌐")
                    mental_language_dropdown = gr.Dropdown(
                        choices=[
                            ("English", "en"), 
                            ("Hindi", "hi"), 
                            ("Spanish", "es"), 
                            ("French", "fr"), 
                            ("German", "de")
                        ], 
                        value="en",
                        label="Language",
                    )

                    gr.Markdown("### 2. Optional: Share a Photo 🙂")
                    mental_image_input = gr.Image(
                        type="filepath",
                        label="Upload your photo (optional)",
                        sources=["upload", "webcam"],
                        height=220,
                        elem_classes=["image-container"],
                    )
                    
                    gr.Markdown("### 3. Express Yourself 🗣️")
                    with gr.Tabs():
                        with gr.TabItem("🎤 Voice Input"):
                            mental_audio_input = gr.Audio(
                                sources=["microphone", "upload"], 
                                type="filepath", 
                                label="Record Voice"
                            )
                        with gr.TabItem("⌨️ Text Input"):
                            mental_text_input = gr.Textbox(
                                label="Type Your Thoughts", 
                                placeholder="E.g., I've been feeling really anxious lately...", 
                                lines=4
                            )
                    
                    # Submit and Clear Buttons
                    with gr.Row():
                        mental_submit_btn = gr.Button(
                            "💬 Talk to Counselor", 
                            variant="primary", 
                            size="lg"
                        )
                        mental_clear_btn = gr.Button(
                            "🔄 Clear Conversation", 
                            variant="secondary"
                        )

                # --- RIGHT COLUMN (Outputs) ---
                with gr.Column(scale=1):
                    # Emergency Alert (initially hidden)
                    emergency_alert = gr.Markdown(
                        value="",
                        visible=False,
                        elem_classes=["emergency-alert"]
                    )
                    
                    gr.Markdown("### 📝 Your Message")
                    mental_input_transcript = gr.Textbox(
                        label="You Said:", 
                        lines=2, 
                        interactive=False,
                        elem_classes=["output-text"]
                    )
                    
                    gr.Markdown("### 💙 Counselor Response")
                    mental_counselor_output = gr.Textbox(
                        label="Response", 
                        lines=8, 
                        interactive=False,
                        show_label=False,
                        elem_classes=["output-text"]
                    )
                    
                    gr.Markdown("### 🔊 Audio Response")
                    mental_audio_output = gr.Audio(
                        label="Listen to Response", 
                        autoplay=True
                    )
                    
                    # Severity Indicator
                    gr.Markdown("### 📊 Concern Level")
                    severity_indicator = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=1,
                        label="Severity Score (0-10)",
                        interactive=False,
                        info="Higher scores indicate more serious concerns",
                        elem_classes=["slider-container"]
                    )
                    
                    # Conversation History (hidden state)
                    mental_conversation_history = gr.State(value=[])

            # Click Events for Mental Wellbeing
            mental_submit_btn.click(
                fn=process_mental_wellbeing,
                inputs=[
                    mental_image_input,
                    mental_audio_input,
                    mental_text_input,
                    mental_language_dropdown,
                    mental_conversation_history,
                ],
                outputs=[
                    mental_input_transcript,
                    mental_counselor_output,
                    mental_audio_output,
                    emergency_alert,
                    severity_indicator,
                    mental_conversation_history,
                ],
            ).then(
                fn=lambda: (None, ""),  # Clear inputs after submission
                outputs=[mental_audio_input, mental_text_input]
            )
            
            mental_clear_btn.click(
                fn=clear_mental_conversation,
                outputs=[
                    mental_conversation_history,
                    mental_input_transcript,
                    mental_counselor_output,
                    mental_audio_output,
                    emergency_alert,
                    severity_indicator
                ]
            ).then(
                fn=lambda: (None, "", None),  # Also clear image, text, and audio inputs
                outputs=[mental_image_input, mental_text_input, mental_audio_input]
            )

    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    <div class="disclaimer">
    <em>Disclaimer: V.A.I.D. is an AI project for educational purposes. 
    <br>• For medical diagnosis: Always consult a real doctor for medical advice.
    <br>• For mental health: This is not a substitute for professional therapy. 
    <br>• In case of emergency: Please contact emergency services (911, 112, or your local emergency number) or a crisis hotline immediately.</em>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(debug=True)