# 🏥 V.A.I.D. (Virtual AI Doctor)

> **Intelligent Health Analysis with Vision & Voice**
> *A Multilingual AI Doctor powered by Llama 4, Whisper, and ElevenLabs.*

## 📖 Overview

**V.A.I.D.** (Virtual AI Doctor) is an advanced telemedicine prototype designed to bridge the gap between patients and medical advice using multimodal AI. It simulates a real doctor-patient interaction by combining vision, voice, and intelligent analysis to provide comprehensive healthcare support.

### Core Capabilities
- **👁️ Vision:** AI-powered image analysis for symptom diagnosis
- **🗣️ Voice:** Multilingual speech recognition (Hindi, English, Spanish, French, German)
- **🔊 Speech:** Human-like voice responses via ElevenLabs
- **💬 Text Support:** Fallback text chat option
- **🧠 Dual Functionality:** Medical diagnosis + Mental health counseling

---

## 🛠️ Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Medical AI Brain** | **Llama-4-Scout-17b** (via Groq Cloud) |
| **Mental Health AI Brain** | **Llama-3.3-70b-Versatile** (via Groq Cloud) |
| **Speech Recognition (STT)** | **Whisper-Large-v3** (via Groq Cloud) |
| **Voice Synthesis (TTS)** | **ElevenLabs Multilingual v2** |
| **Frontend Framework** | **Gradio (Python)** |
| **Audio Processing** | FFmpeg, PyDub, SpeechRecognition |





---

## 📂 Project Structure
```
V.A.I.D.---Virtual-AI-Doctor-main/
├── gradio_app.py          # Main application with dual-tab interface
├── doctor_brain.py        # Medical diagnosis AI (multimodal LLM)
├── doctor_voice.py        # Text-to-speech functionality
├── patient_voice.py       # Speech-to-text functionality
├── mental_wellbeing.py    # Mental health counseling AI with emergency detection
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```
## 🔹 Key Modules
**1. doctor_brain.py**
- Multimodal LLM for medical image analysis
- Integrates visual and textual symptom data
- Uses Groq API with Llama-4-Scout model
- Generates structured medical assessments

**2. patient_voice.py**
- Speech-to-text conversion using Whisper
- Multi-language audio processing
- Handles various audio formats and quality levels

**3. doctor_voice.py**
- Text-to-speech synthesis via ElevenLabs
- Natural-sounding multilingual voice output
- Fallback to gTTS for cost-effective alternatives

**4. mental_wellbeing.py**
- AI-powered mental health counseling engine
- Six-category emotion detection system
- Crisis-level classification algorithm
- Emergency keyword detection and safety protocols
- Severity assessment on 0-10 scale
- Optional image-based emotion analysis
- Conversation memory management
- Uses Groq API with Llama-3.3-70b-Versatile model
---

# 🩺 MEDICAL DIAGNOSIS

## 🔹Features

### 1. Multimodal Analysis
   - **Image-Based Diagnosis:** Upload photos of symptoms for AI-powered visual analysis
   - **Voice Input:** Describe symptoms using voice recording in multiple languages
   - **Text Input:** Type symptoms directly as an alternative to voice
   - **Combined Assessment:** Integrates visual and verbal information for comprehensive diagnosis

### 2.Intelligent Processing
   - **Real-time Analysis:** Fast inference using Groq's Language Processing Unit (LPU)
   - **Multilingual Support:** Understands and responds in English, Hindi, Spanish, French, and German
   - **Professional Output:** Clear, structured medical advice with audio responses
   - **Context-Aware:** Considers both uploaded images and patient descriptions

## 🔹How It Works



1. **Input Collection**
   - User uploads symptom image (optional)
   - User speaks or types symptoms in preferred language

2. **AI Processing**
   - System transcribes voice using Whisper v3
   - Builds comprehensive prompt combining image + text
   - Groq multimodal LLM (Llama-4-Scout) analyzes data

3. **Response Generation**
   - AI generates detailed diagnosis text
   - ElevenLabs/gTTS converts text to natural speech
   - UI displays transcript, diagnosis, and plays audio

4. **Output**
   - Written diagnosis with recommendations
   - Audio explanation in user's preferred language
   - Visual symptom analysis results
   <img width="1000" height="569" alt="V.A.I.D. Architecture" src="https://github.com/user-attachments/assets/20c54acb-c0a8-4d4b-ad07-3d4130fe167e" />


## 🔹Demo Screenshots

- ### English Interface
   <img width="1896" height="880" alt="Medical Diagnosis - English UI" src="https://github.com/user-attachments/assets/622a8fbd-7792-4902-9912-175e0db636cc" />

- ### Hindi Interface
   <img width="1481" height="771" alt="Medical Diagnosis - Hindi UI" src="https://github.com/user-attachments/assets/72ba1721-3bfe-4978-a15f-d435d810d9be" />




---

# 🧠 MENTAL WELLBEING

## 🔹 Features

- ### AI-Powered Counseling
   - **Compassionate Support:** Context-aware mental health assistance with personalized responses
   - **No Generic Replies:** Each response is tailored to the user's specific emotional state
   - **Conversation Memory:** Maintains context across multiple interactions for continuity
   - **Optional Photo Analysis:** Upload a selfie for emotion detection from facial cues
   - **Multilingual Support:** Available in English, Hindi, Spanish, French, and German

- ### Advanced Detection Systems

   - #### Emotion Recognition
      The system identifies six key emotional states:
      - **Anxiety** - Worry, nervousness, panic
      - **Depression** - Sadness, hopelessness, emptiness
      - **Stress** - Overwhelm, pressure, tension
      - **Loneliness** - Isolation, disconnection
      - **Anger** - Frustration, irritability, rage
      - **Positive** - Happiness, contentment, hope
   
   - #### Crisis Level Classification
      Real-time assessment of mental state severity:
      - **Neutral** - Casual conversation, no distress
      - **Low Distress** - Minor concerns, manageable stress
      - **Moderate Distress** - Noticeable emotional difficulty
      - **High Distress** - Significant emotional pain
      - **Crisis** - Immediate danger, suicidal ideation
      - **Positive** - Good mood, constructive mindset

- ### Emergency Response System

   - #### Automatic Detection
      - **Keyword Monitoring:** Scans for suicidal ideation, self-harm mentions
      - **Severity Scoring:** Real-time assessment (0-10 scale) displayed via slider
      - **Immediate Intervention:** Triggers emergency banner for crisis situations
      - **Safety Resources:** Provides instant access to crisis hotlines
   
   - #### Crisis Response Features
      - Automatic display of emergency contact information
      - Non-judgmental, supportive messaging
      - Encouragement to seek professional help

## 🔹 How It Works



1. **Input Collection**
   - User optionally uploads a selfie for emotion analysis
   - User speaks or types their feelings/concerns

2. **AI Analysis**
   - System transcribes voice input (if provided)
   - Analyzes text for emotional content and crisis indicators
   - If photo provided: Extracts emotional cues from facial expression
   - Combines text + photo context into comprehensive prompt

3. **Intelligent Processing**
   - Groq LLM (Llama-3.3-70b-Versatile) generates personalized response
   - Emotion detection classifies emotional state
   - Crisis detector assesses severity level
   - System updates severity slider (0-10 scale)

4. **Response Delivery**
   - Context-aware, empathetic text response
   - Optional audio reply via text-to-speech
   - Emergency banner if crisis detected
   - Conversation history maintained for continuity

5. **User Control**
   - Clear history button to reset conversation
   - Auto-reset on page refresh
   - Option to continue conversation with full context

   <img width="753" height="1024" alt="Medical Diagnosis Flow Chart" src="https://github.com/user-attachments/assets/196eda4c-cd9d-47e4-ba2a-cf2511a014a6" />

## 🔹Demo Screenshot

   <img width="1697" height="861" alt="Mental Wellbeing Interface" src="https://github.com/user-attachments/assets/c09a0d06-c4aa-4913-8c64-6c7b43d7c30b" />



---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.8 or higher**
- **API Keys:**
  - `GROQ_API_KEY` (Required) - Get free key at [console.groq.com](https://console.groq.com/)
  - `ELEVENLABS_API_KEY` (Required) - Get key at [elevenlabs.io](https://elevenlabs.io/)

### Installation Steps

1. **Clone the repository**
```bash
   git clone <repository-url>
   cd V.A.I.D.---Virtual-AI-Doctor-main
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
```env
   GROQ_API_KEY=your_groq_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```
   
   Alternatively, set them in your system environment variables.

4. **Run the application**
```bash
   python gradio_app.py
```

5. **Access the interface**
   - The app will launch at `http://127.0.0.1:7860`
   - Open in your browser to access the dual-tab interface

---

## 🚨 Emergency Resources

### 🔹 India Helplines

- #### Mental Health Crisis
   - **AASRA (24/7 Suicide Prevention):** +91-9820466726
   - **Vandrevala Foundation (24/7):** 1860-2662-345 or 1800-2333-330
   - **iCall Psychosocial Helpline:** +91-9152987821 (Mon-Sat, 8 AM - 10 PM)
   - **NIMHANS Helpline (Bangalore):** +91-80-46110007
   - **Sumaitri (Delhi, 2 PM - 10 PM):** +91-11-23389090
   - **Connecting NGO (Pune):** +91-20-24136953

- #### Medical Emergency
   - **National Emergency Number:** 112
   - **Ambulance:** 102 or 108
   - **Women's Helpline:** 1091
   - **Child Helpline:** 1098

### 🔹 International Helplines

- #### United States
   - **Emergency Services:** 911
   - **Suicide Prevention Lifeline:** 988 or 1-800-273-8255
   - **Crisis Text Line:** Text HOME to 741741

- #### European Union
   - **Emergency Services:** 112

- #### United Kingdom
   - **Emergency Services:** 999 or 112
   - **Samaritans:** 116 123

---

## ⚠️ Important Disclaimer

**V.A.I.D. is an AI project developed for educational and research purposes only.**

### Medical Diagnosis
- ❌ This tool is **NOT** a substitute for professional medical advice, diagnosis, or treatment
- ❌ Do **NOT** use this for emergency medical situations
- ✅ Always seek the advice of your physician or qualified health provider with any questions regarding a medical condition
- ✅ Never disregard professional medical advice or delay seeking it because of information from this tool

### Mental Wellbeing
- ❌ This tool is **NOT** a substitute for professional therapy, psychiatric care, or counseling
- ❌ Do **NOT** rely on this tool during active mental health crises
- ✅ In case of emergency or crisis, immediately contact emergency services or crisis hotlines listed above
- ✅ Always consult qualified mental health professionals for proper diagnosis and treatment

### General Warning
- AI-generated advice may contain errors or inaccuracies
- This tool should be used as a supplementary resource only
- Real medical and mental health professionals have training, experience, and ethical obligations that AI cannot replace

**If you are experiencing a medical or mental health emergency, call emergency services immediately.**

---


**Remember: Your health and safety are paramount. Always seek professional help when needed.**
