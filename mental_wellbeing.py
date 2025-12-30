"""
MENTAL WELL-BEING AGENT - FIXED FOR GRADIO COMPATIBILITY
Fixed function signature to match what Gradio expects
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from groq import Groq
from doctor_brain import encode_image, analyze_image_with_query

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ============================================================================
# CONVERSATION MEMORY SYSTEM
# ============================================================================

class ConversationMemory:
    """Tracks and manages conversation context across sessions."""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.user_profile: Dict = {
            'emotion_patterns': {},
            'topics_discussed': [],
            'coping_preferences': [],
            'last_session_date': None,
            'progress_notes': []
        }
        self.session_context: Dict = {
            'current_mood': None,
            'primary_concern': None,
            'crisis_level': 'low',
            'therapeutic_goals': [],
            'recent_interventions': []
        }
    
    def add_exchange(self, user_msg: str, ai_response: str, emotions: Dict, crisis_level: str):
        """Add a conversation exchange to memory with context."""
        timestamp = datetime.now().isoformat()
        
        # Store the exchange
        exchange = {
            'timestamp': timestamp,
            'user': user_msg,
            'ai': ai_response,
            'emotions': emotions,
            'crisis_level': crisis_level
        }
        self.conversation_history.append(exchange)
        
        # Update user profile
        self._update_user_profile(user_msg, emotions, crisis_level)
        
        # Update session context
        self._update_session_context(user_msg, emotions, crisis_level)
        
        # Keep history manageable (last 20 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def _update_user_profile(self, user_msg: str, emotions: Dict, crisis_level: str):
        """Update user profile based on current interaction."""
        # Track emotional patterns
        for emotion, intensity in emotions.items():
            if emotion not in self.user_profile['emotion_patterns']:
                self.user_profile['emotion_patterns'][emotion] = []
            self.user_profile['emotion_patterns'][emotion].append({
                'intensity': intensity,
                'timestamp': datetime.now().isoformat()
            })
        
        # Extract and store topics
        topics = self._extract_topics(user_msg)
        for topic in topics:
            if topic not in self.user_profile['topics_discussed']:
                self.user_profile['topics_discussed'].append(topic)
        
        # Update last session date
        self.user_profile['last_session_date'] = datetime.now().isoformat()
    
    def _update_session_context(self, user_msg: str, emotions: Dict, crisis_level: str):
        """Update current session context."""
        # Update current mood based on dominant emotion
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            self.session_context['current_mood'] = dominant_emotion
        
        # Update primary concern if crisis level is moderate or higher
        if crisis_level in ['moderate', 'high_distress', 'crisis']:
            # Extract key concern from message
            concern = self._extract_primary_concern(user_msg)
            if concern and concern not in self.session_context['therapeutic_goals']:
                self.session_context['therapeutic_goals'].append(concern)
                self.session_context['primary_concern'] = concern
        
        # Track interventions used
        interventions = self._extract_interventions(user_msg)
        self.session_context['recent_interventions'].extend(interventions)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract relevant topics from text."""
        topics = []
        topic_keywords = {
            'work': ['work', 'job', 'career', 'boss', 'colleague', 'deadline'],
            'family': ['family', 'parent', 'child', 'sibling', 'spouse', 'partner'],
            'relationships': ['friend', 'relationship', 'dating', 'breakup', 'lonely'],
            'health': ['health', 'sleep', 'appetite', 'energy', 'pain'],
            'anxiety': ['worry', 'anxious', 'panic', 'overwhelm', 'stress'],
            'depression': ['sad', 'depressed', 'hopeless', 'empty', 'worthless'],
            'self_care': ['exercise', 'meditation', 'therapy', 'medication']
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics[:3]  # Return top 3 topics
    
    def _extract_primary_concern(self, text: str) -> Optional[str]:
        """Extract primary concern from text."""
        # Simple extraction - in production you'd use NLP
        concern_indicators = [
            'worried about', 'stressed about', 'anxious about', 
            'can\'t stop thinking about', 'concerned about',
            'problem with', 'issue with', 'struggling with'
        ]
        
        text_lower = text.lower()
        for indicator in concern_indicators:
            if indicator in text_lower:
                # Extract the phrase after the indicator
                start_idx = text_lower.find(indicator) + len(indicator)
                concern = text[start_idx:start_idx+50].strip()
                return concern[:100]  # Limit length
        
        return None
    
    def _extract_interventions(self, text: str) -> List[str]:
        """Extract interventions mentioned by user."""
        interventions = []
        intervention_keywords = {
            'breathing': ['breathe', 'breathing', 'inhale', 'exhale'],
            'grounding': ['grounding', '5 things', 'present moment', 'senses'],
            'exercise': ['exercise', 'walk', 'run', 'yoga', 'stretch'],
            'meditation': ['meditate', 'mindfulness', 'calm'],
            'journaling': ['journal', 'write', 'diary'],
            'therapy': ['therapy', 'counseling', 'therapist'],
            'medication': ['medication', 'meds', 'prescription']
        }
        
        text_lower = text.lower()
        for intervention, keywords in intervention_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                interventions.append(intervention)
        
        return interventions
    
    def get_conversation_summary(self, recent_exchanges: int = 5) -> str:
        """Get a summary of recent conversation."""
        if not self.conversation_history:
            return "First interaction."
        
        recent = self.conversation_history[-recent_exchanges:]
        summary_parts = []
        
        for i, exchange in enumerate(recent):
            user_msg_short = exchange['user'][:50] + "..." if len(exchange['user']) > 50 else exchange['user']
            emotions = exchange.get('emotions', {})
            if emotions:
                dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
                summary_parts.append(f"User mentioned: '{user_msg_short}' (emotion: {dominant})")
        
        return " | ".join(summary_parts)
    
    def get_user_patterns(self) -> str:
        """Get patterns observed in user's emotional state."""
        patterns = []
        
        # Check for recurring emotions
        for emotion, history in self.user_profile['emotion_patterns'].items():
            if len(history) >= 2:
                patterns.append(f"Frequent {emotion} experiences")
        
        # Check time patterns if we have enough data
        if len(self.conversation_history) >= 5:
            # Simple pattern detection
            if any('anxiety' in str(exchange.get('emotions', {})) for exchange in self.conversation_history[-5:]):
                patterns.append("Recent anxiety patterns")
        
        return ", ".join(patterns) if patterns else "No clear patterns yet"
    
    def reset_session(self):
        """Reset session-specific context but keep user profile."""
        self.session_context = {
            'current_mood': None,
            'primary_concern': None,
            'crisis_level': 'low',
            'therapeutic_goals': [],
            'recent_interventions': []
        }
        # Keep last 5 exchanges for continuity
        self.conversation_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
    
    def reset_all(self):
        """Completely reset all conversation history and memory."""
        self.conversation_history = []
        self.user_profile = {
            'emotion_patterns': {},
            'topics_discussed': [],
            'coping_preferences': [],
            'last_session_date': None,
            'progress_notes': []
        }
        self.session_context = {
            'current_mood': None,
            'primary_concern': None,
            'crisis_level': 'low',
            'therapeutic_goals': [],
            'recent_interventions': []
        }
        print("🔄 Memory completely reset")
    
    def get_memory_context(self) -> Dict:
        """Get comprehensive memory context for AI prompt."""
        return {
            'recent_summary': self.get_conversation_summary(3),
            'user_patterns': self.get_user_patterns(),
            'topics_discussed': self.user_profile['topics_discussed'][-5:],
            'current_mood': self.session_context['current_mood'],
            'primary_concern': self.session_context['primary_concern'],
            'recent_interventions': self.session_context['recent_interventions'][-3:],
            'therapeutic_goals': self.session_context['therapeutic_goals'][-3:],
            'total_exchanges': len(self.conversation_history)
        }

# Global conversation memory
memory = ConversationMemory()

# ============================================================================
# EMOTION & CRISIS DETECTION (Enhanced)
# ============================================================================

def analyze_emotional_state(text: str) -> Tuple[Dict, str, float]:
    """
    Comprehensive emotional state analysis.
    Returns: (emotions_dict, crisis_level, severity_score)
    """
    text_lower = text.lower()
    
    # Enhanced emotion detection
    emotions = {}
    
    # Anxiety detection
    anxiety_words = ['anxious', 'worried', 'panic', 'overwhelmed', 'racing thoughts', 
                    'can\'t breathe', 'heart racing', 'nervous', 'fear']
    anxiety_score = sum(2 for word in anxiety_words if word in text_lower)
    if anxiety_score > 0:
        emotions['anxiety'] = min(anxiety_score, 10)
    
    # Depression detection
    depression_words = ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 
                       'no energy', 'tired', 'exhausted', 'can\'t get up']
    depression_score = sum(2 for word in depression_words if word in text_lower)
    if depression_score > 0:
        emotions['depression'] = min(depression_score, 10)
    
    # Stress detection
    stress_words = ['stressed', 'pressure', 'overwhelmed', 'burnt out', 'too much',
                   'can\'t handle', 'swamped', 'drowning', 'deadline']
    stress_score = sum(2 for word in stress_words if word in text_lower)
    if stress_score > 0:
        emotions['stress'] = min(stress_score, 10)
    
    # Loneliness detection
    loneliness_words = ['lonely', 'alone', 'isolated', 'no one', 'nobody', 
                       'no friends', 'disconnected', 'abandoned']
    loneliness_score = sum(2 for word in loneliness_words if word in text_lower)
    if loneliness_score > 0:
        emotions['loneliness'] = min(loneliness_score, 10)
    
    # Anger detection
    anger_words = ['angry', 'furious', 'rage', 'irritated', 'frustrated', 
                  'annoyed', 'mad', 'livid', 'pissed']
    anger_score = sum(2 for word in anger_words if word in text_lower)
    if anger_score > 0:
        emotions['anger'] = min(anger_score, 10)
    
    # Positive emotions (score 0.5 each for more granular differentiation)
    positive_words = ['happy', 'good', 'great', 'better', 'improved', 'progress',
                     'grateful', 'thankful', 'hopeful', 'optimistic']
    positive_score = sum(0.5 for word in positive_words if word in text_lower)
    if positive_score > 0:
        emotions['positive'] = min(positive_score, 10)
    
    # Crisis level determination
    crisis_keywords = [
        'suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself',
        'self harm', 'cutting', 'overdose', 'no reason to live'
    ]
    
    high_distress_keywords = [
        'panic attack', 'severe anxiety', 'can\'t cope', 'breaking down',
        'losing control', 'can\'t breathe', 'heart attack'
    ]
    
    # Check for crisis
    if any(keyword in text_lower for keyword in crisis_keywords):
        crisis_level = 'crisis'
        severity = 10.0
    
    # Check for high distress
    elif any(keyword in text_lower for keyword in high_distress_keywords):
        crisis_level = 'high_distress'
        severity = 8.0
    
    # Determine based on emotion intensity
    elif emotions:
        # Check if ONLY positive emotions (no concern)
        if 'positive' in emotions and len(emotions) == 1:
            crisis_level = 'positive'
            severity = 0.0  # No concern for positive emotions
        else:
            # Filter out positive emotions to get actual concern level
            concern_emotions = {k: v for k, v in emotions.items() if k != 'positive'}
            
            if concern_emotions:
                max_intensity = max(concern_emotions.values())
                if max_intensity >= 8:
                    crisis_level = 'high_distress'
                    severity = float(max_intensity)
                elif max_intensity >= 5:
                    crisis_level = 'moderate'
                    severity = float(max_intensity)
                else:
                    crisis_level = 'low'
                    severity = float(max_intensity)
            else:
                # Only positive emotions present
                crisis_level = 'positive'
                severity = 0.0
    else:
        # No emotions detected - neutral/casual conversation
        crisis_level = 'neutral'
        severity = 0.0  # No concern for neutral conversation
    
    return emotions, crisis_level, severity

# ============================================================================
# OPTIONAL IMAGE EMOTION ANALYSIS (for uploaded user photos)
# ============================================================================

def analyze_image_emotion(image_path: str, language: str = "en") -> str:
    """
    Analyze an uploaded image (e.g., user photo) for emotional cues using Groq multimodal.
    This is OPTIONAL and only used if the user chooses to upload an image.
    """
    if not image_path:
        return ""

    lang_map = {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
    }
    target_language = lang_map.get(language, "English")

    prompt = f"""
You are a compassionate mental health assistant.

You are given a photo of a person. Based ONLY on their facial expression and body language:
- Describe what emotional state they APPEAR to be in (e.g., sad, anxious, stressed, angry, fearful, lonely, happy, neutral)
- Speak gently and non-judgmentally
- DO NOT guess age, gender, or physical appearance
- DO NOT mention that you are looking at a photo, just describe their emotional state
- Respond in {target_language} language
- Keep it to 1–2 sentences.
"""

    try:
        encoded = encode_image(image_path)
        description = analyze_image_with_query(
            query=prompt,
            encoded_image=encoded,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        if not description:
            return ""
        return str(description).strip()
    except Exception as e:
        print(f"❌ Image emotion analysis error: {e}")
        return ""

# ============================================================================
# AI PROMPT ENGINEERING - CONTEXT-AWARE
# ============================================================================

def build_therapeutic_prompt(user_input: str, memory_context: Dict, 
                           emotions: Dict, crisis_level: str, language: str) -> str:
    """
    Build a comprehensive therapeutic prompt that remembers everything.
    """
    
    # Language mapping
    lang_map = {
        "en": "English", "hi": "Hindi", "es": "Spanish", 
        "fr": "French", "de": "German"
    }
    target_language = lang_map.get(language, "English")
    
    # Extract key info from memory
    recent_summary = memory_context.get('recent_summary', '')
    user_patterns = memory_context.get('user_patterns', '')
    current_mood = memory_context.get('current_mood', '')
    primary_concern = memory_context.get('primary_concern', '')
    recent_interventions = memory_context.get('recent_interventions', [])
    therapeutic_goals = memory_context.get('therapeutic_goals', [])
    
    # Build context-aware guidance
    context_notes = []
    if recent_summary and "First interaction" not in recent_summary:
        context_notes.append(f"Previous conversation: {recent_summary}")
    
    if user_patterns:
        context_notes.append(f"Observed patterns: {user_patterns}")
    
    if current_mood:
        context_notes.append(f"User's recent mood trend: {current_mood}")
    
    if primary_concern:
        context_notes.append(f"Primary concern mentioned: {primary_concern}")
    
    if recent_interventions:
        context_notes.append(f"Recent coping strategies discussed: {', '.join(recent_interventions)}")
    
    if therapeutic_goals:
        context_notes.append(f"Therapeutic goals mentioned: {', '.join(therapeutic_goals)}")
    
    context_str = "\n".join(context_notes) if context_notes else "First conversation with this user."
    
    # Determine therapeutic approach based on crisis level
    therapeutic_approach = {
        'crisis': "CRISIS INTERVENTION: User is in immediate danger. Provide emergency contacts, ensure safety, and encourage immediate help. Be direct, urgent, and action-oriented.",
        'high_distress': "HIGH DISTRESS SUPPORT: User is experiencing intense distress. Focus on immediate regulation (breathing, grounding), validation, and safety planning. Use calming, present-focused language.",
        'moderate': "MODERATE SUPPORT: User is experiencing ongoing difficulties. Build on previous conversations, explore patterns, and develop coping strategies. Be empathetic and exploratory.",
        'general': "GENERAL SUPPORT: User is sharing concerns. Build rapport, explore gently, and offer support. Be conversational and curious.",
        'positive': "POSITIVE REINFORCEMENT: User is sharing progress. Celebrate achievements, reinforce positive behaviors, and explore what helped. Be encouraging and validating."
    }.get(crisis_level, "GENERAL SUPPORT")
    
    # Emotion-specific guidance
    emotion_guidance = ""
    if emotions:
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
        if dominant_emotion == 'anxiety':
            emotion_guidance = "User is experiencing anxiety. Focus on grounding techniques, present-moment awareness, and breaking down overwhelming thoughts."
        elif dominant_emotion == 'depression':
            emotion_guidance = "User is experiencing depression. Focus on small achievable steps, validation of difficulty, and gentle encouragement."
        elif dominant_emotion == 'stress':
            emotion_guidance = "User is experiencing stress. Help identify specific stressors and practical coping strategies."
        elif dominant_emotion == 'loneliness':
            emotion_guidance = "User is experiencing loneliness. Focus on connection (even small), validation of isolation feelings, and building social support."
    
    # Build the comprehensive prompt
    prompt = f"""You are a compassionate, skilled mental health counselor conducting a therapy session.

THERAPEUTIC APPROACH: {therapeutic_approach}
{emotion_guidance}

CONVERSATION HISTORY & CONTEXT:
{context_str}

CURRENT USER MESSAGE: "{user_input}"

USER'S CURRENT EMOTIONAL STATE: {emotions if emotions else 'Neutral/General'}
CRISIS LEVEL: {crisis_level.upper()}

YOUR TASK (respond in {target_language}):

1. **CONTINUITY FIRST**: Reference something from our previous conversation if relevant
2. **VALIDATE SPECIFICALLY**: Acknowledge exactly what they said, not generic "I hear you"
3. **BUILD ON HISTORY**: If we discussed coping strategies before, ask how they worked
4. **ASK THERAPEUTIC QUESTION**: One question that explores deeper or moves forward
5. **OFFER RELEVANT SUPPORT**: Based on their emotional state and history
6. **NATURAL CONVERSATION**: Sound like a real therapist, not a chatbot

EXAMPLE OF GOOD CONTINUITY:
- If they mentioned anxiety yesterday: "Yesterday you mentioned feeling anxious about work. How has that anxiety been today?"
- If they tried breathing exercises: "Last time we talked about breathing exercises. Did you try any, and how did they feel?"
- If they're repeating patterns: "I notice you often mention feeling overwhelmed in the evenings. Is that happening again today?"

ABSOLUTELY DO NOT:
- Use generic phrases like "I'm here to listen" or "Your feelings are valid"
- Forget what was discussed previously
- Ask disconnected questions
- Give the same response to different situations

CURRENT SITUATION ANALYSIS:
Based on their message "{user_input[:100]}..." and our history, what's the most therapeutic next step?

YOUR RESPONSE (in {target_language}, 3-5 sentences, conversational):"""
    
    return prompt

# ============================================================================
# MAIN THERAPEUTIC AGENT - FIXED SIGNATURE
# ============================================================================

def analyze_mental_wellbeing(user_input, conversation_history=None, language="en"):
    """
    FIXED: Main function for mental wellbeing analysis.
    Signature matches what Gradio expects: (user_input, conversation_history, language)
    
    Returns:
        tuple: (response_text, is_emergency, severity_score)
    """
    # Convert to string if not already
    if user_input is None:
        return "I noticed you didn't share anything. Is there something on your mind you'd like to talk about?", False, 1.0
    
    user_input = str(user_input).strip()
    
    if not user_input:
        return "I'd like to understand what you're going through. Can you tell me more?", False, 1.0
    
    print(f"🧠 Processing mental wellbeing request: '{user_input[:50]}...'")
    
    # Check if conversation_history from Gradio is empty - if so, reset global memory
    if not conversation_history or len(conversation_history) == 0:
        print("🔄 Gradio conversation history is empty - resetting global memory")
        memory.reset_all()
    
    # Step 1: Analyze emotional state
    try:
        emotions, crisis_level, severity = analyze_emotional_state(user_input)
    except Exception as e:
        print(f"❌ Emotion analysis error: {e}")
        emotions, crisis_level, severity = {}, 'general', 3.0
    
    # Step 2: Check for immediate crisis
    if crisis_level == 'crisis':
        print("🚨 Crisis detected!")
        emergency_response = get_emergency_response(language)
        # Still add to memory for continuity
        memory.add_exchange(user_input, "EMERGENCY RESPONSE TRIGGERED", emotions, crisis_level)
        return emergency_response, True, severity
    
    # Step 3: Get memory context
    try:
        memory_context = memory.get_memory_context()
    except Exception as e:
        print(f"❌ Memory context error: {e}")
        memory_context = {}
    
    # Step 4: Build therapeutic prompt
    try:
        prompt = build_therapeutic_prompt(user_input, memory_context, emotions, crisis_level, language)
    except Exception as e:
        print(f"❌ Prompt building error: {e}")
        return generate_fallback_response(user_input, emotions, crisis_level, memory_context, language), False, severity
    
    # Step 5: Generate AI response
    ai_response = None
    
    # Check if API key is available
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY not found in environment variables!")
        print("   Please set GROQ_API_KEY in your environment or .env file")
        ai_response = generate_fallback_response(user_input, emotions, crisis_level, memory_context, language)
    else:
        try:
            print(f"🤖 Calling Groq API with {len(prompt)} chars prompt")
            print(f"   API Key present: {GROQ_API_KEY[:10]}...{GROQ_API_KEY[-5:] if len(GROQ_API_KEY) > 15 else '***'}")
            client = Groq(api_key=GROQ_API_KEY)
            
            # Build conversation for AI
            messages = [{"role": "system", "content": prompt}]
            
            # Add recent conversation history from memory
            recent_history = memory.conversation_history[-6:]  # Last 3 pairs
            for exchange in recent_history:
                messages.append({"role": "user", "content": exchange['user']})
                messages.append({"role": "assistant", "content": exchange['ai']})
            
            messages.append({"role": "user", "content": user_input})
            
            print(f"   Sending {len(messages)} messages to Groq API...")
            
            # Get AI response
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=400,
                top_p=0.9
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"✅ AI Response generated: {ai_response[:100]}...")
            
            # Ensure response is therapeutic and not generic
            if ai_response:
                ai_response = ensure_therapeutic_quality(ai_response, user_input, memory_context)
                print(f"✅ Response after quality check: {ai_response[:100]}...")
            else:
                print("⚠️ AI returned empty response")
    
        except Exception as e:
            print(f"❌ AI Generation Error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback response that still shows continuity
            ai_response = generate_fallback_response(user_input, emotions, crisis_level, memory_context, language)
    
    # Ensure we always have a response
    if not ai_response or not ai_response.strip():
        print("⚠️ No AI response generated, using fallback")
        ai_response = generate_fallback_response(user_input, emotions, crisis_level, memory_context, language)
    
    # Step 6: Update memory with this exchange
    try:
        memory.add_exchange(user_input, ai_response, emotions, crisis_level)
    except Exception as e:
        print(f"❌ Memory update error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 7: Check if we need to adapt based on conversation patterns
    try:
        if ai_response:
            ai_response = adapt_based_on_patterns(ai_response, memory)
    except Exception as e:
        print(f"❌ Pattern adaptation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final safety check - ensure we always return a valid response
    if not ai_response or not ai_response.strip():
        ai_response = f"I hear you saying '{user_input[:50]}...'. Can you tell me more about what's on your mind?"
    
    # Ensure severity is a number (float or int)
    if not isinstance(severity, (int, float)):
        severity = float(severity) if severity else 3.0
    
    return str(ai_response), False, float(severity)

def ensure_therapeutic_quality(response: str, user_input: str, memory_context: Dict) -> str:
    """Ensure response is therapeutic and not generic."""
    
    # Safety check
    if not response or not isinstance(response, str):
        return f"I hear you saying '{user_input[:50]}...'. Can you tell me more about what's on your mind?"
    
    response = response.strip()
    if not response:
        return f"I hear you saying '{user_input[:50]}...'. Can you tell me more about what's on your mind?"
    
    # Check for generic responses
    generic_phrases = [
        "i'm here to listen",
        "i'm here to support you",
        "your feelings are valid",
        "thank you for sharing that with me",
        "i appreciate you opening up",
        "i understand how you feel",
        "that must be difficult"
    ]
    
    # If response starts with generic phrase, enhance it
    response_lower = response.lower()
    for phrase in generic_phrases:
        if response_lower.startswith(phrase):
            # Enhance with specific reference
            specific_part = response[len(phrase):].strip()
            if not specific_part:
                # Create specific reference to user's input
                key_words = [word for word in user_input.split() if len(word) > 4]
                if key_words:
                    specific_ref = f"The way you described '{key_words[0]}'"
                    return f"{specific_ref} {response}"
            break
    
    # Ensure response acknowledges user input specifically
    user_words = set(user_input.lower().split()[:10])  # First 10 words
    response_words = set(response.lower().split())
    overlap = len(user_words.intersection(response_words))
    
    if overlap < 1 and len(user_input.split()) > 3:
        # Response doesn't reference user's words - add specific reference
        key_word = next((w for w in user_input.split() if len(w) > 4), None)
        if key_word:
            return f"When you mention '{key_word}', {response}"
    
    return response

def generate_fallback_response(user_input: str, emotions: Dict, crisis_level: str, 
                             memory_context: Dict, language: str) -> str:
    """Generate a fallback response that maintains continuity."""
    
    # Get conversation continuity
    recent_summary = memory_context.get('recent_summary', '')
    current_mood = memory_context.get('current_mood', '')
    
    # Build continuity reference
    continuity = ""
    if "First interaction" not in recent_summary and current_mood:
        continuity = f"Continuing from our conversation about {current_mood}, "
    elif "First interaction" not in recent_summary:
        continuity = "Building on what we've discussed, "
    
    # Crisis-level specific fallbacks
    if crisis_level == 'high_distress':
        if 'anxiety' in emotions:
            return f"{continuity}that anxiety sounds really intense right now. Let's focus on your breathing - can you take a slow breath in with me?"
        else:
            return f"{continuity}this sounds really overwhelming. What's one small thing that might help you feel more grounded right now?"
    
    elif crisis_level == 'moderate':
        if emotions:
            dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
            if dominant:
                return f"{continuity}that {dominant} sounds difficult. What's been the most challenging part about it today?"
            else:
                return f"{continuity}what's been on your mind the most since we last talked?"
        else:
            return f"{continuity}what's been on your mind the most since we last talked?"
    
    else:  # general or positive
        return f"{continuity}thanks for sharing. What would you like to focus on today?"

def adapt_based_on_patterns(response: str, memory: ConversationMemory) -> str:
    """Adapt response based on observed conversation patterns."""
    
    # Safety check
    if not response or not isinstance(response, str):
        return "I'm here to help. Can you tell me more about what's on your mind?"
    
    original_response = response  # Keep original in case of errors
    
    try:
        # Check if we're repeating interventions too often
        recent_interventions = memory.session_context.get('recent_interventions', [])
        if len(recent_interventions) >= 3:
            last_three = recent_interventions[-3:]
            # If we've suggested the same thing 3 times in a row
            if len(set(last_three)) == 1:
                intervention = last_three[0]
                # Add variety
                if intervention == 'breathing':
                    response += " If breathing exercises aren't helping today, we could also try a different approach like journaling about what's coming up for you."
                elif intervention == 'grounding':
                    response += " If grounding isn't resonating right now, sometimes focusing on physical movement can help too."
        
        # Check if we're stuck on the same topic
        topics = memory.user_profile.get('topics_discussed', [])
        if len(topics) >= 5:
            recent_topics = topics[-3:]
            if len(set(recent_topics)) == 1:
                topic = recent_topics[0]
                response += f" I notice we've been discussing {topic} quite a bit. Would it be helpful to explore any other areas of your life that might be affecting this?"
    except Exception as e:
        print(f"⚠️ Pattern adaptation skipped: {e}")
        # Return original response if adaptation fails
        return original_response if 'original_response' in locals() else response
    
    # Final safety check
    if not response or not isinstance(response, str) or not response.strip():
        return original_response if 'original_response' in locals() else "I'm here to help. Can you tell me more about what's on your mind?"
    
    return response

# ============================================================================
# OTHER REQUIRED FUNCTIONS FOR GRADIO IMPORT
# ============================================================================

def detect_emergency(user_input):
    """Detect if user input contains emergency keywords."""
    if not user_input:
        return False
    user_input_lower = str(user_input).lower()
    
    crisis_keywords = [
        'suicide', 'kill myself', 'end my life', 'want to die',
        'hurt myself', 'self harm', 'cutting', 'overdose'
    ]
    
    return any(keyword in user_input_lower for keyword in crisis_keywords)

def get_emergency_response(language="en"):
    """Get emergency response message in the specified language."""
    emergency_responses = {
        "en": """🚨 EMERGENCY ALERT 🚨

Your safety is the most important thing right now. Please reach out for immediate help:

📞 Emergency Services: 108 or 911
📞 Suicide Prevention Lifeline: 988
💬 Crisis Text Line: Text HOME to 741741

You are not alone. People want to help you through this difficult moment.
Please contact emergency services immediately.""",
        
        "hi": """🚨 आपातकालीन चेतावनी 🚨

आपकी सुरक्षा सबसे महत्वपूर्ण है। कृपया तुरंत मदद के लिए संपर्क करें:

📞 आपातकालीन सेवाएं: 108
📞 मानसिक स्वास्थ्य हेल्पलाइन: 1800-599-0019

आप अकेले नहीं हैं। लोग इस कठिन समय में आपकी मदद करना चाहते हैं।
कृपया तुरंत आपातकालीन सेवाओं से संपर्क करें।"""
    }
    
    return emergency_responses.get(language, emergency_responses["en"])

def reset_conversation():
    """Reset conversation memory completely."""
    try:
        memory.reset_all()
        print("✅ Conversation memory completely cleared")
        return "Conversation has been reset. I'm ready to listen."
    except Exception as e:
        print(f"❌ Reset conversation error: {e}")
        import traceback
        traceback.print_exc()
        # Force reset even if there's an error
        memory.conversation_history = []
        memory.user_profile = {
            'emotion_patterns': {},
            'topics_discussed': [],
            'coping_preferences': [],
            'last_session_date': None,
            'progress_notes': []
        }
        memory.session_context = {
            'current_mood': None,
            'primary_concern': None,
            'crisis_level': 'low',
            'therapeutic_goals': [],
            'recent_interventions': []
        }
        return "Conversation reset."

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_functionality():
    """Test that all functions work correctly."""
    print("🧠 Testing mental wellbeing module...")
    
    # Test the actual function signature
    test_input = "I'm feeling anxious about work"
    print(f"\nTest 1: '{test_input}'")
    
    # Call with the correct signature that Gradio uses
    response, emergency, severity = analyze_mental_wellbeing(
        user_input=test_input,
        conversation_history=[],  # This parameter is accepted but not used
        language="en"
    )
    
    print(f"Response: {response[:100]}...")
    print(f"Emergency: {emergency}, Severity: {severity}")
    print("\n✅ Function test successful!")

if __name__ == "__main__":
    test_functionality()