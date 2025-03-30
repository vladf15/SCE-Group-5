import json
import os
import time
import threading
from datetime import datetime
import speech_recognition as sr
import pyttsx3
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


###HELPER FUNCTIONS####
def normalize_text(text):
    """Normalize text for better matching by removing common words and punctuation"""
    text = text.lower()
    # Remove punctuation
    for char in ",.!?;:\"'()[]{}":
        text = text.replace(char, " ")
    
    # Remove common words that don't help with topic matching
    stopwords = ["the", "and", "is", "in", "to", "a", "of", "for", "with", "about", 
                "that", "on", "at", "this", "my", "i", "me", "you", "would", "like", 
                "want", "please", "could", "can", "do", "tell", "give", "need", "show"]
    
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    return filtered_words

def get_similarity_score(topic_words, input_words):
    """Calculate similarity between topic and input based on word overlap"""
    if not topic_words or not input_words:
        return 0
    
    matches = sum(1 for word in topic_words if word in input_words)
    # Consider both directions to handle cases where input is longer/shorter than topic
    score1 = matches / len(topic_words) if topic_words else 0
    score2 = matches / len(input_words) if input_words else 0
    
    # Average of both directional scores
    return (score1 + score2) / 2


###TTS initialization###
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 200)  # rate of speech
    tts_engine.setProperty('volume', 0.8)  #volume (0.0 to 1.0)
    #set voice
    voices = tts_engine.getProperty('voices')
    voice_name = "hazel"
    for voice in voices:
        if(voice_name in voice.name.lower()):
            tts_engine.setProperty('voice', voice.id)
            print("Voice set to: ", voice.name)
            tts_engine.runAndWait()
            tts_engine.stop()
            
    tts_enabled = True
    print("Text-to-speech initialized successfully")
    
    
except Exception as e:
    tts_enabled = False
    print(f"Text-to-speech initialization failed: {e}")
    print("The program will continue without speech output")


def format_for_speech(text):
    """Format text to sound more natural when spoken"""
    text = text.replace("1.", "1,")
    text = text.replace("2.", "2,")
    text = text.replace("3.", "3,")
    text = text.replace("4.", "4,")
    text = text.replace("5.", "5,")
    text = text.replace("\n1.", ".\n1,")
    text = text.replace("\n2.", ".\n2,")
    text = text.replace("\n3.", ".\n3,")
    text = text.replace("\n4.", ".\n4,")
    text = text.replace("?", ", ?")
    text = text.replace(" and ", ", and ")
    text = text.replace(" but ", ", but ")
    text = text.replace(" or ", ", or ")
    #remove special characters
    text = text.replace('*', '')
    text = text.replace('#', '')
    text = text.replace('_', '')
    while "  " in text:
        text = text.replace("  ", " ")
        
    return text

def speak_text(text, interrupt=True):
    """Convert text to speech"""
    if not tts_enabled:
        return
    
    if interrupt and tts_engine.isBusy():
        tts_engine.stop()
        
    text = format_for_speech(text)
    tts_engine.say(text)
    tts_engine.runAndWait()
    
    
    
# Add a lock to prevent multiple simultaneous TTS operations
import threading
tts_lock = threading.Lock()

def speak_text_async(text, interrupt=True):
    """Convert text to speech in a non-blocking way"""
    if not tts_enabled:
        return
        
    # Create a thread to handle the speech
    def speak_in_thread():
        try:
            with tts_lock:  # Use a lock to prevent multiple simultaneous TTS operations
                if interrupt and tts_engine.isBusy():
                    tts_engine.stop()
                    
                formatted_text = format_for_speech(text)
                tts_engine.say(formatted_text)
                tts_engine.runAndWait()
        except RuntimeError as e:
            # This can happen if runAndWait is called while already running
            print(f"TTS engine error: {e}")
        except Exception as e:
            print(f"Unexpected TTS error: {e}")
    
    # Start the speech in a background thread
    threading.Thread(target=speak_in_thread, daemon=True).start()


# Initialize LLM
try:
    print("Initializing language model...")
    model = ChatOllama(model="llama2:7b-chat-q4_K_M")
    print("Language model initialized successfully")
except Exception as e:
    print(f"Error initializing language model: {str(e)}")
    print("Make sure Ollama is installed and running on your system")

#Speech recognition library
recognizer = sr.Recognizer()
recognizer.pause_threshold = 2  # Seconds of silence before considering the phrase complete

#Conversation topic categories
ALZHEIMERS_TOPIC_CATEGORIES = [
    "Art events from the 80s",
    "Political state of the world in your youth",
    "How has technology changed the way we communicate?",
    "Family traditions from youth",
    "How did you celebrate holidays when you were younger compared to how families celebrate today?",
    "Favourite places to visit in the past"
    ]

class TopicManager:
    def __init__(self, save_file="topic_data.json"):
        self.save_file = save_file
        self.topic_ratings = {}  # topic -> list of effectiveness ratings
        self.topic_history = []  # chronological list of used topics
        self.load_data()
    
    def load_data(self):
        """Load topic data from file if it exists"""
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'r') as f:
                    data = json.load(f)
                    self.topic_ratings = data.get('ratings', {})
                    self.topic_history = data.get('history', [])
                print(f"Loaded data for {len(self.topic_ratings)} topics")
            except Exception as e:
                print(f"Error loading topic data: {e}")
    
    def save_data(self):
        """Save topic data to file"""
        with open(self.save_file, 'w') as f:
            json.dump({
                'ratings': self.topic_ratings,
                'history': self.topic_history
            }, f, indent=2)
        print("Topic data saved successfully")
    
    def add_topic_rating(self, topic, effectiveness):
        """Record a new effectiveness rating for a topic, matching to existing topics where possible"""
        # First, check if this exact topic already exists
        if topic in self.topic_ratings:
            self.topic_ratings[topic].append(effectiveness)
            matched_topic = topic
        else:
            # Get all existing topics (both from ratings and predefined categories)
            all_existing_topics = list(self.topic_ratings.keys()) + ALZHEIMERS_TOPIC_CATEGORIES
            all_existing_topics = list(set(all_existing_topics))  # Remove duplicates
            
            # Try to find a matching topic
            input_words = normalize_text(topic)
            best_match = None
            best_score = 0.3  # Minimum threshold for considering a match
            
            for existing_topic in all_existing_topics:
                existing_words = normalize_text(existing_topic)
                similarity = get_similarity_score(existing_words, input_words)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = existing_topic
            
            # If we found a matching topic, use it instead
            if best_match:
                print(f"Matched '{topic}' to existing topic: '{best_match}'")
                speak_text(f"I've matched this to the existing topic: {best_match}")
                
                # Initialize if this is a predefined topic not yet rated
                if best_match not in self.topic_ratings:
                    self.topic_ratings[best_match] = []
                    
                self.topic_ratings[best_match].append(effectiveness)
                matched_topic = best_match
            else:
                # No match found - create a new topic
                print(f"Creating new topic: '{topic}'")
                self.topic_ratings[topic] = [effectiveness]
                matched_topic = topic
        
        # Record in history with the matched or new topic
        self.topic_history.append({
            "topic": matched_topic, 
            "effectiveness": effectiveness,
            "timestamp": datetime.now().isoformat()
        })
        
        self.save_data()
        return matched_topic  # Return the matched topic for feedback
        
    def get_average_rating(self, topic):
        """Calculate average effectiveness rating for a topic"""
        if topic in self.topic_ratings and self.topic_ratings[topic]:
            return sum(self.topic_ratings[topic]) / len(self.topic_ratings[topic])
        return 0
    
    def get_top_topics(self, count=3):
        """Get the most effective conversation topics"""
        topic_averages = [(topic, self.get_average_rating(topic)) 
                         for topic in self.topic_ratings.keys()]
        sorted_topics = sorted(topic_averages, key=lambda x: x[1], reverse=True)
        return sorted_topics[:count]
    
    def get_recent_topics(self, count=3):
        """Get most recently used topics"""
        recent = []
        for item in reversed(self.topic_history):
            if item["topic"] not in [t for t in recent]:
                recent.append(item["topic"])
            if len(recent) >= count:
                break
        return recent
    
    def suggest_topics(self, show_all=True):
        """Suggest topics based on effectiveness and avoiding recent ones
        
        If show_all is True, returns all topics ordered by effectiveness
        """
        if show_all:
            # Get all topics, prioritizing ones with good ratings
            all_topics = set()
            
            # First add topics with ratings (ordered by effectiveness)
            if self.topic_ratings:
                rated_topics = [(topic, self.get_average_rating(topic)) 
                            for topic in self.topic_ratings.keys()]
                sorted_rated = sorted(rated_topics, key=lambda x: x[1], reverse=True)
                for topic, _ in sorted_rated:
                    all_topics.add(topic)
            
            # Then add all predefined categories
            for category in ALZHEIMERS_TOPIC_CATEGORIES:
                all_topics.add(category)
                
            # Convert to list
            return list(all_topics)
        else:
            # Original limited suggestion logic
            count = 3
            if self.topic_ratings:
                top_topics = self.get_top_topics(count=count+2)
                recent_topics = self.get_recent_topics(count=count)
                
                # Filter out recently used topics
                suggestions = [topic for topic, rating in top_topics 
                            if topic not in recent_topics][:count]
                
                # If we need more suggestions, add some from the predefined categories
                if len(suggestions) < count:
                    for category in ALZHEIMERS_TOPIC_CATEGORIES:
                        if category not in suggestions and category not in recent_topics:
                            suggestions.append(category)
                            if len(suggestions) >= count:
                                break
                
                return suggestions[:count]
            
            # If no ratings yet, suggest from predefined categories
            return ALZHEIMERS_TOPIC_CATEGORIES[:count]
    
    def generate_specific_questions(self, topic):
        """Generate specific questions for a topic using LLM"""
        try:
            print(f"Preparing prompt for topic: {topic}")
            prompt = ChatPromptTemplate.from_template(
                """You are assisting a person with Alzheimer's.
                Generate 3 specific, engaging questions about the topic: {topic}
                
                These questions should:
                1. Focus on long-term memories (which are better preserved in Alzheimer's)
                2. Be simple, brief and clear
                3. Evoke positive emotions when possible
                4. Not test memory or recent events
                5. Do not explain each suggestion, just provide the questions
                
                Format as a numbered list. Do not preface the suggestions. Do not add extra information.
                """
            )
            
            message = prompt.format_messages(topic=topic)
            print("Sending request to language model...")
            
            # Check if Ollama is running and accessible
            try:
                response = model.invoke(message)
                print("Response received")
                return response.content
            except Exception as e:
                print(f"Error invoking language model: {str(e)}")
                # Fallback to hardcoded questions if LLM fails
                return self._get_fallback_questions(topic)
        except Exception as e:
            print(f"Unexpected error in generate_specific_questions: {str(e)}")
            return self._get_fallback_questions(topic)
            
    def _get_fallback_questions(self, topic):
        """Provide fallback questions if LLM fails"""
        print("Using fallback questions")
        return f"""1. What are your favorite memories related to {topic}?
    2. How did {topic} play a role in your early life?
    3. What feelings come to mind when you think about {topic}?"""

def listen_to_conversation(duration=60):
    """Listen to a conversation for the specified duration and transcribe it"""
    print(f"Listening to conversation for {duration} seconds...")
    
    # Break recording into smaller chunks with OVERLAP to avoid cutting sentences
    chunk_size = 13  # Base chunk size
    overlap = 3      # Seconds of overlap between chunks
    
    # Calculate effective chunk progression (how much we advance each chunk)
    chunk_advance = chunk_size - overlap
    
    # Calculate number of chunks needed with overlap
    num_chunks = (duration // chunk_advance) + (1 if duration % chunk_advance > 0 else 0)
    
    all_text = []
    all_segments = {}  # Dictionary to store segments with their start times
    
    with sr.Microphone() as source:
        # Configure recognizer
        recognizer = sr.Recognizer()
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = 200  # Keep same sensitivity as other functions
        
        start_time = time.time()  # Track when we started recording
        recording_complete = False  # Flag to track if recording is complete
        
        print(f"Now listening with overlapping chunks for {duration} seconds...")
        
        # Record in overlapping chunks to improve recognition quality
        for i in range(num_chunks):
            # Stop immediately if we've already flagged the recording as complete
            if recording_complete:
                print("Recording already completed, skipping remaining chunks")
                break
                
            # Calculate chunk start and end times
            chunk_start = i * chunk_advance
            chunk_end = min(chunk_start + chunk_size, duration)
            
            # Skip if this chunk would start after the duration
            if chunk_start >= duration:
                recording_complete = True
                break
            
            # Check if we've already exceeded the total recording time
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                print(f"Total recording duration of {duration}s reached, stopping...")
                recording_complete = True
                break
            
            # Calculate how long to wait before starting this chunk's recording
            wait_time = chunk_start - (time.time() - start_time)
            if wait_time > 0:
                print(f"Waiting {wait_time:.1f}s until next chunk...")
                time.sleep(wait_time)
                
                # Check again if we've exceeded duration while waiting
                if time.time() - start_time >= duration:
                    print(f"Duration exceeded while waiting, stopping...")
                    recording_complete = True
                    break
            
            # Calculate actual duration for this chunk
            current_chunk_duration = chunk_end - chunk_start
            
            print(f"Recording chunk {i+1}/{num_chunks} " +
                  f"(seconds {chunk_start:.1f}-{chunk_end:.1f}, duration {current_chunk_duration:.1f}s)...")
            
            # Record this chunk
            audio = recognizer.record(source, duration=current_chunk_duration)
            chunk_actual_start = time.time() - start_time
            
            # Check one more time if we've exceeded the duration
            if time.time() - start_time >= duration:
                recording_complete = True
                
            try:
                # Process this chunk
                text = recognizer.recognize_google(audio)
                
                # Store segment with its approximate start time
                all_segments[chunk_actual_start] = text
                
                print(f"✓ Chunk {i+1} recognized: {len(text)} characters")
                if len(text) > 30:
                    print(f"  Preview: {text[:30]}...")
                else:
                    print(f"  Full text: {text}")
                
            except sr.UnknownValueError:
                print(f"⚠ No speech detected in chunk {i+1}")
            except sr.RequestError as e:
                print(f"⚠ Google Speech API error in chunk {i+1}: {e}")
                
            # Final check after processing - have we exceeded our time?
            if time.time() - start_time >= duration:
                print("Total duration reached after processing chunk, stopping...")
                recording_complete = True
    
    # Process all segments in order
    ordered_segments = sorted(all_segments.items())
    
    # Merge overlapping segments intelligently
    processed_text = []
    for i, (start_time, text) in enumerate(ordered_segments):
        if i == 0:
            # First segment always included fully
            processed_text.append(text)
        else:
            # Check for overlaps with previous segments
            prev_text = processed_text[-1]
            
            # Try to find where to join the segments
            # Look for shared phrases/words between the end of prev and start of current
            overlap_found = False
            
            # Check for overlap of phrases (at least 3 words)
            prev_words = prev_text.split()
            curr_words = text.split()
            
            # Try different overlap lengths
            for overlap_len in range(5, 2, -1):  # Try 5, 4, 3 word overlaps
                if len(prev_words) >= overlap_len and len(curr_words) >= overlap_len:
                    for j in range(len(prev_words) - overlap_len + 1):
                        overlap_phrase = " ".join(prev_words[j:j+overlap_len])
                        if text.startswith(overlap_phrase):
                            # Found overlap, join at this point
                            processed_text[-1] = prev_text[:prev_text.find(overlap_phrase)]
                            processed_text.append(text)
                            overlap_found = True
                            print(f"Found {overlap_len}-word overlap: '{overlap_phrase}'")
                            break
                if overlap_found:
                    break
            
            if not overlap_found:
                # No significant overlap found, just append
                processed_text.append(text)
    
    # Combine all processed text
    full_text = " ".join(processed_text)
    print(f"Finished listening. Total transcription: {len(full_text)} characters")
    
    return full_text

def generate_followup_questions(conversation_text, original_topic):
    """Generate follow-up questions based on the recorded conversation"""
    try:
        print("Analyzing conversation to create follow-up questions...")
        
        # Trim the conversation if it's very long
        if len(conversation_text) > 2000:
            conversation_text = conversation_text[:2000] + "..."
            print("Conversation truncated to 2000 characters for processing")
        
        # Prepare prompt for LLM 
        prompt = ChatPromptTemplate.from_template(
            """You are assisting a oerson with Alzheimer's who is having a conversation with someone.
            
            The initial topic was: {original_topic}
            
            Below is a transcript of the conversation:
            ---
            {conversation}
            ---
            
            Based on what they discussed, suggest 3 follow-up questions that can help continue the conversation.
            
            Your follow-up questions should:
            1. Connect directly to specific things mentioned in their conversation
            2. Focus on long-term memories (which are better preserved in Alzheimer's)
            3. Be simple, brief and clear
            4. Evoke positive emotions when possible
            5. Not test memory or recent events
            
            Format as a numbered list. Do not preface the suggestions or address the user directly. Do not add extra information.
            """
        )
        
        message = prompt.format_messages(
            original_topic=original_topic,
            conversation=conversation_text
        )
        
        print("Requesting follow-up questions from language model...")
        try:
            response = model.invoke(message)
            print("Response received")
            return response.content
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            # Fallback if the LLM fails
            return f"""1. Could you tell me more about what you just mentioned?
                        2. How did you feel about these experiences?
                        3. What other memories does this bring to mind?"""
            
    except Exception as e:
        print(f"Unexpected error in generate_followup_questions: {e}")
        return f"""1. Could you tell me more about what you just mentioned?
                2. How did you feel about these experiences?
                3. What other memories does this bring to mind?"""

def listen_for_speech(timeout=5):
    """Listen for speech input and convert to text"""
    print("Listening... (speak now)")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = 200
        try:
            audio = recognizer.listen(source, timeout=timeout)
            print("Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

def get_voice_input(prompt_text, retry_count=2):
    """Get input via voice with fallback to keyboard"""
    print(prompt_text)
    for attempt in range(retry_count):
        speech_input = listen_for_speech()
        if speech_input:
            return speech_input
        if attempt < retry_count - 1:
            print("Let's try again...")
    
    print("Falling back to keyboard input")
    return input("> ")

def get_numeric_rating_by_voice():
    """Get a numeric rating (1-10) via voice"""
    for attempt in range(3):
        speech_input = listen_for_speech()
        if speech_input:
            # Try to extract a number from the speech
            words = speech_input.lower().split()
            for word in words:
                try:
                    num = int(word)
                    if 1 <= num <= 10:
                        return num
                except ValueError:
                    # Check for word numbers
                    number_words = {
                        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
                    }
                    if word in number_words and 1 <= number_words[word] <= 10:
                        return number_words[word]
        
        print("Please say a number between 1 and 10")
    
    print("Falling back to keyboard input")
    while True:
        try:
            rating = int(input("Enter rating (1-10): "))
            if 1 <= rating <= 10:
                return rating
            print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
