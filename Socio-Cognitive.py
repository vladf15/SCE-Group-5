import json
import os
import time
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
    
    #don't read very long texts entirely - just the first few sentences
    if len(text) > 500:
        sentences = text.split('.')
        shortened_text = '.'.join(sentences[:5]) + '.'
        tts_engine.say(shortened_text)
        tts_engine.say("I'll continue reading if you press Enter.")
        tts_engine.runAndWait()
        input("Press Enter to continue reading...")
        remaining_text = '.'.join(sentences[5:]) + '.'
        tts_engine.say(remaining_text)
    else:
        tts_engine.say(text)
    
    tts_engine.runAndWait()


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
    "Art events from the 90s",
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
        """Record a new effectiveness rating for a topic"""
        if topic not in self.topic_ratings:
            self.topic_ratings[topic] = []
        
        self.topic_ratings[topic].append(effectiveness)
        self.topic_history.append({
            "topic": topic, 
            "effectiveness": effectiveness,
            "timestamp": datetime.now().isoformat()
        })
        self.save_data()
    
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
                """You are assisting a caregiver for an Alzheimer's patient.
                Generate 3 specific, engaging questions about the topic: {topic}
                
                These questions should:
                1. Focus on long-term memories (which are better preserved in Alzheimer's)
                2. Be simple and clear
                3. Evoke positive emotions when possible
                4. Not test memory or recent events
                5. Do not explain each suggestion, just provide the questions
                
                Format as a numbered list. Do not add extra information.
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
def listen_for_speech(timeout=5):
    """Listen for speech input and convert to text"""
    print("Listening... (speak now)")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
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

def main():
    print("=" * 50)
    print("Alzheimer's Conversation Assistant")
    print("=" * 50)
    print("This tool helps suggest conversation topics and tracks their effectiveness")
    print("Voice recognition is enabled - you can speak your responses")
    print()
    
    topic_manager = TopicManager()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Get topic suggestions")
        print("2. Report topic effectiveness")
        print("3. View top performing topics")
        print("4. Exit")
        
        speak_text("\nWhat would you like to do?\n1. Get topic suggestions\n2. Report topic effectiveness\n3. View top performing topics\n4. Exit")
        
        # Get choice with voice or keyboard input
        choice_input = get_voice_input("Say or type your choice (1-4)")
        
        # Process voice input to extract the choice number
        choice = None
        if choice_input:
            if choice_input.isdigit() and len(choice_input) == 1:
                choice = choice_input
            elif "one" in choice_input.lower() or "get" in choice_input.lower() or "suggestion" in choice_input.lower():
                choice = "1"
            elif "two" in choice_input.lower() or "report" in choice_input.lower() or "effective" in choice_input.lower():
                choice = "2"
            elif "three" in choice_input.lower() or "view" in choice_input.lower() or "performing" in choice_input.lower():
                choice = "3"
            elif "four" in choice_input.lower() or "exit" in choice_input.lower() or "quit" in choice_input.lower():
                choice = "4"
        
        if choice == "1":
             # Suggest all available topics
            suggestions = topic_manager.suggest_topics(show_all=True)
            print("\nAvailable conversation topics:")
            speak_text("Here is a list of conversation topics you can discuss:")
            for i, topic in enumerate(suggestions, 1):
                # If topic has a rating, show it
                if topic in topic_manager.topic_ratings:
                    avg_rating = topic_manager.get_average_rating(topic)
                    print(f"{i}. {topic} (Rating: {avg_rating:.1f}/10)")
                else:
                    print(f"{i}. {topic}")
                    
                    
            # Ask if they want specific questions for a topic
            topic_choice_input = get_voice_input("\nSay a topic you'd like to discuss (or say 'skip')")

            topic_idx = None
            if topic_choice_input:
                if "skip" in topic_choice_input.lower():
                    continue
                
                print(f"Understanding your interest in: '{topic_choice_input}'")
                input_words = normalize_text(topic_choice_input)
                
                # First check for number references (maintain this functionality)
                number_references = {
                    "1": 0, "one": 0, "first": 0,
                    "2": 1, "two": 1, "second": 1,
                    "3": 2, "three": 2, "third": 2,
                    "4": 3, "four": 3, "fourth": 3,
                    "5": 4, "five": 4, "fifth": 4
                }
                
                for word in input_words:
                    if word in number_references and number_references[word] < len(suggestions):
                        topic_idx = number_references[word]
                        print(f"Selected topic #{topic_idx+1} by number reference: {suggestions[topic_idx]}")
                        break
                
                # If no number reference found, use semantic matching
                if topic_idx is None:
                    best_score = 0.3  # Minimum threshold to consider a match
                    best_topic_idx = None
                    
                    for i, topic in enumerate(suggestions):
                        topic_words = normalize_text(topic)
                        
                        # Calculate similarity between normalized topic and input
                        similarity = get_similarity_score(topic_words, input_words)
                        
                        print(f"Topic: '{topic}' - similarity score: {similarity:.2f}")
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_topic_idx = i
                    
                    if best_topic_idx is not None:
                        topic_idx = best_topic_idx
                        print(f"Selected topic: {suggestions[topic_idx]} (similarity: {best_score:.2f})")
                    else:
                        # Try to identify partial matches
                        for i, topic in enumerate(suggestions):
                            for input_word in input_words:
                                if len(input_word) > 3:  # Only consider significant words
                                    for topic_word in normalize_text(topic):
                                        # Check if input word is contained in topic word or vice versa
                                        if input_word in topic_word or topic_word in input_word:
                                            if best_topic_idx is None or len(topic_word) > best_score:
                                                best_score = len(topic_word)
                                                best_topic_idx = i
                        
                        if best_topic_idx is not None:
                            topic_idx = best_topic_idx
                            print(f"Selected topic by partial match: {suggestions[topic_idx]}")

            # If topic selection was successful, generate questions
            if topic_idx is not None:
                selected_topic = suggestions[topic_idx]
                print(f"\nGenerating questions for: {selected_topic}")
                speak_text(f"\nGenerating questions on the topic: {suggestions[topic_idx]}")
                questions = topic_manager.generate_specific_questions(selected_topic)
                print(questions)
                speak_text(questions)
                
                # Add a pause so user can read the questions
                input("\nPress Enter to continue...")
            else:
                print("\nCould not determine which topic you selected. Please try again.")
                speak_text("Could not determine which topic you selected. Please try again.")
                print("You can say the topic name or number, for example:")
                print(f"- \"{suggestions[0]}\"")
                print(f"- \"Number one\"")
                print(f"- \"Childhood\" (if one of the topics contains this word)")
                speak_text("You can say the topic name or number to select a topic")
                time.sleep(3)
        elif choice == "2":
            # Report effectiveness
            speak_text("What topic did you discuss with the patient?")
            topic = get_voice_input("\nWhat topic did you discuss with the patient?")
            
            speak_text("How effective was this topic in stimulating conversation, on a scale of 1 to 10?")
            print("\nHow effective was this topic in stimulating conversation? (1-10)")
            rating = get_numeric_rating_by_voice()
            
            if rating:
                topic_manager.add_topic_rating(topic, rating)
                print(f"Recorded: {topic} with effectiveness rating of {rating}/10")
        
        elif choice == "3":
            # View top topics
            top_topics = topic_manager.get_top_topics(5)
            if top_topics:
                speak_text("Here is a list of the top performing topics:")
                print("\nTop performing topics:")
                for topic, rating in top_topics:
                    print(f"- {topic}: {rating:.1f}/10")
                    speak_text(f"{topic} with an average rating of {rating:.1f} out of 10")
            else:
                print("\nNo topic ratings recorded yet")
                speak_text("No topic ratings have been recorded yet")
        
        elif choice == "4":
            print("\nExiting. Your topic data has been saved.")
            speak_text("Exiting. Your topic data has been saved.")
            break
        
        else:
            print("\nI didn't understand your choice, please try again")
            speak_text("I didn't understand your choice, please try again")
            time.sleep(1)

if __name__ == "__main__":
    main()