import time
from conversation_logic import *
import tkinter as tk
from ui import *

'''
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
        
        speak_text("\nWhat would you like to do?")
        
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
            elif any(word in choice_input.lower() for word in ["four", "exit", "quit", "leave", "goodbye", "bye", 
                                              "stop", "end", "finish", "done", "leave me alone",
                                              "go away", "that's all", "shut down", "close"]):
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
                speak_text(f"\nGenerating questions on the topic: {selected_topic}")
                questions = topic_manager.generate_specific_questions(selected_topic)
                print(questions)
                speak_text(questions)
                
                # Add option to listen to conversation and provide follow-up questions
                listen_prompt = "\nWould you like me to listen to your conversation and suggest follow-up questions?"
                print(listen_prompt + " (yes/no)")
                speak_text(listen_prompt)
                listen_response = get_voice_input("")
                
                if listen_response and any(word in listen_response.lower() for word in ["yes", "yeah", "sure", "okay"]):
                    # Listen to the conversation about the topic
                    conversation_text = listen_to_conversation(duration=60) #listen for a minute
                    
                    # If we captured some conversation
                    if conversation_text and len(conversation_text) > 20:  # Make sure we got something meaningful
                        print("\nAnalyzing your conversation...")
                        speak_text("Thank you for letting me listen. I'll suggest some follow-up questions based on what you discussed.")
                        
                        # Generate follow-up questions based on the conversation
                        followup_questions = generate_followup_questions(conversation_text, selected_topic)
                        
                        print("\nBased on your conversation, here are follow-up questions:")
                        print(followup_questions)
                        speak_text("Based on your conversation, here are some follow-up questions:")
                        speak_text(followup_questions)
                    else:
                        print("\nNot enough conversation was detected to generate follow-up questions.")
                        speak_text("I couldn't capture enough of your conversation to generate follow-up questions.")
                
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
            speak_text("What topic did you discuss?")
            topic = get_voice_input("\nWhat topic did you discuss?")
            
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
            if "leave" in choice_input.lower() or "go away" in choice_input.lower():
                print("\nI understand you want some space. Exiting now.")
                speak_text("I understand you want some space. Exiting now.")
            elif "goodbye" in choice_input.lower() or "bye" in choice_input.lower():
                print("\nGoodbye! Your topic data has been saved.")
                speak_text("Goodbye! Your topic data has been saved.")
            else:
                print("\nExiting. Your topic data has been saved.")
                speak_text("Exiting. Your topic data has been saved.")
            break
        
        else:
            print("\nI didn't understand your choice, please try again")
            speak_text("I didn't understand your choice, please try again")
            time.sleep(1)

if __name__ == "__main__":
    main()
    '''

def main():
    root = tk.Tk()
    app = ConversationAssistantGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()