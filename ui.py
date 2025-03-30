import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import time
import customtkinter as ctk
from conversation_logic import *

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class ScrollableFrame(ctk.CTkScrollableFrame):
    """A modern scrollable frame"""
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)

class ConversationAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EverMind")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg="#242424" if ctk.get_appearance_mode() == "Dark" else "#ebebeb")
        
        # Define standard button styles
        self.small_button_font = ctk.CTkFont(size=15)
        self.small_button_height = 40
        self.medium_button_font = ctk.CTkFont(size=17)
        self.medium_button_height = 50
        
        self.is_recording = False
        self.manually_stopped = False
        self.stop_recording_button = None
        self.current_state = "main_menu"

        # Create a queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Recording duration setting (in seconds)
        self.recording_duration = 60
        
        # Main UI components
        self.create_widgets()
        
        # Initialize topic manager
        self.topic_manager = TopicManager()
        
        # Voice recording status
        self.is_recording = False
        
        # Current conversation state
        self.current_state = "main_menu"
        self.selected_topic = None
        self.conversation_text = ""
        
        # Voice input is enabled by default
        self.voice_enabled = True
        
        # Start with the main menu
        self.show_main_menu()
        
        # Start the queue processing
        self.process_queue()
    
    def create_widgets(self):
        # Main frame with minimal padding
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2) 
        
        # Create a container frame that will hold everything except the controls
        content_container = ctk.CTkFrame(main_frame)
        content_container.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        # Title
        title_label = ctk.CTkLabel(content_container, text="EverMind", 
                               font=ctk.CTkFont(size=40, weight="bold"))
        title_label.pack(pady=15)
        
        # Output display area with scrollbar
        self.output_frame = ctk.CTkFrame(content_container)
        self.output_frame.pack(fill=tk.X, expand=False, pady=10) 
        
        # Create a fixed height container for the output
        output_container = ctk.CTkFrame(self.output_frame)
        output_container.pack(fill=tk.BOTH, padx=5, pady=5)
        
        self.output_area = scrolledtext.ScrolledText(output_container, wrap=tk.WORD, 
                                                  width=80, height=10,  
                                                  font=("Segoe UI", 12))
        self.output_area.pack(fill=tk.BOTH, padx=5, pady=5)
        self.output_area.configure(state=tk.DISABLED)
        
        # Button frame for main navigation
        self.main_button_frame = ctk.CTkFrame(content_container)
        self.main_button_frame.pack(fill=tk.X, pady=10)
        
        # Progress bar for voice recording
        self.progress_frame = ctk.CTkFrame(content_container)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_label = ctk.CTkLabel(self.progress_frame, text=f"Recording: 0/{self.recording_duration}s")
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.progress_bar.set(0)
        
        # Hide progress bar initially
        self.progress_frame.pack_forget()
        
        # Action button frame for context-specific actions
        self.action_button_frame = ScrollableFrame(content_container, height=150)  
        self.action_button_frame.pack(fill=tk.BOTH, expand=True, pady=10) 
        
        # Text input area
        self.input_frame = ctk.CTkFrame(content_container)
        self.input_frame.pack(fill=tk.X, pady=5)
        
        self.input_label = ctk.CTkLabel(self.input_frame, text="Text input:")
        self.input_label.pack(side=tk.LEFT, padx=10)
        
        self.input_entry = ctk.CTkEntry(self.input_frame, width=400, font=ctk.CTkFont(size=14))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.input_button = ctk.CTkButton(self.input_frame, text="Submit", 
                                      command=self.submit_text_input,
                                      font=self.small_button_font,
                                      height=self.small_button_height)
        self.input_button.pack(side=tk.RIGHT, padx=10)
        
        # Hide input frame initially
        self.input_frame.pack_forget()
        
        # IMPORTANT: Controls frame is packed separately in main_frame, not in content_container
        # and with side=tk.BOTTOM to ensure it's always at the bottom
        self.controls_frame = ctk.CTkFrame(main_frame)
        self.controls_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        
        # Settings button
        self.settings_button = ctk.CTkButton(
            self.controls_frame,
            text="Settings",
            command=self.show_settings,
            width=100,
            font=self.small_button_font,
            height=self.small_button_height
        )
        self.settings_button.pack(side=tk.LEFT, padx=10)
        
        # Mode switch (dark/light)
        self.appearance_mode_label = ctk.CTkLabel(self.controls_frame, text="Appearance Mode:")
        self.appearance_mode_label.pack(side=tk.LEFT, padx=10)
        
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.controls_frame, 
                                                   values=["Light", "Dark", "System"],
                                                   command=self.change_appearance_mode)
        self.appearance_mode_menu.pack(side=tk.LEFT, padx=10)
        self.appearance_mode_menu.set("System")
        
        # Voice toggle switch
        self.voice_toggle_switch = ctk.CTkSwitch(self.controls_frame, text="Enabled", 
                                             command=self.toggle_voice)
        self.voice_toggle_switch.pack(side=tk.RIGHT, padx=10)
        self.voice_toggle_switch.select()  # Default to enabled
        self.voice_toggle_label = ctk.CTkLabel(self.controls_frame, text="Voice Input:")
        self.voice_toggle_label.pack(side=tk.RIGHT, padx=10)
        
        
    
    def change_appearance_mode(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)
    
    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        if self.voice_enabled:
            self.voice_toggle_switch.configure(text="Enabled")
            self.voice_toggle_switch.select()
        else:
            self.voice_toggle_switch.configure(text="Disabled")
            self.voice_toggle_switch.deselect()
    
    def clear_button_frames(self):
        # Clear main buttons
        for widget in self.main_button_frame.winfo_children():
            widget.destroy()
        
        # Clear action buttons from scrollable frame
        for widget in self.action_button_frame.winfo_children():
            widget.destroy()
    
    def write_to_output(self, text, clear=False):
        self.output_area.configure(state=tk.NORMAL) 
        if clear:
            self.output_area.delete(1.0, tk.END)
        self.output_area.insert(tk.END, text + "\n")
        self.output_area.see(tk.END)
        self.output_area.configure(state=tk.DISABLED) 
    
    def show_main_menu(self):
        # Clean up before changing states
        self.cleanup_resources()
        
        # Reset state variables
        self.current_state = "main_menu"
        self.conversation_text = ""
        
        # Clear UI elements
        self.clear_button_frames()
        self.input_frame.pack_forget()
        
        # Clear output and show welcome message
        self.write_to_output("=" * 50, clear=True)
        self.write_to_output("Alzheimer's Conversation Assistant")
        self.write_to_output("=" * 50)
        self.write_to_output("This tool helps suggest conversation topics and tracks their effectiveness")
        self.write_to_output("Voice recognition is enabled - you can speak your responses or use the buttons")
        self.write_to_output("\nWhat would you like to do?")
        
        # Speak welcome if TTS is enabled
        if tts_enabled:
            speak_text_async("Welcome! What would you like to do?")
        
        # Create main menu buttons - USING MEDIUM BUTTON STYLE
        ctk.CTkButton(self.main_button_frame, text="Get Topic Suggestions", 
                   command=self.get_topic_suggestions, 
                   font=self.medium_button_font, height=self.medium_button_height).pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        ctk.CTkButton(self.main_button_frame, text="Report Topic Effectiveness", 
                   command=self.report_effectiveness,
                   font=self.medium_button_font, height=self.medium_button_height).pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        ctk.CTkButton(self.main_button_frame, text="View Top Performing Topics", 
                   command=self.view_top_topics,
                   font=self.medium_button_font, height=self.medium_button_height).pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        ctk.CTkButton(self.main_button_frame, text="Exit", 
                   command=self.exit_app,
                   font=self.medium_button_font, height=self.medium_button_height).pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # If voice is enabled, start listening for voice command
        if self.voice_enabled:
            threading.Thread(target=self.listen_for_menu_choice, daemon=True).start()
    
    def listen_for_menu_choice(self):
        choice_input = listen_for_speech(timeout=10)
        
        if choice_input:
            if "one" in choice_input.lower() or "get" in choice_input.lower() or "suggestion" in choice_input.lower():
                self.queue.put(("get_topic_suggestions", None))
            elif "two" in choice_input.lower() or "report" in choice_input.lower() or "effective" in choice_input.lower():
                self.queue.put(("report_effectiveness", None))
            elif "three" in choice_input.lower() or "view" in choice_input.lower() or "performing" in choice_input.lower():
                self.queue.put(("view_top_topics", None))
            elif any(word in choice_input.lower() for word in ["four", "exit", "quit", "leave", "goodbye", "bye"]):
                self.queue.put(("exit_app", None))
    
    def process_queue(self):
        try:
            if hasattr(self, 'processing_queue') and self.processing_queue:
                # Skip if already processing to avoid reentrancy issues
                self.root.after(100, self.process_queue)
                return
                
            self.processing_queue = True
            
            while not self.queue.empty():
                action, data = self.queue.get_nowait()
                
                # Debug print to track actions
                print(f"Processing queue action: {action}")
                
                # IMPORTANT: Skip handle_listen_response if manually_stopped is True
                if action == "handle_listen_response" and self.manually_stopped:
                    print("Skipping handle_listen_response because recording was manually stopped")
                    continue
                    
                if action == "get_topic_suggestions":
                    self.get_topic_suggestions()
                elif action == "report_effectiveness":
                    self.report_effectiveness()
                elif action == "view_top_topics":
                    self.view_top_topics()
                elif action == "exit_app":
                    self.exit_app()
                elif action == "select_topic":
                    self.select_topic(data)
                elif action == "topic_selected":
                    self.topic_selected(data)
                elif action == "show_questions":
                    self.show_topic_questions(data)
                elif action == "handle_listen_response":
                    self.handle_listen_response(data)
                elif action == "show_followup_questions":
                    self.show_followup_questions(data)
                elif action == "report_topic_rating":
                    self.report_topic_rating(data)
                    
            self.processing_queue = False
        except Exception as e:
            self.processing_queue = False
            print(f"Error processing queue: {e}")
            import traceback
            traceback.print_exc()
        
        # Schedule next queue check
        self.root.after(100, self.process_queue)
    
    def get_topic_suggestions(self):
        self.current_state = "topic_suggestions"
        self.clear_button_frames()
        
        # Get all available topics
        suggestions = self.topic_manager.suggest_topics(show_all=True)
        
        # Display topics
        self.write_to_output("\nAvailable conversation topics:", clear=True)
        if tts_enabled:
            speak_text_async("Here is a list of conversation topics you can discuss:")
        
        for i, topic in enumerate(suggestions, 1):
            # If topic has a rating, show it
            if topic in self.topic_manager.topic_ratings:
                avg_rating = self.topic_manager.get_average_rating(topic)
                self.write_to_output(f"{i}. {topic} (Rating: {avg_rating:.1f}/10)")
            else:
                self.write_to_output(f"{i}. {topic}")
        
        # Display instruction
        self.write_to_output("\nSelect a topic or enter a new one:")
        
        # Show topic selection buttons (first 5 only to avoid overcrowding)
        for i, topic in enumerate(suggestions[:5], 1):
            btn = ctk.CTkButton(self.action_button_frame, text=f"{i}. {topic}",
                                font=self.small_button_font, height=self.small_button_height,
                                command=lambda t=topic: self.select_topic(t))
            btn.pack(fill=tk.X, pady=2)
        
        # Show "more topics" button if there are more than 5
        if len(suggestions) > 5:
            more_btn = ctk.CTkButton(self.action_button_frame, text="More Topics...",
                                    font=self.small_button_font, 
                                    height=self.small_button_height, 
                                    command=lambda: self.show_more_topics(suggestions, 5))
            more_btn.pack(fill=tk.X, pady=2)
        
        # Show back button
        back_btn = ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                                 font=self.small_button_font, height=self.small_button_height,
                                command=self.show_main_menu)
        back_btn.pack(side=tk.LEFT, padx=10)
        
        if self.voice_enabled:
            threading.Thread(target=self.listen_for_topic_choice, 
                          args=(suggestions,), daemon=True).start()
    
    def show_more_topics(self, suggestions, start_idx):
        # Clear current topic buttons
        for widget in self.action_button_frame.winfo_children():
            widget.destroy()
        
        # Show the next batch of topics (next 5 or remaining)
        end_idx = min(start_idx + 5, len(suggestions))
        for i, topic in enumerate(suggestions[start_idx:end_idx], start_idx+1):
            btn = ctk.CTkButton(self.action_button_frame, text=f"{i}. {topic}",
                                font=self.small_button_font, height=self.small_button_height,
                                command=lambda t=topic: self.select_topic(t))
            btn.pack(fill=tk.X, pady=2)
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(self.action_button_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        # Previous button if not at the beginning
        if start_idx > 0:
            prev_btn = ctk.CTkButton(nav_frame, text="Previous", 
                                font=self.small_button_font, 
                                height=self.small_button_height,
                                command=lambda: self.show_more_topics(suggestions, 
                                                                   max(0, start_idx-5)))
            prev_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Next button if not at the end
        if end_idx < len(suggestions):
            next_btn = ctk.CTkButton(nav_frame, text="Next", 
                                font=self.small_button_font, 
                                height=self.small_button_height,
                                command=lambda: self.show_more_topics(suggestions, end_idx))
            next_btn.pack(side=tk.RIGHT, padx=5, expand=True)
    
    def listen_for_topic_choice(self, suggestions):
        topic_choice_input = listen_for_speech(timeout=10)
        
        if topic_choice_input and "skip" not in topic_choice_input.lower():
            # Process voice input to identify the topic
            input_words = normalize_text(topic_choice_input)
            
            topic_idx = None
            # Check for number references
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
                    selected_topic = suggestions[topic_idx]
                    self.queue.put(("select_topic", selected_topic))
                    return
            
            # If no number reference found, use semantic matching
            best_score = 0.3  # Minimum threshold
            best_topic_idx = None
            
            for i, topic in enumerate(suggestions):
                topic_words = normalize_text(topic)
                similarity = get_similarity_score(topic_words, input_words)
                
                if similarity > best_score:
                    best_score = similarity
                    best_topic_idx = i
            
            if best_topic_idx is not None:
                selected_topic = suggestions[best_topic_idx]
                self.queue.put(("select_topic", selected_topic))
                return
    
    def select_topic(self, topic):
        self.selected_topic = topic
        self.queue.put(("topic_selected", topic))
    
    def topic_selected(self, topic):
        self.current_state = "generate_questions"
        self.clear_button_frames()
        
        # Show selected topic and generate questions
        self.write_to_output(f"Generating questions for: {topic}", clear=True)
        if tts_enabled:
            speak_text_async(f"Generating questions on the topic: {topic}")
        
        # Start a thread to generate questions
        threading.Thread(target=self.generate_questions_thread, 
                      args=(topic,), daemon=True).start()
    
    def generate_questions_thread(self, topic):
        questions = self.topic_manager.generate_specific_questions(topic)
        self.queue.put(("show_questions", questions))
    
    def show_topic_questions(self, questions):
        self.current_state = "show_questions"
        
        # Display the questions
        self.write_to_output(questions)
        
        # Ask if they want to listen to conversation
        self.write_to_output("\nWould you like me to listen to your conversation and suggest follow-up questions?")
        if tts_enabled:
            speak_text_async("Would you like me to listen to your conversation and suggest follow-up questions?")
        
        # Yes/No buttons for listening to conversation with CONSISTENT HEIGHT and FONT
        yes_btn = ctk.CTkButton(self.action_button_frame, text="Yes", 
                           command=lambda: self.handle_listen_response(True),
                           font=self.medium_button_font, 
                           height=self.medium_button_height)
        yes_btn.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        no_btn = ctk.CTkButton(self.action_button_frame, text="No", 
                          command=lambda: self.handle_listen_response(False),
                          font=self.medium_button_font, 
                          height=self.medium_button_height)
        no_btn.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # If voice is enabled, start listening for yes/no
        if self.voice_enabled:
            threading.Thread(target=self.listen_for_yes_no, 
                          args=("handle_listen_response",), daemon=True).start()
    
    def listen_for_yes_no(self, callback_action):
        response = listen_for_speech(timeout=10)
        
        if response:
            if any(word in response.lower() for word in ["yes", "yeah", "sure", "okay"]):
                self.queue.put((callback_action, True))
            elif any(word in response.lower() for word in ["no", "nope", "pass", "skip"]):
                self.queue.put((callback_action, False))
    
    def handle_listen_response(self, listen_to_conversation):
        if listen_to_conversation:
            self.current_state = "listening"
            self.clear_button_frames()
            
            # Show listening status with dynamic duration
            self.write_to_output(f"\nListening to conversation for {self.recording_duration} seconds...", clear=True)
            self.write_to_output("Please speak naturally about the selected topic.")
            if tts_enabled:
                speak_text_async("I'll listen to your conversation now, please speak naturally.")
            
            self.progress_frame.pack(fill=tk.X, pady=5, after=self.output_frame)
            self.progress_label.configure(text=f"Recording: 0/{self.recording_duration}s")
            self.progress_bar.set(0)
            
            # Start a thread to listen to conversation
            threading.Thread(target=self.listen_conversation_thread, daemon=True).start()
        else:
            # Return to main menu
            self.show_main_menu()
    
    def listen_conversation_thread(self):
        """Listen to conversation using the original listen_to_conversation function"""
        # Set recording state
        self.is_recording = True
        self.manually_stopped = False
        self.conversation_text = ""  # Start with empty conversation
        duration = self.recording_duration
        self.recording_completed = False  # New flag to track completion
        
        # Add a stop button
        self.root.after(0, lambda: self.add_stop_recording_button())
        
        # Define a timer function that updates the progress bar and stops after duration
        def update_progress():
            for i in range(1, duration + 1):
                if not self.is_recording:
                    print("Recording was manually stopped - progress updates cancelled")
                    break
                    
                # Update progress bar
                progress_value = i / duration
                self.root.after(0, lambda val=progress_value, sec=i: self.update_progress_ui(val, sec, duration))
                time.sleep(1)
                    
            # If we completed the loop without manual interruption, mark the recording as ready to stop
            # But don't actually stop it yet - wait for the recording thread to finish
            if self.is_recording:
                print(f"Progress timer completed after {duration}s - waiting for recording to finish")
                # DON'T call auto_stop_recording here anymore
                self.recording_completed = True
        
        # Start progress updater in a separate thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        # Start actual recording in a separate thread
        def record_audio():
            try:
                self.write_to_output("Starting to record conversation...")
                print("\n" + "="*50)
                print(f"Starting conversation recording for topic: {self.selected_topic}")
                print("="*50)
                recorded_text = listen_to_conversation(duration=duration)
                
                # Check if we should process the recording
                if self.is_recording:
                    print("Recording completed normally")
                    # Update the conversation text
                    self.conversation_text = recorded_text
                    print(f"Recorded conversation text: {len(recorded_text)} characters")
                    print(f"Preview: {recorded_text[:50] if len(recorded_text) > 50 else recorded_text}")
                    self.root.after(0, self.auto_stop_recording)
                else:
                    print("Recording was manually stopped - discarding further results")
            except Exception as e:
                print(f"Error in recording: {e}")
                import traceback
                traceback.print_exc()
                
                if self.is_recording:  # Only update UI if not manually stopped
                    # Clean up UI
                    self.is_recording = False
                    self.progress_frame.pack_forget()
                    self.remove_stop_recording_button()
                    
                    self.write_to_output(f"\nError during recording: {str(e)}")
                    ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                                  font=self.small_button_font, height=self.small_button_height,
                                command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
        
        # Start recording in separate thread
        recording_thread = threading.Thread(target=record_audio, daemon=True)
        recording_thread.start()

    def auto_stop_recording(self):
        """Stop recording when timer expires or recording completes normally"""
        if not self.is_recording:
            return
            
        # IMPORTANT: Capture text immediately to prevent race conditions
        current_text = self.conversation_text
        current_topic = self.selected_topic
        
        # Update state and UI
        self.is_recording = False
        self.progress_frame.pack_forget()
        self.remove_stop_recording_button()
        self.write_to_output("\nRecording completed.")
        
        # Debug to find the issue
        print(f"DEBUG: In auto_stop_recording - conversation text length: {len(current_text)}")
        print(f"DEBUG: Text preview: {current_text[:100]}...")
        print(f"DEBUG: FULL TEXT: {current_text}")
        
        # Process the results with the captured text
        if current_text and len(current_text) > 5:
            try:
                self.write_to_output("\nAnalyzing your conversation...")
                if tts_enabled:
                    speak_text_async("Thank you for letting me listen. I'll suggest some follow-up questions.")
                
                print(f"DEBUG: Calling generate_followup_questions with topic: {current_topic}")
                followup_questions = generate_followup_questions(current_text, current_topic)
                print(f"DEBUG: Generated follow-up questions: {followup_questions}")
                self.show_followup_questions(followup_questions)
                
            except Exception as e:
                print(f"Error processing conversation: {e}")
                import traceback
                traceback.print_exc()
                self.write_to_output(f"\nError analyzing conversation: {str(e)}")
                ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                                font=self.small_button_font, height=self.small_button_height,
                            command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
        else:
            self.write_to_output("\nNot enough conversation was detected to generate follow-up questions.")
            if tts_enabled:
                speak_text_async("I couldn't capture enough of your conversation to generate follow-up questions.")
            ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                            font=self.small_button_font, height=self.small_button_height,
                        command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
    
    def stop_recording(self):
        """Stop recording manually and save what we have so far"""
        if not self.is_recording:
            return
            
        print("Manual stop requested - stopping recording")
        
        # This saves current conversation text to prevent changes while it's being processed
        current_text = self.conversation_text
        current_topic = self.selected_topic
        
        # Update the flags so progress thread stops
        self.is_recording = False
        self.manually_stopped = True
        
        # Update UI
        self.progress_frame.pack_forget() 
        self.remove_stop_recording_button()
        self.write_to_output("Recording stopped manually.")
        
        # Wait a brief moment for any in-progress recording to complete
        def process_after_delay():
            # Get the latest conversation text (it might have been updated)
            final_text = self.conversation_text if len(self.conversation_text) > 0 else current_text
            
            print(f"Processing manually stopped recording - text length: {len(final_text)}")
            if len(final_text) > 50:
                print(f"Preview: {final_text[:50]}...")
            
            # Only process if we actually captured something
            if final_text and len(final_text) > 5:
                self.write_to_output("\nAnalyzing your conversation...")
                if tts_enabled:
                    speak_text_async("Thank you for letting me listen. I'll suggest some follow-up questions.")
                
                # Generate follow-up questions with the recorded text
                followup_questions = generate_followup_questions(final_text, current_topic)
                self.show_followup_questions(followup_questions)
            else:
                self.write_to_output("\nNot enough conversation was recorded to generate follow-up questions.")
                ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                                font=self.small_button_font, height=self.small_button_height,
                            command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
        
        # Give recording thread a moment to finish and update conversation_text
        self.root.after(200, process_after_delay)
    
    def process_recorded_conversation(self):
        """Process the recorded conversation to generate follow-up questions"""
        try:
            self.root.after(0, lambda: self.write_to_output("\nAnalyzing your conversation..."))
            
            # Use a delay to prevent TTS errors from overlapping
            if tts_enabled:
                try:
                    speak_text_async("Thank you for letting me listen. I'll suggest some follow-up questions based on what you discussed.")
                except Exception as e:
                    print(f"TTS error (non-critical): {e}")
            
            # Rest of your code...
            self.root.after(0, lambda: self.write_to_output("Generating follow-up questions..."))
            print("Generating follow-up questions for topic:", self.selected_topic)
            print(f"Conversation text preview: {self.conversation_text[:100]}...")
            
            # Make sure to use the same generation logic as the original code
            followup_questions = generate_followup_questions(self.conversation_text, self.selected_topic)
            
            if followup_questions:
                print("Successfully generated follow-up questions")
                self.queue.put(("show_followup_questions", followup_questions))
            else:
                raise Exception("Empty follow-up questions returned")
        except Exception as e:
            import traceback
            print(f"Follow-up questions error: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: self.write_to_output(f"\nError generating follow-up questions: {str(e)}"))
            self.root.after(0, lambda: ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                                        font=self.small_button_font, height=self.small_button_height,
                                        command=self.show_main_menu).pack(side=tk.LEFT, padx=10))
    
    def show_followup_questions(self, followup_questions):
        """Display follow-up questions directly without using the queue"""
        self.current_state = "followup_questions"
        self.clear_button_frames()
        
        # Display the follow-up questions
        self.write_to_output("\nBased on your conversation, here are follow-up questions:", clear=True)
        self.write_to_output(followup_questions)
        
        # Speak the questions if TTS is enabled
        if tts_enabled:
            try:
                speak_text_async("Based on your conversation, here are some follow-up questions:")
                #speak_text_async(followup_questions)
            except Exception as e:
                print(f"TTS error (non-critical): {e}")
        
        # Add a button to return to main menu
        ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                 command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
        
        # Add a button to rate this topic
        ctk.CTkButton(self.main_button_frame, text="Rate This Topic", 
                 command=lambda: self.start_topic_rating(self.selected_topic),
                 font=self.small_button_font,
                 height=self.small_button_height).pack(side=tk.LEFT, padx=10)
    
    def report_effectiveness(self):
        self.current_state = "report_effectiveness"
        self.clear_button_frames()
        
        # Prompt for topic
        self.write_to_output("What topic did you discuss?", clear=True)
        if tts_enabled:
            speak_text_async("What topic did you discuss?")
        
        self.input_frame.pack(fill=tk.X, pady=5)
        self.input_label.configure(text="Enter topic:")
        self.input_entry.delete(0, tk.END)
        
        # Button to submit topic
        self.input_button.configure(command=self.submit_topic_for_rating)
        
        # Back button
        ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                      font=self.small_button_font, height=self.small_button_height,
                    command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
        
        # If voice is enabled, start listening for topic
        if self.voice_enabled:
            threading.Thread(target=self.listen_for_topic_to_rate, daemon=True).start()
    
    def listen_for_topic_to_rate(self):
        topic_input = listen_for_speech(timeout=10)
        if topic_input:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, topic_input)
            self.submit_topic_for_rating()
    
    def submit_topic_for_rating(self):
        topic = self.input_entry.get().strip()
        if topic:
            # Get all existing topics to match against
            all_topics = self.topic_manager.suggest_topics(show_all=True)
            
            # Match with existing topics
            if all_topics:
                # First check for exact match (case insensitive)
                for existing_topic in all_topics:
                    if existing_topic.lower() == topic.lower():
                        print(f"Found exact match for topic: {existing_topic}")
                        self.start_topic_rating(existing_topic)
                        return
                
                # If no exact match, try semantic matching
                input_words = normalize_text(topic)
                best_score = 0.3  # Minimum threshold 
                best_match = None
                
                for existing_topic in all_topics:
                    topic_words = normalize_text(existing_topic)
                    similarity = get_similarity_score(topic_words, input_words)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = existing_topic
                
                if best_match:
                    print(f"Found similar topic: '{best_match}' (similarity: {best_score:.2f})")
                    # Show user what the topic was matched to
                    self.write_to_output(f"Matched to existing topic: '{best_match}'")
                    topic = best_match
            
            # Start rating process with exact or matched topic
            self.start_topic_rating(topic)
    
    def start_topic_rating(self, topic):
        self.current_state = "rate_topic"
        self.selected_topic = topic
        self.clear_button_frames()
        
        # Prompt for rating
        self.write_to_output(f"\nHow effective was discussing '{topic}' in stimulating conversation? (1-10)", clear=True)
        if tts_enabled:
            speak_text_async(f"How effective was this topic in stimulating conversation, on a scale of 1 to 10?")

        # Organize rating buttons into two rows since it looks nicer
        rating_frame1 = ctk.CTkFrame(self.action_button_frame)
        rating_frame1.pack(fill=tk.X, pady=(5, 2))
        rating_frame2 = ctk.CTkFrame(self.action_button_frame)
        rating_frame2.pack(fill=tk.X, pady=(2, 5))
        
        # Use different colors for different rating ranges
        for i in range(1, 11):
            if i <= 3:
                button_color = "#E74C3C"  # Red for low ratings
            elif i <= 7:
                button_color = "#F39C12"  # Orange/yellow for medium ratings
            else:
                button_color = "#2ECC71"  # Green for high ratings
                
            # Choose the correct frame based on the value
            parent_frame = rating_frame1 if i <= 5 else rating_frame2
                
            btn = ctk.CTkButton(parent_frame, text=str(i), 
                          command=lambda r=i: self.submit_rating(r),
                          fg_color=button_color,
                          width=50)  # Set fixed width for consistent sizing
            btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Return to Main Menu button
        ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                   command=self.show_main_menu,
                     font=self.small_button_font, height=self.small_button_height).pack(side=tk.LEFT, padx=10)
        
        # If voice is enabled, start listening for rating
        if self.voice_enabled:
            threading.Thread(target=self.listen_for_rating, daemon=True).start()
    
    def listen_for_rating(self):
        rating = get_numeric_rating_by_voice()
        if rating:
            self.queue.put(("report_topic_rating", rating))
    
    def submit_rating(self, rating):
        """Submit a rating for the current topic"""
        print(f"Rating submitted: {rating} for topic: {self.selected_topic}")
        
        # Provide visual feedback
        self.write_to_output(f"Submitting rating: {rating}/10...")
        
        # Put in queue and process immediately 
        self.queue.put(("report_topic_rating", rating))
        
        # Disable all rating buttons to prevent multiple submissions
        for widget in self.action_button_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for button in widget.winfo_children():
                    button.configure(state="disabled")
    
    def report_topic_rating(self, rating):
        """Process a rating for the currently selected topic"""
        # Clear the button frames to ensure new buttons are visible
        self.clear_button_frames()
        
        if rating is not None: 
            try:
                # Debug output to trace the process
                print(f"Processing rating: {rating} for topic: {self.selected_topic}")
                
                # Add the rating to the topic manager
                matched_topic = self.topic_manager.add_topic_rating(self.selected_topic, rating)
                
                # Show confirmation to user
                self.write_to_output(f"Recorded: {matched_topic} with effectiveness rating of {rating}/10", clear=True)
                if tts_enabled:
                    speak_text_async(f"Thank you. I've recorded a rating of {rating} out of 10 for the topic: {matched_topic}")
                
                # Button to return to main menu
                ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                            font=self.small_button_font, height=self.small_button_height,
                                command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
            except Exception as e:
                # Add error handling
                print(f"Error processing topic rating: {e}")
                import traceback
                traceback.print_exc()
                self.write_to_output(f"Error saving rating: {str(e)}")
                ctk.CTkButton(self.main_button_frame, text="Return to Main Menu",
                                font=self.small_button_font, height=self.small_button_height,
                                    command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
        else:
            # Handle case where rating is None or invalid
            print("Invalid rating value received")
            self.write_to_output("Invalid rating provided. Please try again.")
            ctk.CTkButton(self.main_button_frame, text="Return to Main Menu",
                          font=self.small_button_font, height=self.small_button_height,
                         command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
    
    def view_top_topics(self):
        self.current_state = "view_top_topics"
        self.clear_button_frames()
        
        # Get top topics
        top_topics = self.topic_manager.get_top_topics(5)
        
        if top_topics:
            self.write_to_output("Top performing topics:", clear=True)
            if tts_enabled:
                speak_text_async("Here is a list of the top performing topics:")
            
            for topic, rating in top_topics:
                self.write_to_output(f"- {topic}: {rating:.1f}/10")
                if tts_enabled:
                    speak_text_async(f"{topic} with an average rating of {rating:.1f} out of 10")
        else:
            self.write_to_output("\nNo topic ratings recorded yet", clear=True)
            if tts_enabled:
                speak_text_async("No topic ratings have been recorded yet")
        
        # Button to return to main menu
        ctk.CTkButton(self.main_button_frame, text="Return to Main Menu", 
                      font=ctk.CTkFont(size=14), height=30,
                 command=self.show_main_menu).pack(side=tk.LEFT, padx=10)
    
    def submit_text_input(self):
        if self.current_state == "topic_suggestions":
            # Submit topic choice
            topic = self.input_entry.get().strip()
            if topic:
                self.select_topic(topic)
        elif self.current_state == "report_effectiveness":
            # Submit topic for rating
            self.submit_topic_for_rating()
    
    def exit_app(self):
        self.write_to_output("\nExiting. Your topic data has been saved.", clear=True)
        if tts_enabled:
            speak_text_async("Exiting. Your topic data has been saved.")
        self.topic_manager.save_data()
        self.root.after(2000, self.root.destroy)

    def add_stop_recording_button(self):
        self.stop_recording_button = ctk.CTkButton(
            self.progress_frame, 
            text="Stop Recording", 
            command=self.stop_recording,
            fg_color="#E74C3C",  # Red color for stop button
            font=self.small_button_font,
            height=self.small_button_height
        )
        self.stop_recording_button.pack(side=tk.RIGHT, padx=10)

    def remove_stop_recording_button(self):
        if hasattr(self, 'stop_recording_button') and self.stop_recording_button:
            self.stop_recording_button.destroy()
            self.stop_recording_button = None

    def update_progress_ui(self, value, current_sec, total_sec):
        """Update progress bar from any thread safely"""
        self.progress_bar.set(value)  # CustomTkinter uses set() method
        self.progress_label.configure(text=f"Recording: {current_sec}/{total_sec}s")

    def show_settings(self):
        """Show settings as a frame in the main UI"""
        # Clear current UI
        self.current_state = "settings"
        self.clear_button_frames()
        self.input_frame.pack_forget()
        
        # Store the current appearance state to restore later if canceled
        original_duration = self.recording_duration
        
        # Create settings directly in the action_button_frame
        # Title
        ctk.CTkLabel(self.action_button_frame, text="Settings", 
                   font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Recording duration settings
        duration_frame = ctk.CTkFrame(self.action_button_frame)
        duration_frame.pack(fill=tk.X, pady=10, padx=20)
        
        ctk.CTkLabel(duration_frame, text="Recording Duration (seconds):").pack(side=tk.LEFT, padx=10)
        
        # Value label to show current setting
        duration_value_label = ctk.CTkLabel(duration_frame, text=str(self.recording_duration))
        duration_value_label.pack(side=tk.RIGHT, padx=10)
        
        # Slider for duration
        duration_slider = ctk.CTkSlider(
            self.action_button_frame, 
            from_=15, 
            to=120,
            number_of_steps=21,  # 15, 20, 25, ... 120
            command=lambda value: duration_value_label.configure(text=str(int(value)))
        )
        duration_slider.pack(fill=tk.X, padx=40, pady=5)
        duration_slider.set(self.recording_duration)
        
        # Button frame for Save/Cancel
        button_frame = ctk.CTkFrame(self.action_button_frame)
        button_frame.pack(fill=tk.X, pady=20, padx=20)
        
        # Save button
        save_button = ctk.CTkButton(
            button_frame,
            text="Save Settings",
            command=lambda: self.save_inline_settings(int(duration_slider.get())),
            fg_color="#2ECC71",  # Green
            font=self.medium_button_font,
            height=self.medium_button_height
        )
        save_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # Cancel button 
        cancel_button = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=lambda: self.cancel_settings(original_duration),
            fg_color="#E74C3C",  # Red
            font=self.medium_button_font,
            height=self.medium_button_height
        )
        cancel_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # Show explanation
        self.write_to_output("Settings", clear=True)
        self.write_to_output("\nAdjust the recording duration for conversation listening.")
        self.write_to_output("\nA longer duration allows for more conversation to be captured,")
        self.write_to_output("while a shorter duration may be better for brief exchanges.")

    def save_inline_settings(self, duration):
        """Save the settings and return to main menu"""
        self.recording_duration = duration
        print(f"Recording duration set to {duration} seconds")
        self.show_main_menu()

    def cancel_settings(self, original_duration):
        """Cancel settings changes and return to main menu"""
        self.recording_duration = original_duration  # Restore original value
        self.show_main_menu()

    def cleanup_resources(self):
        """Clean up any resources that need explicit management"""
        # Make sure recording is stopped
        self.is_recording = False
        self.manually_stopped = False
        
        # Hide and reset UI elements
        if self.progress_frame.winfo_ismapped():
            self.progress_frame.pack_forget()
        
        if hasattr(self, 'stop_recording_button') and self.stop_recording_button:
            self.stop_recording_button.destroy()
            self.stop_recording_button = None


if __name__ == "__main__":
    def show_error(exc_type, exc_value, exc_tb):
        """Global exception handler to prevent app from crashing silently"""
        import traceback
        print("Uncaught exception:")
        traceback.print_exception(exc_type, exc_value, exc_tb)
        error_msg = f"{exc_type.__name__}: {exc_value}"
        
        # Try to show an error window
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Application Error", 
                               f"An unexpected error occurred:\n\n{error_msg}\n\nSee console for details.")
        except:
            pass
    
    # Set the global exception handler
    import sys
    sys.excepthook = show_error
    
    root = tk.Tk()
    app = ConversationAssistantGUI(root)
    root.mainloop()