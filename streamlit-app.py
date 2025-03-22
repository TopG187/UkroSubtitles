#!/usr/bin/env python3
"""
Web-based Audio Subtitle Generator using Streamlit
-------------------------------------------------
A web interface for generating subtitles from audio files.
"""

import os
import sys
import time
import tempfile
import math
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Try to import required packages, install if missing
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence, detect_nonsilent
except ImportError:
    st.warning("Installing required packages... Please wait.")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "SpeechRecognition", "pydub"])
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence, detect_nonsilent

# Define the subtitle segment class
class SubtitleSegment:
    """Represents a single subtitle segment with timing and text."""
    
    def __init__(self, index: int, start_time: float, end_time: float, text: str):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
    
    def format_time(self, seconds: float) -> str:
        """Format time in SRT format (HH:MM:SS,mmm)."""
        hours = math.floor(seconds / 3600)
        minutes = math.floor((seconds % 3600) / 60)
        seconds = seconds % 60
        milliseconds = math.floor((seconds - math.floor(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def to_srt(self) -> str:
        """Convert the segment to SRT format."""
        start_formatted = self.format_time(self.start_time)
        end_formatted = self.format_time(self.end_time)
        
        return f"{self.index}\n{start_formatted} --> {end_formatted}\n{self.text}\n"

class AudioToSubtitle:
    """Main class for converting audio to subtitles."""
    
    # Map language codes to SR language codes
    LANGUAGE_MAP = {
        "en": "en-US",  # English
        "es": "es-ES",  # Spanish
        "fr": "fr-FR",  # French
        "de": "de-DE",  # German
        "it": "it-IT",  # Italian
        "pt": "pt-BR",  # Portuguese
        "nl": "nl-NL",  # Dutch
        "ru": "ru-RU",  # Russian
        "zh": "zh-CN",  # Chinese (Simplified)
        "ja": "ja-JP",  # Japanese
        "ko": "ko-KR",  # Korean
    }
    
    def __init__(self, input_file: str, language: str = "en", 
                 min_silence_len: int = 500, silence_thresh: int = -40,
                 use_api: str = "google"):
        """
        Initialize the converter.
        
        Args:
            input_file: Path to the input audio file
            language: Language code for speech recognition
            min_silence_len: Minimum length of silence (in ms) for splitting
            silence_thresh: Threshold (in dB) for silence detection
            use_api: Speech recognition API to use (google, sphinx)
        """
        self.input_file = input_file
        self.language = self.LANGUAGE_MAP.get(language, "en-US")
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.use_api = use_api
        self.recognizer = sr.Recognizer()
        
        # Adjust recognizer settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def __del__(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    
    def convert_to_wav(self) -> str:
        """Convert input audio to WAV format."""
        st.text("Converting audio to WAV format...")
        
        # Create output path
        input_path = Path(self.input_file)
        output_path = Path(self.temp_dir.name) / f"{input_path.stem}.wav"
        
        # Use FFmpeg to convert to WAV (mono, 16kHz)
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-i", str(input_path), "-ar", "16000", "-ac", "1",
                "-y", str(output_path)
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            st.error("FFmpeg failed to convert the audio file. Make sure the file is valid.")
            raise
        except FileNotFoundError:
            st.error("FFmpeg not found. Please make sure FFmpeg is installed on the server.")
            raise
        
        return str(output_path)
    
    def detect_segments(self, audio_file: str) -> list:
        """Detect speech segments in the audio file."""
        st.text("Detecting speech segments...")
        
        # Load audio file
        audio = AudioSegment.from_wav(audio_file)
        
        # Detect non-silent chunks
        segments = detect_nonsilent(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh
        )
        
        st.text(f"Found {len(segments)} speech segments")
        return segments
    
    def recognize_segment(self, audio_data, segment_index: int) -> str:
        """Recognize speech in a segment."""
        try:
            if self.use_api == "google":
                return self.recognizer.recognize_google(audio_data, language=self.language)
            elif self.use_api == "sphinx":
                return self.recognizer.recognize_sphinx(audio_data)
            else:
                st.warning(f"Unknown API {self.use_api}, falling back to Google")
                return self.recognizer.recognize_google(audio_data, language=self.language)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            st.error(f"Error with speech recognition service: {e}")
            return f"[Recognition error on segment {segment_index}]"
    
    def process_audio(self, progress_bar) -> list:
        """Process the audio file and return subtitle segments."""
        # Convert audio to WAV format if necessary
        wav_file = self.convert_to_wav()
        progress_bar.progress(10)
        
        # Get speech segments
        segments = self.detect_segments(wav_file)
        progress_bar.progress(20)
        
        # Process each segment
        subtitle_segments = []
        
        st.text("Transcribing speech segments...")
        
        # Load the entire audio file
        audio = AudioSegment.from_wav(wav_file)
        
        # Open the wav file for speech recognition
        with sr.AudioFile(wav_file) as source:
            # Iterate over segments
            segment_count = len(segments)
            for i, (start_ms, end_ms) in enumerate(segments, 1):
                st.text(f"Processing segment {i}/{segment_count}...")
                
                # Update progress
                progress_value = 20 + (70 * i / segment_count)
                progress_bar.progress(int(progress_value))
                
                # Calculate timing in seconds
                start_time = start_ms / 1000.0
                end_time = end_ms / 1000.0
                
                # Extract segment audio
                segment_audio = audio[start_ms:end_ms]
                
                # Export segment to temp file
                segment_path = os.path.join(self.temp_dir.name, f"segment_{i}.wav")
                segment_audio.export(segment_path, format="wav")
                
                # Recognize the segment
                with sr.AudioFile(segment_path) as segment_source:
                    audio_data = self.recognizer.record(segment_source)
                    text = self.recognize_segment(audio_data, i)
                    
                    if text:
                        # Add to segments
                        subtitle = SubtitleSegment(i, start_time, end_time, text)
                        subtitle_segments.append(subtitle)
        
        progress_bar.progress(100)
        return subtitle_segments
    
    def generate_srt(self, segments: list) -> str:
        """Generate SRT content from segments."""
        if not segments:
            st.warning("No subtitle segments found.")
            return ""
        
        # Generate SRT content
        srt_content = ""
        for segment in segments:
            srt_content += segment.to_srt() + "\n"
        
        return srt_content

def save_uploaded_file(uploaded_file: UploadedFile) -> str:
    """Save an uploaded file to a temporary location and return the path."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create file path
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Write the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def main():
    st.set_page_config(
        page_title="Audio Subtitle Generator",
        page_icon="🎙️",
        layout="wide",
    )
    
    # Page header
    st.title("🎙️ Audio Subtitle Generator")
    st.markdown("""
    Upload an audio file to automatically generate SRT subtitles.
    The tool uses speech recognition to transcribe the audio and create accurately timed subtitles.
    """)
    
    # Sidebar with settings
    st.sidebar.title("Settings")
    
    # Language selection
    language_options = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko"
    }
    selected_language = st.sidebar.selectbox(
        "Select Language",
        options=list(language_options.keys()),
        index=0
    )
    language_code = language_options[selected_language]
    
    # Recognition API
    api_options = ["Google", "Sphinx"]
    selected_api = st.sidebar.selectbox(
        "Speech Recognition API",
        options=api_options,
        index=0,
        help="Google (online) provides better accuracy. Sphinx (offline) works without internet."
    )
    api_code = selected_api.lower()
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        min_silence_length = st.slider(
            "Minimum Silence Length (ms)",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
            help="Shorter values create more segments but may split sentences."
        )
        
        silence_threshold = st.slider(
            "Silence Threshold (dB)",
            min_value=-60,
            max_value=-20,
            value=-40,
            step=5,
            help="Lower values detect more speech but may include background noise."
        )
    
    # Check if FFmpeg is installed
    import shutil
    if not shutil.which("ffmpeg"):
        st.error("""
        FFmpeg is not installed or not in PATH.
        
        The server needs FFmpeg to process audio files.
        Please contact the administrator to install FFmpeg.
        """)
    
    # File uploader
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "flac", "aac", "ogg", "mp4"],
        help="Most common audio formats are supported"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File Information:")
        st.json(file_details)
        
        # Audio player
        st.audio(uploaded_file)
        
        # Process button
        if st.button("Generate Subtitles"):
            # Save uploaded file to temp location
            temp_file_path = save_uploaded_file(uploaded_file)
            
            with st.spinner("Processing audio..."):
                # Progress bar
                progress_bar = st.progress(0)
                
                # Create a container for log messages
                log_container = st.empty()
                
                try:
                    # Initialize the subtitle generator
                    converter = AudioToSubtitle(
                        temp_file_path,
                        language=language_code,
                        min_silence_len=min_silence_length,
                        silence_thresh=silence_threshold,
                        use_api=api_code
                    )
                    
                    # Process the audio
                    segments = converter.process_audio(progress_bar)
                    
                    # Generate SRT content
                    srt_content = converter.generate_srt(segments)
                    
                    # Display success message
                    if srt_content:
                        st.success(f"Successfully generated subtitles with {len(segments)} segments!")
                        
                        # Show subtitle content
                        st.subheader("Generated Subtitles")
                        st.text_area("SRT Content", srt_content, height=300)
                        
                        # Download button
                        output_filename = f"{os.path.splitext(uploaded_file.name)[0]}.srt"
                        st.download_button(
                            label="Download SRT File",
                            data=srt_content,
                            file_name=output_filename,
                            mime="text/plain"
                        )
                    else:
                        st.error("No speech segments were detected in the audio. Try adjusting the settings.")
                
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                        os.rmdir(os.path.dirname(temp_file_path))
                    except:
                        pass
    
    # Information section
    with st.expander("How it works"):
        st.markdown("""
        ### How the Subtitle Generator Works
        
        1. **Audio Processing**:
           - Converts the input audio to WAV format
           - Detects speech segments by identifying non-silent parts
        
        2. **Speech Recognition**:
           - Uses the selected recognition API to transcribe speech
           - Processes each speech segment individually
        
        3. **SRT Generation**:
           - Creates properly formatted SRT file with timing information
           - Matches text to the correct timestamps
        
        ### Tips for Best Results
        
        - Use clear audio with minimal background noise
        - Adjust silence threshold for different audio qualities
        - Try different minimum silence lengths for different speech patterns
        - For faster processing with lower accuracy, use the Sphinx API
        """)

if __name__ == "__main__":
    main()
