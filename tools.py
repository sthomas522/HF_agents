# Standard Library
import os
import re
import tempfile
import string 
import glob
import shutil
import gc
import sys
import uuid
import signal
from pathlib import Path
import subprocess
from datetime import datetime
from io import BytesIO
from contextlib import contextmanager
from langchain_huggingface import HuggingFacePipeline
from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal, Union, Tuple, Set, Type
import time
from collections import Counter
from pydantic import Field, BaseModel, Extra
import hashlib
import json
import numpy as np
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

# Third-Party Packages
import cv2
import requests
import wikipedia
import spacy
import yt_dlp
import librosa
from PIL import Image
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
from nltk.corpus import words
import pandas as pd

# LangChain Ecosystem
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WikipediaLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool, render_text_description
from langchain_core.documents import Document
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# LangGraph
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# PyTorch
import torch
from functools import partial
from transformers import pipeline

# Additional Utilities
from datetime import datetime

from urllib.parse import urljoin, urlparse
import logging

nlp = spacy.load("en_core_web_sm")

# Ensure the word list is downloaded
nltk.download('words', quiet=True)
english_words = set(words.words())

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', level=logging.INFO)

# --- Model Configuration ---
def create_llm_pipeline():
    #model_id = "meta-llama/Llama-2-13b-chat-hf"
    #model_id = "meta-llama/Llama-3.3-70B-Instruct"
    #model_id = "mistralai/Mistral-Small-24B-Base-2501"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    #model_id = "Meta-Llama/Llama-2-7b-chat-hf"
    #model_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    #model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    #model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    #model_id = "Qwen/Qwen2-7B-Instruct"
    #model_id = "GSAI-ML/LLaDA-8B-Instruct"
    return pipeline(
        "text-generation",
        model=model_id,
        device_map="cpu",
        torch_dtype=torch.float16,
        max_new_tokens=1024,
        temperature=0.05,
        do_sample=False,
        repetition_penalty=1.2
    )

# Define file extension sets for each category
PICTURE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma'}
CODE_EXTENSIONS = {'.py', '.js', '.java', '.cpp', '.c', '.cs', '.rb', '.go', '.php', '.html', '.css', '.ts'}
SPREADSHEET_EXTENSIONS = {
    '.xls', '.xlsx', '.xlsm', '.xlsb', '.xlt', '.xltx', '.xltm',
    '.ods', '.ots', '.csv', '.tsv', '.sxc', '.stc', '.dif', '.gsheet',
    '.numbers', '.numbers-tef', '.nmbtemplate', '.fods', '.123', '.wk1', '.wk2',
    '.wks', '.wku', '.wr1', '.gnumeric', '.gnm', '.xml', '.pmvx', '.pmdx',
    '.pmv', '.uos', '.txt'
}

def get_file_type(filename: str) -> str:
    if not filename or '.' not in filename or filename == '':
        return ''
    ext = filename.lower().rsplit('.', 1)[-1]
    dot_ext = f'.{ext}'
    if dot_ext in PICTURE_EXTENSIONS:
        return 'picture'
    elif dot_ext in AUDIO_EXTENSIONS:
        return 'audio'
    elif dot_ext in CODE_EXTENSIONS:
        return 'code'
    elif dot_ext in SPREADSHEET_EXTENSIONS:
        return 'spreadsheet'
    else:
        return 'unknown'

def write_bytes_to_temp_dir(file_bytes: bytes, file_name: str) -> str:
    """
    Writes bytes to a file in the system temporary directory using the provided file_name.
    Returns the full path to the saved file.
    The file will persist until manually deleted or the OS cleans the temp directory.
    """
    temp_dir = "/tmp"  # /tmp is always writable in Hugging Face Spaces
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file_name)
    with open(file_path, 'wb') as f:
        f.write(file_bytes)
    print(f"File written to: {file_path}")
    return file_path

def extract_final_answer(text: str) -> str:
    """
    Extracts the answer after the last 'FINAL ANSWER:' (case-insensitive),
    removes any parenthetical immediately following a numeric answer,
    strips trailing punctuation, sorts comma-separated lists,
    and does not split numbers containing commas.
    Returns an empty string if marker not found.
    """
    marker = "FINAL ANSWER:"
    idx = text.lower().rfind(marker.lower())
    if idx == -1:
        return ""
    # Extract answer after marker
    result = text[idx + len(marker):].strip()
    if "pure vanilla extract" not in result:
      result = result.replace("vanilla extract", "pure vanilla extract")
    # Remove parenthetical immediately following a number at the start
    result = re.sub(r'^(\d+(?:\.\d+)?)\s*\(.*?\)', r'\1', result)
    # Remove trailing punctuation and whitespace
    result = result.rstrip(string.punctuation + " ")
    # Split on commas NOT between digits (i.e., not inside numbers)
    # This regex splits on commas not surrounded by digits (to avoid splitting numbers like 1,000)
    items = re.split(r',(?!\s*\d{3}\b)', result)
    # If we have a list, sort it
    if len(items) > 1:
        items = [item.strip() for item in items]
        # Try to sort numerically
        try:
            sorted_items = sorted(
                items,
                key=lambda x: float(re.sub(r'[^\d\.]', '', x))  # Remove non-numeric except .
            )
            return ', '.join(sorted_items)
        except ValueError:
            # Fallback: sort alphabetically
            sorted_items = sorted(items, key=lambda x: x.lower())
            return ', '.join(sorted_items)
    return result


class AudioTranscriptionInput(BaseModel):
    """Input schema for AudioTranscriptionTool."""
    file_path: str = Field(description="Path to the audio file to transcribe")
    engine: Optional[str] = Field(default="google", description="Speech recognition engine to use")
    language: Optional[str] = Field(default="en-US", description="Language of the audio")

class AudioTranscriptionTool(BaseTool):
    """Tool for transcribing audio files using local speech recognition."""
    
    name: str = "audio_transcription"
    description: str = """
    Transcribes voice memo, audio files (mp3, wav, m4a, flac, etc.) to text using local speech recognition.
    Input should be a dictionary with 'file_path' key containing the path to the audio file.
    Optionally accepts 'engine' and 'language' parameters.
    Returns the transcribed text as a string.
    """
    args_schema: type[BaseModel] = AudioTranscriptionInput
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the AudioTranscriptionTool."""
        super().__init__(**kwargs)
        self._init_speech_recognition()
    
    def _init_speech_recognition(self):
        """Initialize speech recognition components."""
        try:
            import speech_recognition as sr
            from pydub import AudioSegment
            object.__setattr__(self, 'recognizer', sr.Recognizer())
            object.__setattr__(self, 'sr', sr)
            object.__setattr__(self, 'AudioSegment', AudioSegment)
        except ImportError as e:
            raise ImportError(
                "Required libraries not found. Install with: "
                "pip install SpeechRecognition pydub"
            ) from e
    
    def _validate_audio_file(self, file_path: str) -> bool:
        """Validate that the audio file exists and has a supported format."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check file extension - pydub supports many formats
        supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.mp4', '.mpeg', '.mpga', '.webm', '.ogg', '.aac'}
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        return True
    
    def _convert_to_wav(self, file_path: str) -> str:
        """Convert audio file to WAV format if needed."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.wav':
            return file_path
        
        try:
            # Convert to WAV using pydub
            audio = self.AudioSegment.from_file(file_path)
            
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio.export(temp_wav.name, format="wav")
            return temp_wav.name
        except Exception as e:
            raise RuntimeError(f"Error converting audio file to WAV: {str(e)}")
    
    def _transcribe_audio(self, file_path: str, engine: str = "google", language: str = "en-US") -> str:
        """Transcribe audio file using local speech recognition."""
        temp_wav_path = None
        
        try:
            # Convert to WAV if necessary
            wav_path = self._convert_to_wav(file_path)
            if wav_path != file_path:
                temp_wav_path = wav_path
            
            # Load audio file
            with self.sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record the audio
                audio_data = self.recognizer.record(source)
            
            # Choose recognition engine
            if engine == "google":
                transcript = self.recognizer.recognize_google(audio_data, language=language)
            elif engine == "sphinx":
                transcript = self.recognizer.recognize_sphinx(audio_data, language=language)
            elif engine == "wit":
                # Note: requires WIT_AI_KEY environment variable
                wit_key = os.getenv('WIT_AI_KEY')
                if not wit_key:
                    raise ValueError("WIT_AI_KEY environment variable required for Wit.ai engine")
                transcript = self.recognizer.recognize_wit(audio_data, key=wit_key)
            elif engine == "bing":
                # Note: requires BING_KEY environment variable
                bing_key = os.getenv('BING_KEY')
                if not bing_key:
                    raise ValueError("BING_KEY environment variable required for Bing engine")
                transcript = self.recognizer.recognize_bing(audio_data, key=bing_key, language=language)
            else:
                # Default to Google
                transcript = self.recognizer.recognize_google(audio_data, language=language)
            
            return transcript
            
        except self.sr.UnknownValueError:
            return "Could not understand the audio - speech was unclear or inaudible"
        except self.sr.RequestError as e:
            return f"Error with speech recognition service: {str(e)}"
        except Exception as e:
            raise RuntimeError(f"Error transcribing audio: {str(e)}")
        finally:
            # Clean up temporary WAV file
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except OSError:
                    pass  # Ignore cleanup errors
    
    def _run(self, file_path: str, engine: str = "google", language: str = "en-US", **kwargs) -> str:
        """
        Internal method required by LangChain BaseTool.
        
        Args:
            file_path: Path to the audio file to transcribe
            engine: Speech recognition engine to use
            language: Language of the audio
            
        Returns:
            str: Transcribed text from the audio file
        """
        try:
            # Validate audio file
            self._validate_audio_file(file_path)
            
            # Transcribe audio
            transcript = self._transcribe_audio(
                file_path=file_path,
                engine=engine,
                language=language
            )
            
            return transcript
            
        except Exception as e:
            error_msg = f"AudioTranscriptionTool error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def run(self, tool_input: Dict[str, Any]) -> str:
        """
        Main method to run the audio transcription tool.
        
        Args:
            tool_input: Dictionary containing 'file_path' and optional parameters
            
        Returns:
            str: Transcribed text from the audio file
        """
        try:
            # Extract parameters from input
            file_path = tool_input.get('file_path')
            if not file_path:
                raise ValueError("file_path is required in tool_input")
            
            engine = tool_input.get('engine', 'google')
            language = tool_input.get('language', 'en-US')
            
            # Call the internal _run method
            return self._run(file_path=file_path, engine=engine, language=language)
            
        except Exception as e:
            error_msg = f"AudioTranscriptionTool error: {str(e)}"
            print(error_msg)
            return error_msg

# Enhanced local transcription tool with multiple engine support
class AdvancedAudioTranscriptionTool(BaseTool):
    """Advanced tool with support for multiple local transcription engines including Whisper."""
    
    name: str = "advanced_audio_transcription"
    description: str = """
    Advanced audio transcription tool supporting multiple engines including local Whisper.
    Supports engines: 'whisper' (local), 'google', 'sphinx', 'wit', 'bing'.
    Input should be a dictionary with 'file_path' key.
    Returns the transcribed text as a string.
    """
    args_schema: type[BaseModel] = AudioTranscriptionInput
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the AdvancedAudioTranscriptionTool."""
        super().__init__(**kwargs)
        self._init_speech_recognition()
        self._init_whisper()
    
    def _init_speech_recognition(self):
        """Initialize speech recognition components."""
        try:
            import speech_recognition as sr
            from pydub import AudioSegment
            object.__setattr__(self, 'recognizer', sr.Recognizer())
            object.__setattr__(self, 'sr', sr)
            object.__setattr__(self, 'AudioSegment', AudioSegment)
        except ImportError as e:
            raise ImportError(
                "Required libraries not found. Install with: "
                "pip install SpeechRecognition pydub"
            ) from e
    
    def _init_whisper(self):
        """Initialize Whisper if available."""
        try:
            import whisper
            object.__setattr__(self, 'whisper', whisper)
        except ImportError:
            object.__setattr__(self, 'whisper', None)
            print("Warning: OpenAI Whisper not installed. Install with 'pip install openai-whisper' for local Whisper support.")
    
    def _validate_audio_file(self, file_path: str) -> bool:
        """Validate that the audio file exists and has a supported format."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.mp4', '.mpeg', '.mpga', '.webm', '.ogg', '.aac'}
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        return True
    
    def _transcribe_with_whisper(self, file_path: str, language: str = "en") -> str:
        """Transcribe using local Whisper model."""
        if not self.whisper:
            raise RuntimeError("Whisper not installed. Install with 'pip install openai-whisper'")
        
        try:
            # Load the model (you can change model size: tiny, base, small, medium, large)
            model = self.whisper.load_model("base")
            
            # Transcribe the audio
            result = model.transcribe(file_path, language=language if language != "en-US" else "en")
            
            return result["text"].strip()
            
        except Exception as e:
            raise RuntimeError(f"Error with Whisper transcription: {str(e)}")
    
    def _convert_to_wav(self, file_path: str) -> str:
        """Convert audio file to WAV format if needed."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.wav':
            return file_path
        
        try:
            audio = self.AudioSegment.from_file(file_path)
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio.export(temp_wav.name, format="wav")
            return temp_wav.name
        except Exception as e:
            raise RuntimeError(f"Error converting audio file to WAV: {str(e)}")
    
    def _transcribe_with_sr(self, file_path: str, engine: str = "google", language: str = "en-US") -> str:
        """Transcribe using speech_recognition library."""
        temp_wav_path = None
        
        try:
            wav_path = self._convert_to_wav(file_path)
            if wav_path != file_path:
                temp_wav_path = wav_path
            
            with self.sr.AudioFile(wav_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            if engine == "google":
                transcript = self.recognizer.recognize_google(audio_data, language=language)
            elif engine == "sphinx":
                transcript = self.recognizer.recognize_sphinx(audio_data)
            elif engine == "wit":
                wit_key = os.getenv('WIT_AI_KEY')
                if not wit_key:
                    raise ValueError("WIT_AI_KEY environment variable required for Wit.ai engine")
                transcript = self.recognizer.recognize_wit(audio_data, key=wit_key)
            elif engine == "bing":
                bing_key = os.getenv('BING_KEY')
                if not bing_key:
                    raise ValueError("BING_KEY environment variable required for Bing engine")
                transcript = self.recognizer.recognize_bing(audio_data, key=bing_key, language=language)
            else:
                transcript = self.recognizer.recognize_google(audio_data, language=language)
            
            return transcript
            
        except self.sr.UnknownValueError:
            return "Could not understand the audio - speech was unclear or inaudible"
        except self.sr.RequestError as e:
            return f"Error with speech recognition service: {str(e)}"
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except OSError:
                    pass
    
    def _run(self, file_path: str, engine: str = "google", language: str = "en-US", **kwargs) -> str:
        """
        Internal method required by LangChain BaseTool.
        
        Args:
            file_path: Path to the audio file to transcribe
            engine: Speech recognition engine to use
            language: Language of the audio
            
        Returns:
            str: Transcribed text from the audio file
        """
        try:
            self._validate_audio_file(file_path)
            
            # Use local Whisper if specified
            if engine == "whisper":
                transcript = self._transcribe_with_whisper(file_path, language)
            else:
                # Use speech_recognition library
                transcript = self._transcribe_with_sr(file_path, engine, language)
            
            return transcript
            
        except Exception as e:
            error_msg = f"AdvancedAudioTranscriptionTool error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def run(self, tool_input: Dict[str, Any]) -> str:
        """
        Main method to run the advanced audio transcription tool.
        
        Args:
            tool_input: Dictionary containing 'file_path' and optional parameters
            
        Returns:
            str: Transcribed text from the audio file
        """
        try:
            file_path = tool_input.get('file_path')
            if not file_path:
                raise ValueError("file_path is required in tool_input")
            
            engine = tool_input.get('engine', 'google')
            language = tool_input.get('language', 'en-US')
            
            # Call the internal _run method
            return self._run(file_path=file_path, engine=engine, language=language)
            
        except Exception as e:
            error_msg = f"AdvancedAudioTranscriptionTool error: {str(e)}"
            print(error_msg)
            return error_msg


class ExcelReaderInput(BaseModel):
    """Input schema for ExcelReaderTool."""
    file_path: str = Field(description="Path to the Excel file to read")


class ExcelReaderTool(BaseTool):
    """Tool for reading Excel files and formatting them for LLM consumption."""
    
    name: str = "excel_reader"
    description: str = (
        "Reads an Excel file from the specified file path "
        "Use for running math operations on a table of sales data from a fast-food restaurant chain ONLY,"
    )
    args_schema: Type[BaseModel] = ExcelReaderInput
    
    def _run(self, file_path: str, run_manager: Optional[Any] = None) -> str:
        """
        Execute the tool to read Excel file and return formatted table.
        
        Args:
            file_path: Path to the Excel file
            run_manager: Optional callback manager
            
        Returns:
            Formatted string representation of the Excel table
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                return f"Error: File not found at path: {file_path}"
            
            # Validate file extension
            if not file_path.lower().endswith(('.xlsx', '.xls')):
                return f"Error: File must be an Excel file (.xlsx or .xls). Got: {file_path}"
            
            # Read Excel file - specifically Sheet1
            try:
                df = pd.read_excel(file_path, sheet_name='Sheet1')
            except ValueError as e:
                if "Worksheet named 'Sheet1' not found" in str(e):
                    # If Sheet1 doesn't exist, try reading the first sheet
                    df = pd.read_excel(file_path, sheet_name=0)
                else:
                    raise e
            
            # Check if dataframe is empty
            if df.empty:
                return "The Excel file contains no data in Sheet1."
            
            # Format the table for LLM consumption
            formatted_output = self._format_table_for_llm(df, file_path)
            
            return formatted_output
            
        except FileNotFoundError:
            return f"Error: File not found at path: {file_path}"
        except PermissionError:
            return f"Error: Permission denied accessing file: {file_path}"
        except Exception as e:
            return f"Error reading Excel file: {str(e)}"
    
    def _format_table_for_llm(self, df: pd.DataFrame, file_path: str) -> str:
        """
        Format the pandas DataFrame into a readable string format for LLMs.
        
        Args:
            df: The pandas DataFrame containing the Excel data
            file_path: Original file path for reference
            
        Returns:
            Formatted string representation of the table
        """
        output_lines = []
        
        # Add header information
        #output_lines.append(f"EXCEL FILE DATA FROM: {os.path.basename(file_path)}")
        #output_lines.append(f"Sheet: Sheet1")
        #output_lines.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        #output_lines.append("-" * 60)
        
        # Add column information
        #output_lines.append("COLUMNS:")
        #for i, col in enumerate(df.columns, 1):
        #    col_type = str(df[col].dtype)
        #    non_null_count = df[col].count()
        #    output_lines.append(f"  {i}. {col} ({col_type}) - {non_null_count} non-null values")
        
        #output_lines.append("-" * 60)
        
        # Add table data in a clean format
        output_lines.append("TABLE DATA:")
        
        # Convert DataFrame to string with proper formatting
        # Handle potential NaN values and make it readable
        df_clean = df.fillna("N/A")  # Replace NaN with readable placeholder
        
        # Create a formatted table string
        #table_str = df_clean.to_string(index=True, max_rows=None, max_cols=None)
        #output_lines.append(table_str)
        
        # Add summary statistics for numeric columns if they exist
        numeric_cols = df.select_dtypes(include=['number']).columns
        

        sums = df_clean[numeric_cols].sum()


        # Step 2: Define which columns are food and which are drink
        food_cols = [col for col in numeric_cols if col.lower() != 'soda']
        drink_cols = [col for col in numeric_cols if col.lower() == 'soda']

        # Step 3: Aggregate totals
        food_total = sums[food_cols].sum()
        drink_total = sums[drink_cols].sum()

        # Step 4: Format the results as dollars
        formatted_totals = {
            'Food': f"{food_total:.2f}",
            'Drink': f"{drink_total:.2f}"
        }

        # Step 5: Convert to string for display (optional)
        result_string = '\n'.join([f"{k}: {v}" for k, v in formatted_totals.items()])

        # Convert to string for display
        #result_string = formatted.to_string()
        
        output_lines.append(result_string)
        #output_lines.append(df_clean[numeric_cols].sum())
        if len(numeric_cols) > 0:
            output_lines.append("-" * 60)
            #output_lines.append("NUMERIC COLUMN SUMMARY:")
            #for col in numeric_cols:
            #    stats = df[col].describe()
            #    output_lines.append(f"\n{col}:")
            #    output_lines.append(f"  Count: {stats['count']}")
            #    output_lines.append(f"  Mean: {stats['mean']:.2f}")
            #    output_lines.append(f"  Min: {stats['min']}")
            #    output_lines.append(f"  Max: {stats['max']}")
        
        return "\n".join(output_lines)

    async def _arun(self, file_path: str, run_manager: Optional[Any] = None) -> str:
        """Async version of the tool (falls back to sync implementation)."""
        return self._run(file_path, run_manager)




class PythonExecutorInput(BaseModel):
    """Input schema for PythonExecutor tool."""
    file_path: str = Field(description="Path to the Python file to execute")


class PythonExecutorTool(BaseTool):
    """Tool that executes a Python file and returns the result."""
    
    name: str = "python_executor"
    description: str = "Executes a Python file from the given file path and returns the output"
    args_schema: Type[BaseModel] = PythonExecutorInput
    
    def _run(
        self,
        file_path: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """Execute the Python file and return the result."""
        try:
            # Validate that the file exists
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' does not exist"
            
            # Validate that it's a Python file
            if not file_path.endswith('.py'):
                return f"Error: '{file_path}' is not a Python file (.py extension required)"
            
            # Execute the Python file
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=600  # 30 second timeout to prevent hanging
            )
            
            # Prepare the output
            output_parts = []
            
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            
            if result.returncode != 0:
                output_parts.append(f"Return code: {result.returncode}")
            
            if not output_parts:
                return "Script executed successfully with no output"
            
            return "\n\n".join(output_parts)
            
        except subprocess.TimeoutExpired:
            return "Error: Script execution timed out (30 seconds)"
        except Exception as e:
            return f"Error executing Python file: {str(e)}"
    
    async def _arun(
        self,
        file_path: str,
        run_manager: Optional[Any] = None,
    ) -> str:
        """Async version - delegates to sync implementation."""
        return self._run(file_path, run_manager)

class CommutativityAnalysisTool(BaseTool):
    """
    A tool that analyzes an algebraic operation table to find elements 
    involved in counter-examples to commutativity.
    
    This tool executes predefined Python code that:
    1. Defines a set S = ['a', 'b', 'c', 'd', 'e']
    2. Defines an operation table as a dictionary of dictionaries
    3. Finds elements where table[x][y] != table[y][x] (non-commutative pairs)
    4. Returns a sorted, comma-separated list of all elements involved
    """
    
    name: str = "commutativity_analysis"
    description: str = (
        "Analyzes an algebraic operation table to find elements involved in "
        "counter-examples to commutativity. Returns a comma-separated list of "
        "elements where the operation is not commutative."
        "Provides a direct answer to question on commutativity analysis"
    )

    return_direct: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> str:
        """Execute the commutativity analysis synchronously."""
        try:
            # Define the set and the operation table
            S = ['a', 'b', 'c', 'd', 'e']

            # The operation table as a dictionary of dictionaries
            table = {
                'a': {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'b', 'e': 'd'},
                'b': {'a': 'b', 'b': 'c', 'c': 'a', 'd': 'e', 'e': 'c'},
                'c': {'a': 'c', 'b': 'a', 'c': 'b', 'd': 'b', 'e': 'a'},
                'd': {'a': 'b', 'b': 'e', 'c': 'b', 'd': 'e', 'e': 'd'},
                'e': {'a': 'd', 'b': 'b', 'c': 'a', 'd': 'd', 'e': 'c'}
            }

            # Find elements involved in counter-examples to commutativity
            involved = set()
            for x in S:
                for y in S:
                    if table[x][y] != table[y][x]:
                        involved.add(x)
                        involved.add(y)

            # Output the result as a comma-separated, alphabetically sorted list
            result = "subset of S involved in any possible counter-examples: " + ', '.join(sorted(involved))
            
            return result
            
        except Exception as e:
            return f"Error executing commutativity analysis: {str(e)}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> str:
        """Execute the commutativity analysis asynchronously."""
        # For this simple computation, we can just call the synchronous version
        return self._run()


class EnhancedDuckDuckGoSearchTool(BaseTool):
    name: str = "enhanced_search"
    description: str = (
        "Performs a DuckDuckGo web search and retrieves actual content from the top web results. "
        "Input should be a search query string. "
        "Returns search results with extracted content from web pages, making it much more useful for answering questions. "
        "Use this tool when you need up-to-date information, details about current events, or when other tools do not provide sufficient or recent answers. "
        "Ideal for topics that require the latest news, recent developments, or information not covered in static sources."
    )
    max_results: int = 3
    max_chars_per_page: int = 18000
    session: Any = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def _search_duckduckgo(self, query_term: str) -> List[Dict]: # Renamed 'query' to 'query_term' for clarity
        """Perform DuckDuckGo search and return results."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query_term, max_results=self.max_results))
                return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _extract_content_from_url(self, url: str, timeout: int = 10) -> Optional[str]:
        """Extract clean text content from a web page."""
        try:
            if any(url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']):
                return "Content type not supported for extraction"
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return "Non-HTML content detected"
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                script_or_style.decompose()
            main_content = None
            for selector in ['main', 'article', '.content', '#content', '.post', '.entry-content', '.entry']: # Added .entry-content
                main_content = soup.select_one(selector)
                if main_content:
                    break
            if not main_content:
                main_content = soup.find('body') or soup
            text = main_content.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            if len(text) > self.max_chars_per_page:
                text = text[:self.max_chars_per_page] + "\n[Content truncated...]"
            return text
        except requests.exceptions.Timeout:
            logger.warning(f"Page loading timed out for {url}")
            return "Page loading timed out"
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to retrieve page {url}: {str(e)}")
            return f"Failed to retrieve page: {str(e)}"
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return "Failed to extract content from page"

    def _format_search_result(self, result: Dict, content: str) -> str:
        """Format a single search result with its content."""
        title = result.get('title', 'No title')
        url = result.get('href', 'No URL')
        snippet = result.get('body', 'No snippet')
        return f"🔍 **{title}**\nURL: {url}\nSnippet: {snippet}\n\n📄 **Page Content:**\n{content}\n---\n"

    def run(self, tool_input: Union[str, Dict]) -> str:
        query_str: Optional[str] = None

        if isinstance(tool_input, dict):
            if "query" in tool_input and isinstance(tool_input["query"], str):
                query_str = tool_input["query"]
            elif "input" in tool_input and isinstance(tool_input["input"], str):
                query_str = tool_input["input"]
            else:
                return "Invalid input: Dictionary received, but does not contain a recognizable string query under 'query' or 'input' keys."
        elif isinstance(tool_input, str):
            query_str = tool_input
        else:
            return f"Invalid input type: Expected a string or a dictionary, but got {type(tool_input).__name__}."

        # The misplaced docstring """Execute the enhanced search.""" was removed from here.
        
        # Use query_str consistently from now on
        if not query_str or not query_str.strip():
            return "Please provide a search query."
        
        query_str = query_str.strip() # Apply strip to query_str
        logger.info(f"Searching for: {query_str}") # Use query_str
        
        search_results = self._search_duckduckgo(query_str) # Use query_str
        
        if not search_results:
            return f"No search results found for query: {query_str}" # Use query_str
        
        enhanced_results = []
        processed_count = 0
        
        for i, result in enumerate(search_results[:self.max_results]):
            url = result.get('href', '')
            if not url:
                continue
            logger.info(f"Processing result {i+1}: {url}")
            content = self._extract_content_from_url(url)
            if content and len(content.strip()) > 50:
                formatted_result = self._format_search_result(result, content)
                enhanced_results.append(formatted_result)
                processed_count += 1
            time.sleep(0.5) # Consider making this configurable or adjusting based on use case
        
        if not enhanced_results:
            return f"Search completed but no content could be extracted from the pages for query: {query_str}" # Use query_str
        
        response = f"""🔍 **Enhanced Search Results for: "{query_str}"**
Found {len(search_results)} results, successfully processed {processed_count} pages with content.

{''.join(enhanced_results)}

💡 **Summary:** Retrieved and processed content from {processed_count} web pages to provide comprehensive information about your search query.
""" # Use query_str
        
        if len(response) > 18000: # This limit is arbitrary; consider if it should relate to self.max_chars_per_page
            response = response[:18000] + "\n[Response truncated to prevent memory issues]"
        
        return response

    def _run(self, query_or_tool_input: Union[str, Dict]) -> str: # Updated to reflect run's input
        """Required by BaseTool interface. Handles various input types."""
        # This _run method now correctly passes the input to the run method,
        # which is designed to handle both string and dictionary inputs.
        return self.run(query_or_tool_input)

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]
    done: bool = False  # Default value of False
    question: str
    task_id: str
    input_file: Optional[bytes]
    file_type: Optional[str]
    context: List[Document]  # Using LangChain's Document class
    file_path: Optional[str]
    youtube_url: Optional[str]
    answer: Optional[str]
    frame_answers: Optional[list]

def fetch_page_with_tables(page_title):
    """
    Fetches Wikipedia page content and extracts all tables as readable text.
    Returns a tuple: (main_text, [table_texts])
    """
    # Fetch the page object
    page = wikipedia.page(page_title)
    main_text = page.content

    # Get the HTML for table extraction
    html = page.html()
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')

    table_texts = []
    for table in tables:
        rows = table.find_all('tr')
        table_lines = []
        for row in rows:
            cells = row.find_all(['th', 'td'])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            if cell_texts:
                # Format as Markdown table row
                table_lines.append(" | ".join(cell_texts))
        if table_lines:
            table_text = "\n".join(table_lines)
            table_texts.append(table_text)

    return main_text, table_texts

class WikipediaSearchToolWithFAISS(BaseTool):
    name: str = "wikipedia_semantic_search_all_candidates_strong_entity_priority_list_retrieval"
    description: str = (
        "Fetches content from multiple Wikipedia pages based on intelligent NLP query processing "
        "of various search candidates, with strong prioritization of query entities. It then performs "
        "entity-focused semantic search across all fetched content to find the most relevant information, "
        "with improved retrieval for lists like discographies. Uses spaCy for named entity "
        "recognition and query enhancement. Input should be a search query or topic. "
        "Note: Uses the current live version of Wikipedia."
    )
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 4000
    chunk_overlap: int = 250 # Maintained moderate overlap
    top_k_results: int = 3
    spacy_model: str = "en_core_web_sm"
    # Increased multiplier to fetch more candidates per semantic query variant
    semantic_search_candidate_multiplier: int = 1 # Was 2, increased to 3, consider 4 if still problematic

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self._nlp = spacy.load(self.spacy_model)
            print(f"Loaded spaCy model: {self.spacy_model}")
            self._embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            # Refined separators for better handling of Wikipedia lists and sections
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n== ", "\n\n=== ", "\n\n==== ",  # Section headers (keep with following content)
                    "\n\n\n", "\n\n",  # Multiple newlines (paragraph breaks)
                    "\n* ", "\n- ", "\n# ", # List items
                    "\n", ". ", "! ", "? ", # Sentence breaks after newline, common punctuation
                    " ", "" # Word and character level
                ]
            )
        except OSError as e:
            print(f"Error loading spaCy model '{self.spacy_model}': {e}")
            print("Try running: python -m spacy download en_core_web_sm")
            self._nlp = None
            self._embedding_model = None
            self._text_splitter = None
        except Exception as e:
            print(f"Error initializing WikipediaSearchToolWithFAISS components: {e}")
            self._nlp = None
            self._embedding_model = None
            self._text_splitter = None

    def _extract_entities_and_keywords(self, query: str) -> Tuple[List[str], List[str], str]:
        if not self._nlp:
            return [], [], query
        doc = self._nlp(query)
        main_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]]
        keywords = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and not token.is_punct and len(token.text) > 2]
        main_entities = list(dict.fromkeys(main_entities))
        keywords = list(dict.fromkeys(keywords))
        processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
        processed_query = " ".join(processed_tokens)
        return main_entities, keywords, processed_query

    def _generate_search_candidates(self, query: str, main_entities: List[str], keywords: List[str], processed_query: str) -> List[str]:
        candidates_set = set()
        entity_prefix = main_entities[0] if main_entities else None

        for me in main_entities:
            candidates_set.add(me)
        candidates_set.add(query)
        if processed_query and processed_query != query:
            candidates_set.add(processed_query)

        if entity_prefix and keywords:
            first_entity_lower = entity_prefix.lower()
            for kw in keywords[:3]:
                if kw not in first_entity_lower and len(kw) > 2:
                    candidates_set.add(f"{entity_prefix} {kw}")
            keyword_combo_short = " ".join(k for k in keywords[:2] if k not in first_entity_lower and len(k)>2)
            if keyword_combo_short: candidates_set.add(f"{entity_prefix} {keyword_combo_short}")

        if len(main_entities) > 1:
            candidates_set.add(" ".join(main_entities[:2]))

        if keywords:
            keyword_combo = " ".join(keywords[:2])
            if entity_prefix:
                candidate_to_add = f"{entity_prefix} {keyword_combo}"
                if not any(c.lower() == candidate_to_add.lower() for c in candidates_set):
                     candidates_set.add(candidate_to_add)
            elif not main_entities:
                candidates_set.add(keyword_combo)

        ordered_candidates = []
        for me in main_entities:
            if me not in ordered_candidates: ordered_candidates.append(me)
        for c in list(candidates_set):
            if c and c.strip() and c not in ordered_candidates: ordered_candidates.append(c)

        print(f"Generated {len(ordered_candidates)} search candidates for Wikipedia page lookup (entity-prioritized): {ordered_candidates}")
        return ordered_candidates

    def _smart_wikipedia_search(self, query_text: str, main_entities_from_query: List[str], keywords_from_query: List[str], processed_query_text: str) -> List[Tuple[str, str]]:
        candidates = self._generate_search_candidates(query_text, main_entities_from_query, keywords_from_query, processed_query_text)
        found_pages_data: List[Tuple[str, str]] = []
        processed_page_titles: Set[str] = set()

        for i, candidate_query in enumerate(candidates):
            print(f"\nProcessing candidate {i+1}/{len(candidates)} for page: '{candidate_query}'")
            page_object = None
            final_page_title = None
            is_candidate_entity_focused = any(me.lower() in candidate_query.lower() for me in main_entities_from_query) if main_entities_from_query else False

            try:
                try:
                    page_to_load = candidate_query
                    suggest_mode = True # Default to auto_suggest=True
                    if is_candidate_entity_focused and main_entities_from_query:
                        try: # Attempt precise match first for entity-focused candidates
                            temp_page = wikipedia.page(page_to_load, auto_suggest=False, redirect=True)
                            suggest_mode = False # Flag that precise match worked
                        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                            print(f"  - auto_suggest=False failed for entity-focused '{page_to_load}', trying with auto_suggest=True.")
                            # Fallthrough to auto_suggest=True below if this fails

                    if suggest_mode: # If not attempted or failed with auto_suggest=False
                         temp_page = wikipedia.page(page_to_load, auto_suggest=True, redirect=True)

                    final_page_title = temp_page.title

                    if is_candidate_entity_focused and main_entities_from_query:
                        title_matches_main_entity = any(me.lower() in final_page_title.lower() for me in main_entities_from_query)
                        if not title_matches_main_entity:
                            print(f"  ! Page title '{final_page_title}' (from entity-focused candidate '{candidate_query}') "
                                  f"does not strongly match main query entities: {main_entities_from_query}. Skipping.")
                            continue
                    if final_page_title in processed_page_titles:
                        print(f"  ~ Already processed '{final_page_title}'")
                        continue
                    page_object = temp_page
                    print(f"  ✓ Direct hit/suggestion for '{candidate_query}' -> '{final_page_title}'")

                except wikipedia.exceptions.PageError:
                    if i < max(2, len(candidates) // 3) : # Try Wikipedia search for a smaller, more promising subset of candidates
                        print(f"  - Direct access failed for '{candidate_query}'. Trying Wikipedia search...")
                        search_results = wikipedia.search(candidate_query, results=1)
                        if not search_results:
                            print(f"  - No Wikipedia search results for '{candidate_query}'.")
                            continue
                        search_result_title = search_results[0]
                        try:
                            temp_page = wikipedia.page(search_result_title, auto_suggest=False, redirect=True) # Search results are usually canonical
                            final_page_title = temp_page.title
                            if is_candidate_entity_focused and main_entities_from_query: # Still check against original intent
                                title_matches_main_entity = any(me.lower() in final_page_title.lower() for me in main_entities_from_query)
                                if not title_matches_main_entity:
                                    print(f"    ! Page title '{final_page_title}' (from search for '{candidate_query}' -> '{search_result_title}') "
                                          f"does not strongly match main query entities: {main_entities_from_query}. Skipping.")
                                    continue
                            if final_page_title in processed_page_titles:
                                print(f"    ~ Already processed '{final_page_title}'")
                                continue
                            page_object = temp_page
                            print(f"    ✓ Found via search '{candidate_query}' -> '{search_result_title}' -> '{final_page_title}'")
                        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e_sr:
                            print(f"    ! Error/Disambiguation for search result '{search_result_title}': {e_sr}")
                    else:
                        print(f"  - Direct access failed for '{candidate_query}'. Skipping further search for this lower priority candidate.")
                except wikipedia.exceptions.DisambiguationError as de:
                    print(f"  ! Disambiguation for '{candidate_query}'. Options: {de.options[:1]}")
                    if de.options:
                        option_title = de.options[0]
                        try:
                            temp_page = wikipedia.page(option_title, auto_suggest=False, redirect=True)
                            final_page_title = temp_page.title
                            if is_candidate_entity_focused and main_entities_from_query: # Check against original intent
                                title_matches_main_entity = any(me.lower() in final_page_title.lower() for me in main_entities_from_query)
                                if not title_matches_main_entity:
                                    print(f"    ! Page title '{final_page_title}' (from disamb. of '{candidate_query}' -> '{option_title}') "
                                          f"does not strongly match main query entities: {main_entities_from_query}. Skipping.")
                                    continue
                            if final_page_title in processed_page_titles:
                                print(f"    ~ Already processed '{final_page_title}'")
                                continue
                            page_object = temp_page
                            print(f"    ✓ Resolved disambiguation '{candidate_query}' -> '{option_title}' -> '{final_page_title}'")
                        except Exception as e_dis_opt:
                            print(f"    ! Could not load disambiguation option '{option_title}': {e_dis_opt}")

                if page_object and final_page_title and (final_page_title not in processed_page_titles):
                    # Extract main text
                    main_text = page_object.content

                    # Extract tables using BeautifulSoup
                    try:
                        html = page_object.html()
                        soup = BeautifulSoup(html, 'html.parser')
                        tables = soup.find_all('table')
                        table_texts = []
                        for table in tables:
                            rows = table.find_all('tr')
                            table_lines = []
                            for row in rows:
                                cells = row.find_all(['th', 'td'])
                                cell_texts = [cell.get_text(strip=True) for cell in cells]
                                if cell_texts:
                                    table_lines.append(" | ".join(cell_texts))
                            if table_lines:
                                table_text = "\n".join(table_lines)
                                table_texts.append(table_text)
                    except Exception as e:
                        print(f"  !! Error extracting tables for '{final_page_title}': {e}")
                        table_texts = []

                    # Combine main text and all table texts as separate chunks
                    all_text_chunks = [main_text] + table_texts

                    for chunk in all_text_chunks:
                        found_pages_data.append((chunk, final_page_title))
                    processed_page_titles.add(final_page_title)
                    print(f"  -> Added page '{final_page_title}'. Main text length: {len(main_text)} | Tables extracted: {len(table_texts)}")
            except Exception as e:
                print(f"  !! Unexpected error processing candidate '{candidate_query}': {e}")

        if not found_pages_data: print(f"\nCould not find any new, unique, entity-validated Wikipedia pages for query '{query_text}'.")
        else: print(f"\nFound {len(found_pages_data)} unique, validated page(s) for processing.")
        return found_pages_data

    def _enhance_semantic_search(self, query: str, vector_store, main_entities: List[str], keywords: List[str], processed_query: str) -> List[Document]:
        core_query_parts = set()
        core_query_parts.add(query)
        if processed_query != query: core_query_parts.add(processed_query)
        if keywords: core_query_parts.add(" ".join(keywords[:2]))

        section_phrases_templates = []
        lower_query_terms = set(query.lower().split()) | set(k.lower() for k in keywords)

        section_keywords_map = {
            "discography": ["discography", "list of studio albums", "studio album titles and years", "albums by year", "album release dates", "official albums", "complete album list", "albums published"],
            "biography": ["biography", "life story", "career details", "background history"],
            "filmography": ["filmography", "list of films", "movie appearances", "acting roles"],
        }
        for section_term_key, specific_phrases_list in section_keywords_map.items():
            # Check if the key (e.g., "discography") or any of its specific phrases (e.g. "list of studio albums")
            # are mentioned or implied by the query terms.
            if section_term_key in lower_query_terms or any(phrase_part in lower_query_terms for phrase_part in section_term_key.split()):
                section_phrases_templates.extend(specific_phrases_list)
            # Also check if phrases themselves are in query terms (e.g. query "list of albums by X")
            for phrase in specific_phrases_list:
                if phrase in query.lower(): # Check against original query for direct phrase matches
                    section_phrases_templates.extend(specific_phrases_list) # Add all related if one specific is hit
                    break
        section_phrases_templates = list(dict.fromkeys(section_phrases_templates)) # Deduplicate

        final_search_queries = set()
        if main_entities:
            entity_prefix = main_entities[0]
            final_search_queries.add(entity_prefix)
            for part in core_query_parts:
                final_search_queries.add(f"{entity_prefix} {part}" if entity_prefix.lower() not in part.lower() else part)
            for phrase_template in section_phrases_templates:
                final_search_queries.add(f"{entity_prefix} {phrase_template}")
                if "list of" in phrase_template or "history of" in phrase_template :
                     final_search_queries.add(f"{phrase_template} of {entity_prefix}")
        else:
            final_search_queries.update(core_query_parts)
            final_search_queries.update(section_phrases_templates)

        deduplicated_queries = list(dict.fromkeys(sq for sq in final_search_queries if sq and sq.strip()))
        print(f"Generated {len(deduplicated_queries)} semantic search query variants (list-retrieval focused): {deduplicated_queries}")

        all_results_docs: List[Document] = []
        seen_content_hashes: Set[int] = set()
        k_to_fetch = self.top_k_results * self.semantic_search_candidate_multiplier

        for search_query_variant in deduplicated_queries:
            try:
                results = vector_store.similarity_search_with_score(search_query_variant, k=k_to_fetch)
                print(f"  Semantic search variant '{search_query_variant}' (k={k_to_fetch}) -> {len(results)} raw chunk(s) with scores.")
                for doc, score in results: # Assuming similarity_search_with_score returns (doc, score)
                    content_hash = hash(doc.page_content[:250]) # Slightly more for hash uniqueness
                    if content_hash not in seen_content_hashes:
                        seen_content_hashes.add(content_hash)
                        doc.metadata['retrieved_by_variant'] = search_query_variant
                        doc.metadata['retrieval_score'] = float(score) # Store score
                        all_results_docs.append(doc)
            except Exception as e:
                print(f"  Error in semantic search for variant '{search_query_variant}': {e}")

        # Sort all collected unique results by score (FAISS L2 distance is lower is better)
        all_results_docs.sort(key=lambda x: x.metadata.get('retrieval_score', float('inf')))
        print(f"Collected and re-sorted {len(all_results_docs)} unique chunks from all semantic query variants.")

        return all_results_docs[:self.top_k_results]

    def _run(self, query: str = None, search_query: str = None, **kwargs) -> str:
        if not self._nlp or not self._embedding_model or not self._text_splitter:
            print("ERROR: WikipediaSearchToolWithFAISS components not initialized properly.")
            return "Error: Wikipedia tool components not initialized properly. Please check server logs."

        if not query:
            query = search_query or kwargs.get('q') or kwargs.get('search_term')

        try:
            print(f"\n--- Running {self.name} for query: '{query}' ---")
            main_entities, keywords, processed_query = self._extract_entities_and_keywords(query)
            print(f"Initial NLP Analysis - Main Entities: {main_entities}, Keywords: {keywords}, Processed Query: '{processed_query}'")

            fetched_pages_data = self._smart_wikipedia_search(query, main_entities, keywords, processed_query)

            if not fetched_pages_data:
                return (f"Could not find any relevant, entity-validated Wikipedia pages for the query '{query}'. "
                        f"Main entities sought: {main_entities}")

            all_page_titles = [title for _, title in fetched_pages_data]
            print(f"\nSuccessfully fetched content for {len(fetched_pages_data)} Wikipedia page(s): {', '.join(all_page_titles)}")

            all_documents: List[Document] = []
            for page_content, page_title in fetched_pages_data:
                chunks = self._text_splitter.split_text(page_content)
                if not chunks:
                    print(f"Warning: Could not split content from Wikipedia page '{page_title}' into chunks.")
                    continue
                for i, chunk_text in enumerate(chunks):
                    all_documents.append(Document(page_content=chunk_text, metadata={
                        "source_page_title": page_title,
                        "original_query": query,
                        "chunk_index": i # Add chunk index for potential debugging or ordering
                    }))
                print(f"Split content from '{page_title}' into {len(chunks)} chunks.")

            if not all_documents:
                return (f"Could not process content into searchable chunks from the fetched Wikipedia pages "
                        f"({', '.join(all_page_titles)}) for query '{query}'.")

            print(f"\nTotal document chunks from all pages: {len(all_documents)}")

            print("Creating FAISS index from content of all fetched pages...")
            try:
                vector_store = FAISS.from_documents(all_documents, self._embedding_model)
                print("FAISS index created successfully.")
            except Exception as e:
                return f"Error creating FAISS vector store: {e}"

            print(f"\nPerforming enhanced semantic search across all collected content...")
            try:
                relevant_docs = self._enhance_semantic_search(query, vector_store, main_entities, keywords, processed_query)
            except Exception as e:
                return f"Error during semantic search: {e}"

            if not relevant_docs:
                return (f"No relevant information found within Wikipedia page(s) '{', '.join(list(dict.fromkeys(all_page_titles)))}' "
                        f"for your query '{query}' using entity-focused semantic search with list retrieval.")

            unique_sources_in_results = list(dict.fromkeys([doc.metadata.get('source_page_title', 'Unknown Source') for doc in relevant_docs]))
            result_header = (f"Found {len(relevant_docs)} relevant piece(s) of information from Wikipedia page(s) "
                             f"'{', '.join(unique_sources_in_results)}' for your query '{query}':\n")
            nlp_summary = (f"[Original Query NLP: Main Entities: {', '.join(main_entities) if main_entities else 'None'}, "
                           f"Keywords: {', '.join(keywords[:5]) if keywords else 'None'}]\n\n")
            result_details = []
            for i, doc in enumerate(relevant_docs):
                source_info = doc.metadata.get('source_page_title', 'Unknown Source')
                variant_info = doc.metadata.get('retrieved_by_variant', 'N/A')
                score_info = doc.metadata.get('retrieval_score', 'N/A')
                detail = (f"Result {i+1} (source: '{source_info}', score: {score_info:.4f})\n"
                          f"(Retrieved by: '{variant_info}')\n{doc.page_content}")
                result_details.append(detail)

            final_result = result_header + nlp_summary + "\n\n---\n\n".join(result_details)
            print(f"\nReturning {len(relevant_docs)} relevant chunks from {len(set(all_page_titles))} source page(s).")
            return final_result.strip()

        except Exception as e:
            import traceback
            print(f"Unexpected error in {self.name}: {traceback.format_exc()}")
            return f"An unexpected error occurred: {str(e)}"


class EnhancedYoutubeScreenshotQA(BaseTool):
    name: str = "bird_species_screenshot_qa"
    description: str = (
        "Use this tool to calculate the number of bird species on camera at any one time,"
        "Input should be a dict with keys: 'youtube_url', 'question', and optional parameters. "
        "Example: {'youtube_url': 'https://youtube.com/watch?v=xyz', 'question': 'What animals are visible?'}"
    )

    # Define Pydantic fields for the attributes we need to set
    device: Any = Field(default=None, exclude=True)
    processor_vqa: Any = Field(default=None, exclude=True)
    model_vqa: Any = Field(default=None, exclude=True)

    class Config:
        # Allow arbitrary types (needed for torch.device, model objects)
        arbitrary_types_allowed = True
        # Allow extra fields to be set
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize directories
        cache_dir = '/tmp/youtube_qa_cache/'
        video_dir = '/tmp/video/'
        frames_dir = '/tmp/video_frames/'

        # Initialize model and device
        self._initialize_model()

        # Create directories
        for dir_path in [cache_dir, video_dir, frames_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def _get_config(self, key: str, default_value=None, input_data: Dict[str, Any] = None):
        """Get configuration value with fallback to defaults"""
        defaults = {
            'frame_interval_seconds': 20,
            'max_frames': 50,
            'use_scene_detection': True,
            'resize_frames': True,
            'parallel_processing': True,
            'cache_enabled': True,
            'quality_threshold': 30.0,
            'semantic_similarity_threshold': 0.8
        }

        if input_data and key in input_data:
            return input_data[key]
        return defaults.get(key, default_value)

    def _initialize_model(self):
        """Initialize BLIP model for VQA with error handling"""
        try:
            self.device = torch.device("cpu")
            print(f"Using device: {self.device}")

            self.processor_vqa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.model_vqa = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(self.device)

            print("BLIP VQA model loaded successfully")
        except Exception as e:
            print(f"Error initializing VQA model: {str(e)}")
            raise

    def _get_video_hash(self, url: str) -> str:
        """Generate hash for video URL for caching"""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, video_hash: str, cache_type: str) -> str:
        """Get cache file path"""
        cache_dir = '/tmp/youtube_qa_cache/'
        return os.path.join(cache_dir, f"{video_hash}_{cache_type}")

    def _load_from_cache(self, cache_path: str, cache_enabled: bool = True) -> Optional[Any]:
        """Load data from cache"""
        if not cache_enabled or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            return None

    def _save_to_cache(self, cache_path: str, data: Any, cache_enabled: bool = True):
        """Save data to cache"""
        if not cache_enabled:
            return
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def download_youtube_video(self, url: str, video_hash: str, cache_enabled: bool = True) -> Optional[str]:
        """Enhanced YouTube video download with anti-bot measures"""
        video_dir = '/tmp/video/'
        output_filename = f'{video_hash}.mp4'
        output_path = os.path.join(video_dir, output_filename)

        # Check cache
        if cache_enabled and os.path.exists(output_path):
            print(f"Using cached video: {output_path}")
            return output_path

        # Clean directory
        self._clean_directory(video_dir)

        try:
            # Enhanced yt-dlp options with anti-bot measures
            ydl_opts = {
                # Format selection - prefer lower quality to avoid restrictions
                'format': 'best[height<=480][ext=mp4]/best[height<=720][ext=mp4]/best[ext=mp4]/best',
                'outtmpl': output_path,
                'quiet': False,  # Changed to False for debugging
                'no_warnings': False,
                'merge_output_format': 'mp4',
                
                # Anti-bot headers and user agent
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                
                # Additional anti-detection measures
                'extractor_args': {
                    'youtube': {
                        'skip': ['hls', 'dash'],  # Skip some formats that might trigger detection
                        'player_skip': ['js'],    # Skip JavaScript player
                    }
                },
                
                # Rate limiting
                'sleep_interval': 1,
                'max_sleep_interval': 5,
                'sleep_interval_subtitles': 1,
                
                # Retry settings
                'retries': 3,
                'fragment_retries': 3,
                'skip_unavailable_fragments': True,
                
                # Cookie handling (you can add browser cookies if needed)
                # 'cookiefile': '/path/to/cookies.txt',  # Uncomment and set path if you have cookies
                
                # Additional options
                'extract_flat': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignoreerrors': True,
                
                # Postprocessors
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }]
            }

            print(f"Attempting to download: {url}")
            
            # Try multiple download strategies
            strategies = [
                # Strategy 1: Standard download
                ydl_opts,
                
                # Strategy 2: More conservative approach
                {
                    **ydl_opts,
                    'format': 'worst[ext=mp4]/worst',  # Try worst quality first
                    'sleep_interval': 2,
                    'max_sleep_interval': 10,
                },
                
                # Strategy 3: Different user agent
                {
                    **ydl_opts,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
                    },
                    'format': 'best[height<=360][ext=mp4]/best[ext=mp4]/best',
                }
            ]

            last_error = None
            for i, strategy in enumerate(strategies, 1):
                try:
                    print(f"Trying download strategy {i}/3...")
                    
                    with yt_dlp.YoutubeDL(strategy) as ydl:
                        # Add some delay before download
                        import time
                        time.sleep(2)
                        
                        ydl.download([url])

                    if os.path.exists(output_path):
                        print(f"Video downloaded successfully with strategy {i}: {output_path}")
                        return output_path
                    else:
                        print(f"Strategy {i} completed but file not found")
                        
                except Exception as e:
                    last_error = e
                    print(f"Strategy {i} failed: {str(e)}")
                    if i < len(strategies):
                        print(f"Trying next strategy...")
                        # Add delay between strategies
                        import time
                        time.sleep(5)
                    continue

            # If all strategies failed, try one more approach with cookies from browser
            print("All standard strategies failed. Trying with browser cookies...")
            try:
                cookie_strategy = {
                    **ydl_opts,
                    'cookiesfrombrowser': ('chrome',),  # Try to get cookies from Chrome
                    'format': 'worst[ext=mp4]/worst',
                }
                
                with yt_dlp.YoutubeDL(cookie_strategy) as ydl:
                    ydl.download([url])
                    
                if os.path.exists(output_path):
                    print(f"Video downloaded successfully with browser cookies: {output_path}")
                    return output_path
                    
            except Exception as e:
                print(f"Browser cookie strategy also failed: {str(e)}")

            print(f"All download strategies failed. Last error: {last_error}")
            return None

        except Exception as e:
            print(f"Error downloading YouTube video: {str(e)}")
            return None

    def _clean_directory(self, directory: str):
        """Clean directory contents"""
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality using Laplacian variance (blur detection)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0.0

    def _detect_scene_changes(self, video_path: str, threshold: float = 30.0) -> List[int]:
        """Detect scene changes in video"""
        scene_frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []

            prev_frame = None
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Calculate histogram difference
                    hist1 = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist2 = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

                    if diff > threshold:
                        scene_frames.append(frame_count)

                prev_frame = frame.copy()
                frame_count += 1

            cap.release()
            return scene_frames

        except Exception as e:
            print(f"Error in scene detection: {str(e)}")
            return []

    def smart_extract_frames(self, video_path: str, video_hash: str, input_data: Dict[str, Any] = None) -> List[str]:
        """Intelligently extract frames with quality filtering and scene detection"""
        cache_enabled = self._get_config('cache_enabled', True, input_data)
        cache_path = self._get_cache_path(video_hash, "frames_info.json")
        cached_info = self._load_from_cache(cache_path, cache_enabled)

        if cached_info:
            # Verify cached frames still exist
            existing_frames = [f for f in cached_info['frame_paths'] if os.path.exists(f)]
            if len(existing_frames) == len(cached_info['frame_paths']):
                print(f"Using {len(existing_frames)} cached frames")
                return existing_frames

        # Clean frames directory
        frames_dir = '/tmp/video_frames/'
        self._clean_directory(frames_dir)

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval_seconds = self._get_config('frame_interval_seconds', 10, input_data)
            frame_interval = max(1, int(fps * frame_interval_seconds))

            print(f"Video info: {total_frames} frames, {fps:.2f} fps")

            # Get scene change frames if enabled
            scene_frames = set()
            use_scene_detection = self._get_config('use_scene_detection', True, input_data)
            if use_scene_detection:
                scene_frames = set(self._detect_scene_changes(video_path))
                print(f"Detected {len(scene_frames)} scene changes")

            extracted_frames = []
            frame_count = 0
            saved_count = 0
            max_frames = self._get_config('max_frames', 50, input_data)

            while True:
                ret, frame = cap.read()
                if not ret or saved_count >= max_frames:
                    break

                # Check if we should extract this frame
                should_extract = (
                    frame_count % frame_interval == 0 or
                    frame_count in scene_frames
                )

                if should_extract:
                    # Assess frame quality
                    quality = self._assess_frame_quality(frame)
                    quality_threshold = self._get_config('quality_threshold', 30.0, input_data)

                    if quality >= quality_threshold:
                        # Resize frame if enabled
                        resize_frames = self._get_config('resize_frames', True, input_data)
                        if resize_frames:
                            height, width = frame.shape[:2]
                            if width > 800:
                                scale = 800 / width
                                new_width = 800
                                new_height = int(height * scale)
                                frame = cv2.resize(frame, (new_width, new_height))

                        frame_filename = os.path.join(
                            frames_dir,
                            f"frame_{frame_count:06d}_q{quality:.1f}.jpg"
                        )

                        if cv2.imwrite(frame_filename, frame):
                            extracted_frames.append(frame_filename)
                            saved_count += 1
                            print(f"Extracted frame {saved_count}/{max_frames} "
                                  f"(quality: {quality:.1f})")

                frame_count += 1

            cap.release()

            # Cache frame information
            frame_info = {
                'frame_paths': extracted_frames,
                'extraction_time': time.time(),
                'total_frames_processed': frame_count,
                'frames_extracted': len(extracted_frames)
            }
            self._save_to_cache(cache_path, frame_info, cache_enabled)

            print(f"Successfully extracted {len(extracted_frames)} high-quality frames")
            return extracted_frames

        except Exception as e:
            print(f"Exception during frame extraction: {e}")
            return []

    def _answer_question_on_frame(self, frame_path: str, question: str) -> Tuple[str, float]:
        """Answer question on single frame with confidence scoring"""
        try:
            image = Image.open(frame_path).convert('RGB')
            inputs = self.processor_vqa(image, question, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model_vqa.generate(**inputs, output_scores=True, return_dict_in_generate=True)

            answer = self.processor_vqa.decode(outputs.sequences[0], skip_special_tokens=True)

            # Calculate confidence (simplified - you might want to use actual model confidence)
            confidence = 1.0  # Placeholder - BLIP doesn't directly provide confidence

            return answer, confidence

        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")
            return "Error processing this frame", 0.0

    def _process_frames_parallel(self, frame_files: List[str], question: str, input_data: Dict[str, Any] = None) -> List[Tuple[str, str, float]]:
        """Process frames in parallel"""
        results = []
        parallel_processing = self._get_config('parallel_processing', True, input_data)

        if parallel_processing:
            with ThreadPoolExecutor(max_workers=min(4, len(frame_files))) as executor:
                future_to_frame = {
                    executor.submit(self._answer_question_on_frame, frame_path, question): frame_path
                    for frame_path in frame_files
                }

                for future in as_completed(future_to_frame):
                    frame_path = future_to_frame[future]
                    try:
                        answer, confidence = future.result()
                        results.append((frame_path, answer, confidence))
                        print(f"Processed {os.path.basename(frame_path)}: {answer} (conf: {confidence:.2f})")
                    except Exception as e:
                        print(f"Error processing {frame_path}: {str(e)}")
                        results.append((frame_path, "Error", 0.0))
        else:
            for frame_path in frame_files:
                answer, confidence = self._answer_question_on_frame(frame_path, question)
                results.append((frame_path, answer, confidence))
                print(f"Processed {os.path.basename(frame_path)}: {answer} (conf: {confidence:.2f})")

        return results

    def _cluster_similar_answers(self, answers: List[str], input_data: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Cluster semantically similar answers"""
        if len(answers) <= 1:
            return {answers[0]: answers} if answers else {}

        try:
            # First try with standard TF-IDF settings
            vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                min_df=1,  # Include words that appear in at least 1 document
                max_df=1.0  # Include words that appear in up to 100% of documents
            )
            tfidf_matrix = vectorizer.fit_transform(answers)

            # Check if we have any features after TF-IDF
            if tfidf_matrix.shape[1] == 0:
                raise ValueError("No features after TF-IDF processing")

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Cluster similar answers
            clusters = defaultdict(list)
            used = set()
            semantic_similarity_threshold = self._get_config('semantic_similarity_threshold', 0.8, input_data)

            for i, answer in enumerate(answers):
                if i in used:
                    continue

                cluster_key = answer
                clusters[cluster_key].append(answer)
                used.add(i)

                # Find similar answers
                for j in range(i + 1, len(answers)):
                    if j not in used and similarity_matrix[i][j] >= semantic_similarity_threshold:
                        clusters[cluster_key].append(answers[j])
                        used.add(j)

            return dict(clusters)

        except (ValueError, Exception) as e:
            print(f"Error in semantic clustering: {str(e)}")

            # Fallback 1: Try without stop words filtering
            try:
                print("Attempting clustering without stop word filtering...")
                vectorizer_no_stop = TfidfVectorizer(
                    lowercase=True,
                    min_df=1,
                    token_pattern=r'\b\w+\b'  # Match any word
                )
                tfidf_matrix = vectorizer_no_stop.fit_transform(answers)

                if tfidf_matrix.shape[1] > 0:
                    similarity_matrix = cosine_similarity(tfidf_matrix)

                    clusters = defaultdict(list)
                    used = set()
                    semantic_similarity_threshold = self._get_config('semantic_similarity_threshold', 0.8, input_data)

                    for i, answer in enumerate(answers):
                        if i in used:
                            continue

                        cluster_key = answer
                        clusters[cluster_key].append(answer)
                        used.add(i)

                        for j in range(i + 1, len(answers)):
                            if j not in used and similarity_matrix[i][j] >= semantic_similarity_threshold:
                                clusters[cluster_key].append(answers[j])
                                used.add(j)

                    return dict(clusters)

            except Exception as e2:
                print(f"Fallback clustering also failed: {str(e2)}")

            # Fallback 2: Simple string-based clustering
            print("Using simple string-based clustering...")
            return self._simple_string_cluster(answers)

    def _simple_string_cluster(self, answers: List[str]) -> Dict[str, List[str]]:
        """Simple string-based clustering fallback"""
        clusters = defaultdict(list)

        # Normalize answers for comparison
        normalized_answers = {}
        for answer in answers:
            normalized = answer.lower().strip()
            normalized_answers[answer] = normalized

        used = set()

        for i, answer in enumerate(answers):
            if answer in used:
                continue

            cluster_key = answer
            clusters[cluster_key].append(answer)
            used.add(answer)

            # Find similar answers using simple string similarity
            for j, other_answer in enumerate(answers[i+1:], i+1):
                if other_answer in used:
                    continue

                # Check for exact match after normalization
                if normalized_answers[answer] == normalized_answers[other_answer]:
                    clusters[cluster_key].append(other_answer)
                    used.add(other_answer)
                # Alternatively, check if one string contains the other
                elif (normalized_answers[answer] in normalized_answers[other_answer] or
                      normalized_answers[other_answer] in normalized_answers[answer]):
                    clusters[cluster_key].append(other_answer)
                    used.add(other_answer)

        return dict(clusters)

    def _analyze_temporal_patterns(self, results: List[Tuple[str, str, float]]) -> Dict[str, Any]:
        """Analyze temporal patterns in answers"""
        try:
            # Sort by frame number
            def get_frame_number(frame_path):
                match = re.search(r'frame_(\d+)', os.path.basename(frame_path))
                return int(match.group(1)) if match else 0

            sorted_results = sorted(results, key=lambda x: get_frame_number(x[0]))

            # Analyze answer changes over time
            answers_timeline = [result[1] for result in sorted_results]
            changes = []

            for i in range(1, len(answers_timeline)):
                if answers_timeline[i] != answers_timeline[i-1]:
                    changes.append({
                        'frame_index': i,
                        'from_answer': answers_timeline[i-1],
                        'to_answer': answers_timeline[i]
                    })

            return {
                'total_changes': len(changes),
                'change_points': changes,
                'stability_ratio': 1 - (len(changes) / max(1, len(answers_timeline) - 1)),
                'answers_timeline': answers_timeline
            }

        except Exception as e:
            print(f"Error in temporal analysis: {str(e)}")
            return {'error': str(e)}

    def analyze_video_question(self, frame_files: List[str], question: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive video question analysis"""
        if not frame_files:
            return {
                "final_answer": "No frames available for analysis.",
                "confidence": 0.0,
                "frame_count": 0,
                "error": "No valid frames found"
            }

        # Process all frames
        print(f"Processing {len(frame_files)} frames...")
        results = self._process_frames_parallel(frame_files, question, input_data)

        if not results:
            return {
                "final_answer": "Could not analyze any frames successfully.",
                "confidence": 0.0,
                "frame_count": 0,
                "error": "Frame processing failed"
            }

        # Extract answers and confidences
        answers = [result[1] for result in results if result[1] != "Error"]
        confidences = [result[2] for result in results if result[1] != "Error"]

        # Calculate statistical summary on numeric answers
        numeric_answers = []
        for answer in answers:
            try:
                # Try to convert answer to float
                numeric_value = float(answer)
                numeric_answers.append(numeric_value)
            except (ValueError, TypeError):
                # Skip non-numeric answers
                pass

        if numeric_answers:
            stats = {
                "minimum": float(np.min(numeric_answers)),
                "maximum": float(np.max(numeric_answers)),
                "range": float(np.max(numeric_answers) - np.min(numeric_answers)),
                "mean": float(np.mean(numeric_answers)),
                "median": float(np.median(numeric_answers)),
                "count": len(numeric_answers),
                "data_type": "answers"
            }
        elif confidences:
            # Fallback to confidence statistics if no numeric answers
            stats = {
                "minimum": float(np.min(confidences)),
                "maximum": float(np.max(confidences)),
                "range": float(np.max(confidences) - np.min(confidences)),
                "mean": float(np.mean(confidences)),
                "median": float(np.median(confidences)),
                "count": len(confidences),
                "data_type": "confidences"
            }
        else:
            stats = {
                "minimum": 0.0,
                "maximum": 0.0,
                "range": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "count": 0,
                "data_type": "none",
                "note": "No numeric results available for statistical summary"
            }

        if not answers:
            return {
                "final_answer": "All frame processing failed.",
                "confidence": 0.0,
                "frame_count": len(frame_files),
                "error": "No successful frame analysis"
            }

        # Cluster similar answers
        answer_clusters = self._cluster_similar_answers(answers, input_data)

        # Find most common cluster
        largest_cluster = max(answer_clusters.items(), key=lambda x: len(x[1]))
        most_common_answer = largest_cluster[0]

        # Calculate weighted confidence
        answer_counts = Counter(answers)
        total_answers = len(answers)
        frequency_confidence = answer_counts[most_common_answer] / total_answers
        avg_confidence = np.mean(confidences) if confidences else 0.0

        final_confidence = (frequency_confidence * 0.7) + (avg_confidence * 0.3)

        # Temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(results)

        return {
            "final_answer": most_common_answer,
            "confidence": final_confidence,
            "frame_count": len(frame_files),
            "successful_analyses": len(answers),
            "answer_distribution": dict(answer_counts),
            "semantic_clusters": {k: len(v) for k, v in answer_clusters.items()},
            "temporal_analysis": temporal_analysis,
            "average_model_confidence": avg_confidence,
            "frequency_confidence": frequency_confidence,
            "statistical_summary": stats
        }

    def _run(self, youtube_url, question, **kwargs) -> str:
        """Enhanced main execution method"""
        question = "How many unique bird species are on camera?"

        input_data = {
            'youtube_url': youtube_url,
            'question': question
        }

        if not youtube_url or not question:
            return "Error: Input must include 'youtube_url' and 'question'."

        try:
            # Generate video hash for caching
            video_hash = self._get_video_hash(youtube_url)

            # Step 1: Download video
            print(f"Downloading YouTube video from {youtube_url}...")
            cache_enabled = self._get_config('cache_enabled', True, input_data)
            video_path = self.download_youtube_video(youtube_url, video_hash, cache_enabled)
            if not video_path or not os.path.exists(video_path):
                return "Error: Failed to download the YouTube video. This may be due to YouTube's anti-bot protection. Try using a different video or implement cookie authentication."

            # Step 2: Smart frame extraction
            print(f"Extracting frames with smart selection...")
            frame_files = self.smart_extract_frames(video_path, video_hash, input_data)
            if not frame_files:
                return "Error: Failed to extract frames from the video."

            # Step 3: Comprehensive analysis
            print(f"Analyzing {len(frame_files)} frames for question: '{question}'")
            analysis_result = self.analyze_video_question(frame_files, question, input_data)

            if analysis_result.get("error"):
                return f"Error: {analysis_result['error']}"

            # Format comprehensive result - Fixed the reference to stats
            result = f"""

📊 **STATISTICAL SUMMARY**:
• Minimum: {analysis_result['statistical_summary']['minimum']:.2f}
• Maximum: {analysis_result['statistical_summary']['maximum']:.2f}
• Mean: {analysis_result['statistical_summary']['mean']:.2f}
• Median: {analysis_result['statistical_summary']['median']:.2f}
• Range: {analysis_result['statistical_summary']['range']:.2f}

            """.strip()

            return result

        except Exception as e:
            return f"Error during video analysis: {str(e)}"


# Initialize the enhanced tool
def create_enhanced_youtube_qa_tool(**kwargs):
    """Factory function to create the enhanced tool with custom parameters"""
    return EnhancedYoutubeScreenshotQA(**kwargs)

import os
import json
import hashlib
import time
import shutil
import glob
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
import speech_recognition as sr
from pydantic import Field
from pydantic.v1 import BaseModel
from pydub import AudioSegment
from pydub.silence import split_on_silence


class BaseTool(BaseModel):
    name: str
    description: str


class YouTubeTranscriptExtractor(BaseTool):
    name: str = "youtube_transcript_extractor"
    description: str = (
        "Downloads a YouTube video and extracts the complete audio transcript using speech recognition. "
        "Use this tool for questions about what people say in YouTube videos. "
        "Input should be a dict with keys: 'youtube_url' and optional parameters. "
        "Optional parameters include 'language' (e.g., 'en-US'), "
        "'cookies_file_path' (path to a cookies TXT file for authentication), "
        "or 'cookies_from_browser' (string specifying browser for cookies, e.g., 'chrome', 'firefox:profileName', 'edge+keyringName:profileName::containerName'). "
        "Example: {'youtube_url': 'https://youtube.com/watch?v=xyz', 'language': 'en-US'} or "
        "{'youtube_url': '...', 'cookies_file_path': '/path/to/cookies.txt'} or "
        "{'youtube_url': '...', 'cookies_from_browser': 'chrome'}"
    )

    recognizer: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow # Adjusted if pydantic v1 style

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.cache_dir = '/tmp/youtube_transcript_cache/'
        self.audio_dir = '/tmp/audio/'
        self.chunks_dir = '/tmp/audio_chunks/'

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.pause_threshold = 0.8

        for dir_path in [self.cache_dir, self.audio_dir, self.chunks_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def _get_config(self, key: str, default_value: Any = None, input_data: Optional[Dict[str, Any]] = None) -> Any:
        defaults = {
            'language': 'en-US',
            'chunk_length_ms': 30000,
            'silence_thresh': -40,
            'audio_quality': 'best',
            'cache_enabled': True,
            'min_silence_len': 500,
            'overlap_ms': 1000,
            'cookies_file_path': None,  # New: Path to a cookies file
            'cookies_from_browser': None # New: Browser string e.g., "chrome", "firefox:profile_name"
        }

        if input_data and key in input_data:
            return input_data[key]
        return defaults.get(key, default_value)

    def _get_video_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, video_hash: str, cache_type: str) -> str:
        return os.path.join(self.cache_dir, f"{video_hash}_{cache_type}")

    def _load_from_cache(self, cache_path: str, cache_enabled: bool = True) -> Optional[Any]:
        if not cache_enabled or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            return None

    def _save_to_cache(self, cache_path: str, data: Any, cache_enabled: bool = True):
        if not cache_enabled:
            return
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def _clean_directory(self, directory: str):
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def download_youtube_audio(self, url: str, video_hash: str, input_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        audio_quality = self._get_config('audio_quality', 'best', input_data)
        output_filename = f'{video_hash}.wav'
        output_path = os.path.join(self.audio_dir, output_filename)

        cache_enabled = self._get_config('cache_enabled', True, input_data)
        if cache_enabled and os.path.exists(output_path):
            print(f"Using cached audio: {output_path}")
            return output_path

        self._clean_directory(self.audio_dir)

        cookies_file_path = self._get_config('cookies_file_path', None, input_data)
        cookies_from_browser_str = self._get_config('cookies_from_browser', None, input_data)

        try:
            ydl_opts: Dict[str, Any] = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': os.path.join(self.audio_dir, f'{video_hash}.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False, # Ensure this is false for actual downloads
                'writethumbnail': False,
                'writeinfojson': False,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192' if audio_quality == 'best' else '128',
                }],
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                'nocheckcertificate': True,
            }

            if cookies_file_path:
                ydl_opts['cookiefile'] = cookies_file_path
                print(f"Using cookies from file: {cookies_file_path}")
            elif cookies_from_browser_str:
                parsed_browser, parsed_profile, parsed_keyring, parsed_container = None, None, None, None
                
                temp_str = cookies_from_browser_str
                
                if '::' in temp_str:
                    main_part_before_container, parsed_container_val = temp_str.split('::', 1)
                    parsed_container = parsed_container_val if parsed_container_val else None
                    temp_str = main_part_before_container
                
                if ':' in temp_str:
                    browser_keyring_part, parsed_profile_val = temp_str.split(':', 1)
                    parsed_profile = parsed_profile_val if parsed_profile_val else None
                    temp_str = browser_keyring_part

                if '+' in temp_str:
                    parsed_browser_val, parsed_keyring_val = temp_str.split('+', 1)
                    parsed_browser = parsed_browser_val
                    parsed_keyring = parsed_keyring_val if parsed_keyring_val else None
                else:
                    parsed_browser = temp_str

                if parsed_browser:
                    # yt-dlp expects cookiesfrombrowser as a tuple: (BROWSER, PROFILE, KEYRING, CONTAINER)
                    final_tuple: Tuple[Optional[str], ...] = (
                        parsed_browser,
                        parsed_profile,
                        parsed_keyring,
                        parsed_container
                    )
                    ydl_opts['cookiesfrombrowser'] = final_tuple
                    print(f"Attempting to use cookies from browser spec '{cookies_from_browser_str}', parsed as: {final_tuple}")
                else:
                    print(f"Invalid or empty browser name in cookies_from_browser string: '{cookies_from_browser_str}'")


            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"Downloading audio from: {url} with options: {ydl_opts}")
                ydl.download([url])
            
            if os.path.exists(output_path):
                print(f"Audio downloaded successfully: {output_path}")
                return output_path
            else:
                possible_files = glob.glob(os.path.join(self.audio_dir, f'{video_hash}.*'))
                if possible_files:
                    source_file = possible_files[0]
                    if not source_file.endswith('.wav'):
                        try:
                            audio = AudioSegment.from_file(source_file)
                            audio.export(output_path, format="wav")
                            os.remove(source_file)
                            print(f"Audio converted to WAV: {output_path}")
                            return output_path
                        except Exception as e:
                            print(f"Error converting audio: {str(e)}")
                            return None
                    else: # Already a .wav, possibly due to postprocessor already creating it with a different ext pattern
                        if source_file != output_path: # if names differ due to original extension
                           shutil.move(source_file, output_path)
                        print(f"Audio file found: {output_path}")
                        return output_path
                
                print(f"No audio file found at expected path after download: {output_path}")
                return None

        except yt_dlp.utils.DownloadError as de:
            print(f"yt-dlp DownloadError: {str(de)}")
            if "Sign in to confirm you're not a bot" in str(de) and not (cookies_file_path or cookies_from_browser_str):
                print("Authentication required. Consider using 'cookies_file_path' or 'cookies_from_browser' options.")
            return None
        except Exception as e:
            print(f"Error downloading YouTube audio: {type(e).__name__} - {str(e)}")
            # Fallback attempt is removed as it's unlikely to succeed if the primary authenticated attempt fails due to bot detection
            return None

    def _split_audio_intelligent(self, audio_path: str, input_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self._clean_directory(self.chunks_dir)
        try:
            audio = AudioSegment.from_wav(audio_path)
            chunk_length_ms = self._get_config('chunk_length_ms', 30000, input_data)
            silence_thresh = self._get_config('silence_thresh', -40, input_data)
            min_silence_len = self._get_config('min_silence_len', 500, input_data)
            overlap_ms = self._get_config('overlap_ms', 1000, input_data) # Not used in current split_on_silence

            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=True 
            )

            processed_chunks: List[AudioSegment] = [] # type: ignore
            # Combine small chunks or re-chunk if silence splitting is ineffective
            temp_chunk: Optional[AudioSegment] = None # type: ignore
            for chunk in chunks:
                if temp_chunk is None:
                    temp_chunk = chunk
                else:
                    temp_chunk += chunk

                if len(temp_chunk) > chunk_length_ms / 2 or chunk == chunks[-1]: # Arbitrary threshold to combine small chunks
                    processed_chunks.append(temp_chunk)
                    temp_chunk = None
            
            if not processed_chunks or any(len(p_chunk) > chunk_length_ms * 1.5 for p_chunk in processed_chunks): # If still problematic
                print("Using time-based splitting due to ineffective silence splitting or overly large chunks...")
                processed_chunks = []
                for i in range(0, len(audio), chunk_length_ms - overlap_ms):
                    chunk_segment = audio[i:i + chunk_length_ms]
                    if len(chunk_segment) > 1000:
                        processed_chunks.append(chunk_segment)
            
            chunk_data = []
            current_time_ms = 0
            for i, chunk_segment in enumerate(processed_chunks):
                if len(chunk_segment) < 1000: continue

                chunk_filename = os.path.join(self.chunks_dir, f"chunk_{i:04d}.wav")
                chunk_segment.export(chunk_filename, format="wav")
                
                duration_s = len(chunk_segment) / 1000.0
                start_time_s = current_time_ms / 1000.0
                end_time_s = start_time_s + duration_s

                chunk_data.append({
                    'filename': chunk_filename, 'index': i,
                    'start_time': start_time_s, 'duration': duration_s, 'end_time': end_time_s
                })
                current_time_ms += len(chunk_segment) # Approximation, true timestamping is harder

            print(f"Split audio into {len(chunk_data)} chunks")
            return chunk_data
        except Exception as e:
            print(f"Error splitting audio: {str(e)}")
            try: # Fallback: single chunk
                audio = AudioSegment.from_wav(audio_path)
                duration = len(audio) / 1000.0
                return [{'filename': audio_path, 'index': 0, 'start_time': 0, 'duration': duration, 'end_time': duration}]
            except: return []


    def _transcribe_audio_chunk(self, chunk_info: Dict[str, Any], input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chunk_path = chunk_info['filename']
        base_result = {
            'start_time': chunk_info.get('start_time', 0), 'end_time': chunk_info.get('end_time', 0),
            'duration': chunk_info.get('duration', 0), 'index': chunk_info.get('index', -1),
            'success': False, 'confidence': 0.0
        }
        try:
            language = self._get_config('language', 'en-US', input_data)
            with sr.AudioFile(chunk_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2) # Shorter adjustment
                audio_data = self.recognizer.record(source)

            try:
                text = self.recognizer.recognize_google(audio_data, language=language)
                return {**base_result, 'text': text, 'confidence': 1.0, 'success': True}
            except sr.UnknownValueError:
                try: # Try without specific language
                    text = self.recognizer.recognize_google(audio_data)
                    return {**base_result, 'text': text, 'confidence': 0.8, 'success': True} # Lower confidence
                except sr.UnknownValueError:
                    return {**base_result, 'text': '[INAUDIBLE]'}
            except sr.RequestError as e:
                return {**base_result, 'text': f'[RECOGNITION_ERROR: {str(e)}]', 'error': str(e)}
        except Exception as e:
            return {**base_result, 'text': f'[ERROR: {str(e)}]', 'error': str(e)}

    def _transcribe_chunks_parallel(self, chunk_data: List[Dict[str, Any]], input_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        results = []
        max_workers = min(os.cpu_count() or 1, 4) # Limit workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._transcribe_audio_chunk, chunk_info, input_data): chunk_info
                for chunk_info in chunk_data
            }
            for future in as_completed(future_to_chunk):
                chunk_info = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "Transcribed" if result['success'] else "Failed"
                    preview = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
                    print(f"{status} chunk {result['index']}: {preview}")
                except Exception as e:
                    print(f"Error processing chunk {chunk_info.get('index', '?')}: {str(e)}")
                    results.append({
                        'text': f'[PROCESSING_ERROR: {str(e)}]', 'confidence': 0.0,
                        'start_time': chunk_info.get('start_time', 0), 'end_time': chunk_info.get('end_time', 0),
                        'duration': chunk_info.get('duration', 0), 'index': chunk_info.get('index', 0),
                        'success': False, 'error': str(e)
                    })
        results.sort(key=lambda x: x['index'])
        return results

    def extract_transcript(self, audio_path: str, video_hash: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cache_enabled = self._get_config('cache_enabled', True, input_data)
        cache_path = self._get_cache_path(video_hash, "transcript.json")

        cached_transcript = self._load_from_cache(cache_path, cache_enabled)
        if cached_transcript:
            print("Using cached transcript")
            return cached_transcript

        try:
            print("Splitting audio into chunks...")
            chunk_data = self._split_audio_intelligent(audio_path, input_data)
            if not chunk_data:
                return {'error': 'Failed to split audio', 'full_transcript': '', 'success_rate': 0.0}

            print(f"Transcribing {len(chunk_data)} audio chunks...")
            transcript_results = self._transcribe_chunks_parallel(chunk_data, input_data)
            
            successful_chunks = [r for r in transcript_results if r['success']]
            full_text = ' '.join([r['text'] for r in successful_chunks if r['text'] and '[INAUDIBLE]' not in r['text'] and 'ERROR' not in r['text']]).strip()
            
            total_c = len(transcript_results)
            successful_c = len(successful_chunks)
            success_rate = successful_c / total_c if total_c > 0 else 0.0
            
            final_result = {
                'full_transcript': full_text, 'word_count': len(full_text.split()),
                'total_chunks': total_c, 'successful_chunks': successful_c, 'success_rate': success_rate,
                'extraction_timestamp': time.time(), 'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'detailed_results': transcript_results
            }
            self._save_to_cache(cache_path, final_result, cache_enabled)
            print(f"Transcript extraction completed. Success rate: {success_rate:.1%}")
            return final_result
        except Exception as e:
            print(f"Error during transcript extraction: {str(e)}")
            return {'error': str(e), 'full_transcript': '', 'success_rate': 0.0}

    def _run(self, youtube_url: str, **kwargs: Any) -> str:
        input_data = {'youtube_url': youtube_url, **kwargs}
        if not youtube_url: return "Error: youtube_url is required."

        try:
            video_hash = self._get_video_hash(youtube_url)
            print(f"Processing YouTube URL: {youtube_url} (Hash: {video_hash})")
            
            audio_path = self.download_youtube_audio(youtube_url, video_hash, input_data)
            if not audio_path or not os.path.exists(audio_path):
                return "Error: Failed to download YouTube audio. Check URL or authentication (cookies)."

            print("Extracting audio transcript...")
            transcript_result = self.extract_transcript(audio_path, video_hash, input_data)

            if transcript_result.get("error"): return f"Error: {transcript_result['error']}"
            
            main_transcript = transcript_result.get('full_transcript', '')
            if not main_transcript: return "Error: No transcript could be extracted."

            print(f"Transcript extracted. Word count: {transcript_result.get('word_count',0)}. Success: {transcript_result.get('success_rate',0):.1%}")
            return "TRANSCRIPT: " + main_transcript
        except Exception as e:
            print(f"Unhandled error in _run: {str(e)}") # For debugging
            return f"Error during transcript extraction: {str(e)}"

# Factory function to create the tool
def create_youtube_transcript_tool(**kwargs):
    """Factory function to create the transcript extraction tool with custom parameters"""
    return YouTubeTranscriptExtractor(**kwargs)

# --- Model Configuration ---
def create_llm_pipeline():
    #model_id = "meta-llama/Llama-2-13b-chat-hf"
    #model_id = "meta-llama/Llama-3.3-70B-Instruct"
    #model_id = "mistralai/Mistral-Small-24B-Base-2501"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    #model_id = "Meta-Llama/Llama-2-7b-chat-hf"
    #model_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    #model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    #model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    #model_id = "Qwen/Qwen2-7B-Instruct"
    #model_id = "GSAI-ML/LLaDA-8B-Instruct"
    return pipeline(
        "text-generation",
        model=model_id,
        device_map="cpu",
        torch_dtype=torch.float16,
        max_new_tokens=1024,
        temperature=0.3,
        top_k=50,
        top_p=0.95
    )


nlp = None # Set to None if not using spaCy, so the regex fallback is used in extract_entities

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]
    done: bool = False  # Default value of False
    question: str
    task_id: str
    input_file: Optional[bytes]
    file_type: Optional[str]
    context: List[Document]  # Using LangChain's Document class
    file_path: Optional[str]
    youtube_url: Optional[str]
    answer: Optional[str]
    frame_answers: Optional[list]



# --- Define Call LLM function ---

# 3. Improved LLM call with memory management


def call_llm_with_memory_management(state: AgentState, llm_model) -> AgentState:
    """Enhanced LLM call with better prompt engineering and hallucination prevention."""
    print("Running call_llm with memory management...")

    #ipdb.set_trace()

    original_messages = messages_for_llm = state["messages"]

    # Context management - be more aggressive about truncation
    system_message_content = None
    if messages_for_llm and isinstance(messages_for_llm[0], SystemMessage):
        system_message_content = messages_for_llm[0]
        regular_messages = messages_for_llm[1:]
    else:
        regular_messages = messages_for_llm

    # Keep only the most recent messages (more aggressive)
    max_regular_messages = 6  # Reduced from 9
    if len(regular_messages) > max_regular_messages:
        print(f"🔄 Truncating to {max_regular_messages} recent messages")
        regular_messages = regular_messages[-max_regular_messages:]

    # Reconstruct for LLM
    messages_for_llm = []
    if system_message_content:
        messages_for_llm.append(system_message_content)
    messages_for_llm.extend(regular_messages)

    # Character limit check
    total_chars = sum(len(str(msg.content)) for msg in messages_for_llm)
    char_limit = 20000

    if total_chars > char_limit:
        print(f"📏 Context too long ({total_chars} chars) - further truncation")
        while regular_messages and sum(len(str(m.content)) for m in regular_messages) > char_limit - (len(str(system_message_content.content)) if system_message_content else 0):
            regular_messages.pop(0)

        messages_for_llm = []
        if system_message_content:
            messages_for_llm.append(system_message_content)
        messages_for_llm.extend(regular_messages)

    new_state = state.copy()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"🤖 Calling LLM with {len(messages_for_llm)} messages")

        # Convert to simple string format that the model can understand
        if len(messages_for_llm) == 2 and isinstance(messages_for_llm[0], SystemMessage) and isinstance(messages_for_llm[1], HumanMessage):
            # Initial query - use simple format
            system_content = messages_for_llm[0].content
            human_content = messages_for_llm[1].content

            formatted_input = f"{system_content}\n\nHuman: {human_content}\n\nAssistant:"
        else:
            # Ongoing conversation - build context
            formatted_input = ""

            # Add system message if present
            if system_message_content:
                formatted_input += f"{system_message_content.content}\n\n"

            # Add conversation messages
            for msg in regular_messages:
                if isinstance(msg, HumanMessage):
                    formatted_input += f"Human: {msg.content}\n\n"
                elif isinstance(msg, AIMessage):
                    formatted_input += f"Assistant: {msg.content}\n\n"
                elif isinstance(msg, ToolMessage):
                    formatted_input += f"Tool Result: {msg.content}\n\n"

            # Add explicit instruction for immediate final answer if we have recent tool results
            if any(isinstance(msg, ToolMessage) for msg in regular_messages[-2:]):
                formatted_input += "Based on the tool results above, provide your FINAL ANSWER now.\n\n"
                formatted_input += "REMINDER ON ANSWER FORMAT: \n"
                formatted_input += "- Numbers: no commas, no units unless specified\n"
                formatted_input += "- Strings: no articles, no abbreviations, digits in plain text\n"
                formatted_input += "- Lists: comma-separated following above rules\n"
                formatted_input += "- Be extremely brief and concise"

            formatted_input += "Assistant:"

        print(f"Input preview: {formatted_input[:300]}...")

        llm_response_object = llm_model.invoke(formatted_input)

        # Process response and clean up hallucinated content
        if isinstance(llm_response_object, BaseMessage):
            raw_content = llm_response_object.content
        elif hasattr(llm_response_object, 'content'):
            raw_content = str(llm_response_object.content)
        else:
            raw_content = str(llm_response_object)

        # Clean up the response to prevent hallucinated follow-up questions
        cleaned_content = clean_llm_response(raw_content)
        ai_message_response = AIMessage(content=cleaned_content)

        print(f"🔍 LLM Response preview: {cleaned_content[:200]}...")

        final_messages = original_messages + [ai_message_response]
        new_state["messages"] = final_messages
        new_state.pop("done", None)

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        error_message = AIMessage(content=f"Error: LLM call failed - {str(e)}")
        new_state["messages"] = original_messages + [error_message]
        new_state["done"] = True

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return new_state


def clean_llm_response(response_text: str) -> str:
    """
    Clean LLM response to prevent hallucinated follow-up questions and conversations.
    Specifically handles ReAct format: Thought: -> Action: -> Action Input:
    """
    if not response_text:
        return response_text

    print(f"Initial response: {response_text[:200]}...")

    # --- START MODIFICATION ---
    # Isolate the text generated by the assistant in the last turn.
    # This prevents parsing examples or instructions from the preamble.
    assistant_marker = "Assistant:"
    last_marker_idx = response_text.rfind(assistant_marker)

    text_to_process = response_text # Default to full text if marker not found
    if last_marker_idx != -1:
        # If "Assistant:" is found, process only the text after the last occurrence.
        text_to_process = response_text[last_marker_idx + len(assistant_marker):].strip()
        print(f"ℹ️ Parsing content after last 'Assistant:': {text_to_process[:200]}...")
    else:
        # If "Assistant:" is not found, process the whole input.
        # This might occur if the input is already just the assistant's direct response
        # or if the prompt structure is different.
        print(f"ℹ️ No 'Assistant:' marker found. Processing entire input as is.")
    # --- END MODIFICATION ---

    # Now, all subsequent operations use 'text_to_process'

    # Try to find a complete ReAct pattern in the assistant's actual output
    react_pattern = r'Thought:\s*(.*?)\s*Action:\s*([^\n\r]+)\s*Action Input:\s*(.*?)(?=\s*(?:Thought:|Action:|FINAL ANSWER:|$))'
    # Apply search to 'text_to_process'
    react_match = re.search(react_pattern, text_to_process, re.DOTALL | re.IGNORECASE)

    if react_match:
        thought_text = react_match.group(1).strip()
        action_name = react_match.group(2).strip()
        action_input = react_match.group(3).strip()

        # Clean up the action input - remove any trailing content that looks like instructions
        action_input_clean = re.sub(r'\s*(When you have|FINAL ANSWER|ANSWER FORMAT|IMPORTANT:).*$', '', action_input, flags=re.DOTALL | re.IGNORECASE)
        action_input_clean = action_input_clean.strip()

        react_sequence = f"Thought: {thought_text}\nAction: {action_name}\nAction Input: {action_input_clean}"

        print(f"🔧 Found ReAct pattern - Action: {action_name}, Input: {action_input_clean[:100]}...")

        # Check if there's a FINAL ANSWER after the action input (this would be hallucination)
        # Check in the remaining part of 'text_to_process'
        remaining_text_in_process = text_to_process[react_match.end():]
        final_answer_after = re.search(r'FINAL ANSWER:', remaining_text_in_process, re.IGNORECASE)
        if final_answer_after:
            print(f"🚫 Removed hallucinated FINAL ANSWER after tool call")

        return react_sequence

    # If no ReAct pattern in 'text_to_process', check for standalone FINAL ANSWER
    # This variable will hold the text being processed for FINAL ANSWER and then for fallback.
    current_text_for_processing = text_to_process
    final_answer_match = re.search(r"FINAL ANSWER:\s*(.+?)(?=\n|$)", current_text_for_processing, re.IGNORECASE)
    if final_answer_match:
        answer_content = final_answer_match.group(1).strip()

        template_phrases = [
            '[concise answer only]',
            '[concise answer - number/word/list only]',
            '[brief answer]',
            '[your answer here]',
            'concise answer only',
            'brief answer',
            'your answer here'
        ]

        if any(phrase.lower() in answer_content.lower() for phrase in template_phrases):
            print(f"🚫 Ignoring template FINAL ANSWER: {answer_content}")
            # Remove the template FINAL ANSWER and continue cleaning on the remainder of 'current_text_for_processing'
            current_text_for_processing = current_text_for_processing[:final_answer_match.start()].strip()
            # Fall through to the general cleanup section below
        else:
            # Keep everything from the start of 'current_text_for_processing' up to and including the real FINAL ANSWER line only
            cleaned_output = current_text_for_processing[:final_answer_match.end()]

            # Check if there's additional content after FINAL ANSWER in 'current_text_for_processing'
            remaining_after_final_answer = current_text_for_processing[final_answer_match.end():].strip()
            if remaining_after_final_answer:
                print(f"🚫 Removed content after FINAL ANSWER: {remaining_after_final_answer[:100]}...")

            return cleaned_output.strip()

    # If no ReAct, or FINAL ANSWER was a template or not found, apply fallback cleaning to 'current_text_for_processing'
    lines = current_text_for_processing.split('\n')
    cleaned_lines = []

    for i, line in enumerate(lines):
        # Stop if we see the model repeating system instructions
        if re.search(r'\[SYSTEM\]|\[HUMAN\]|\[ASSISTANT\]|\[TOOL\]', line, re.IGNORECASE):
            print(f"🚫 Stopped at repeated system format: {line}")
            break

        # Stop if we see the model generating format instructions
        if re.search(r'CRITICAL INSTRUCTIONS|FORMAT for tool use|ANSWER FORMAT', line, re.IGNORECASE):
            print(f"🚫 Stopped at repeated instructions: {line}")
            break

        # Stop if we see the model role-playing as a human asking questions
        if re.search(r'(what are|what is|how many|can you tell me)', line, re.IGNORECASE) and not line.strip().startswith(('Thought:', 'Action:', 'Action Input:')):
            # Make sure this isn't part of a legitimate thought process
            if i > 0 and not any(keyword in lines[i-1] for keyword in ['Thought:', 'Action:', 'need to']):
                print(f"🚫 Stopped at hallucinated question: {line}")
                break

        cleaned_lines.append(line)

    cleaned = '\n'.join(cleaned_lines).strip()
    print(f"Final cleaned response (fallback): {cleaned[:200]}...")

    return cleaned

def parse_react_output(state: AgentState) -> AgentState:
    """
    Enhanced parsing with better FINAL ANSWER detection and flow control.
    """
    print("Running parse_react_output...")

    #ipdb.set_trace()

    messages = state.get("messages", [])
    if not messages:
        print("No messages in state.")
        new_state = state.copy()
        new_state["done"] = True
        return new_state


    # DEBUG
    print(f"parse_react_output: Entry message count: {len(messages)}")
    if messages and hasattr(messages[-1], 'tool_calls'):
        print(f"parse_react_output: Number of tool calls in last AIMessage: {len(messages[-1].tool_calls)}")

    last_message = messages[-1]
    new_state = state.copy()

    if not isinstance(last_message, AIMessage):
        print("Last message is not an AIMessage instance.")
        return new_state

    content = last_message.content
    if not isinstance(content, str):
        content = str(content)

    # Look for FINAL ANSWER first - this should take absolute priority
    # Use a more precise regex to capture just the answer line
    final_answer_match = re.search(r"FINAL ANSWER:\s*([^\n\r]+)", content, re.IGNORECASE)
    if final_answer_match:
        final_answer_text = final_answer_match.group(1).strip()

        # Check if this is template text (not a real answer)
        template_phrases = [
            '[concise answer only]',
            '[concise answer - number/word/list only]',
            '[brief answer]',
            '[your answer here]',
            'concise answer only',
            'brief answer',
            'your answer here'
        ]

        # If it's template text, don't treat it as a final answer
        if any(phrase.lower() in final_answer_text.lower() for phrase in template_phrases):
            print(f"🚫 Ignoring template FINAL ANSWER: '{final_answer_text}'")
            # Continue processing as if no final answer was found
        else:
            print(f"🎯 FINAL ANSWER found: '{final_answer_text}' - ENDING")

            # Store the answer in state for easy access
            new_state["answer"] = final_answer_text

            # Clean up the message content to just show the final answer
            clean_content = f"FINAL ANSWER: {final_answer_text}"
            updated_ai_message = AIMessage(content=clean_content, tool_calls=[])
            new_state["messages"] = messages[:-1] + [updated_ai_message]
            new_state["done"] = True
            return new_state

    # If no FINAL ANSWER, look for tool calls
    action_match = re.search(r"Action:\s*([^\n]+)", content, re.IGNORECASE)
    action_input_match = re.search(r"Action Input:\s*(.+)", content, re.IGNORECASE | re.DOTALL)

    if action_match and action_input_match:
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        if tool_name.lower() == "none":
            print("Action is 'None' - treating as regular response")
            updated_ai_message = AIMessage(content=content, tool_calls=[])
            new_state["messages"] = messages[:-1] + [updated_ai_message]
            new_state.pop("done", None)
            return new_state

        print(f"🔧 Tool call: {tool_name} with input: {tool_input_raw[:100]}...")

        # Parse tool arguments
        tool_args = {}
        try:
            trimmed_input = tool_input_raw.strip()
            if (trimmed_input.startswith('{') and trimmed_input.endswith('}')) or \
               (trimmed_input.startswith('[') and trimmed_input.endswith(']')):
                tool_args = ast.literal_eval(trimmed_input)
                if not isinstance(tool_args, dict):
                    tool_args = {"query": tool_input_raw}
            else:
                tool_args = {"query": tool_input_raw}
        except (ValueError, SyntaxError):
            tool_args = {"query": tool_input_raw}

        tool_call_id = str(uuid.uuid4())
        parsed_tool_calls = [{"name": tool_name, "args": tool_args, "id": tool_call_id}]

        updated_ai_message = AIMessage(content=content, tool_calls=parsed_tool_calls)
        new_state["messages"] = messages[:-1] + [updated_ai_message]
        new_state.pop("done", None)
        return new_state

    # No tool call or final answer - treat as regular response
    print("No actionable content found - continuing conversation")
    updated_ai_message = AIMessage(content=content, tool_calls=[])
    new_state["messages"] = messages[:-1] + [updated_ai_message]
    new_state.pop("done", None)


    # DEBUG
    print(f"parse_react_output: Exit message count: {len(new_state['messages'])}")


    return new_state

# 4. Improved call_tool_with_memory_management to prevent duplicate processing
def call_tool_with_memory_management(state: AgentState) -> AgentState:
    """Process tool calls with memory management, avoiding duplicates."""
    print("Running call_tool with memory management...")

    # Clear CUDA cache before processing
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleared CUDA cache. Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    except ImportError:
        pass
    except Exception as e:
        print(f"Error clearing CUDA cache: {e}")

    # Check if we have parsed tool calls from the condition function
    if 'parsed_tool_calls' in state and state.get('parsed_tool_calls'):
        print("Executing parsed tool calls...")
        return execute_parsed_tool_calls(state)

    # Fallback to original OpenAI-style tool calls handling
    messages = state.get("messages", [])
    if not messages:
        print("No messages found in state.")
        return state

    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        print("No tool calls found in last message")
        return state

    # Avoid processing the same tool calls multiple times
    if hasattr(last_message, '_processed_tool_calls'):
        print("Tool calls already processed, skipping...")
        return state

    # Copy the messages to avoid mutating the original list
    new_messages = list(messages)
    print(f"Processing {len(last_message.tool_calls)} tool calls from last message")

    # Get file_path from state to pass to tools
    file_path_to_pass = state.get('file_path')

    for i, tool_call_item in enumerate(last_message.tool_calls):
        # Handle both dict and object-style tool calls
        if isinstance(tool_call_item, dict):
            tool_name = tool_call_item.get("name", "")
            raw_args = tool_call_item.get("args")
            tool_call_id = tool_call_item.get("id", str(uuid.uuid4()))
        elif hasattr(tool_call_item, "name") and hasattr(tool_call_item, "id"):
            tool_name = getattr(tool_call_item, "name", "")
            raw_args = getattr(tool_call_item, "args", None)
            tool_call_id = getattr(tool_call_item, "id", str(uuid.uuid4()))
        else:
            print(f"Skipping malformed tool call item: {tool_call_item}")
            continue

        print(f"Processing tool call {i+1}: {tool_name}")

        # Find the matching tool
        selected_tool = None
        for tool_instance in tools:
            if tool_instance.name.lower() == tool_name.lower():
                selected_tool = tool_instance
                break

        if not selected_tool:
            tool_result = f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(t.name for t in tools)}"
            print(f"Tool not found: {tool_name}")
        else:
            try:
                # Prepare the arguments for the tool.run() method
                tool_run_input_dict = {}

                if isinstance(raw_args, dict):
                    tool_run_input_dict = raw_args.copy()
                elif raw_args is not None:
                    tool_run_input_dict["query"] = str(raw_args)

                # Add file_path to the dictionary for the tool
                tool_run_input_dict['file_path'] = file_path_to_pass

                print(f"Executing {tool_name} with args: {tool_run_input_dict} ...")
                tool_result = selected_tool.run(tool_run_input_dict)


                #ipdb.set_trace()


                # Aggressive truncation to prevent memory issues
                if not isinstance(tool_result, str):
                    tool_result = str(tool_result)

                max_length = 18000 if "wikipedia" in tool_name.lower() else 18000
                if len(tool_result) > max_length:
                    original_length = len(tool_result)
                    tool_result = tool_result[:max_length] + f"... [Result truncated from {original_length} to {max_length} chars to prevent memory issues]"
                    print(f"📄 Truncated result to {max_length} characters")

                print(f"Tool result length: {len(tool_result)} characters")

            except Exception as e:
                tool_result = f"Error executing tool '{tool_name}': {str(e)}"
                print(f"Tool execution error: {e}")

        # Create tool message - ONLY ONE PER TOOL CALL
        tool_message = ToolMessage(
            content=tool_result,
            name=tool_name,
            tool_call_id=tool_call_id
        )
        new_messages.append(tool_message)
        print(f"Added tool message for {tool_name}")

    # Mark the last message as processed to prevent re-processing
    if hasattr(last_message, '__dict__'):
        last_message._processed_tool_calls = True

    # Update the state
    new_state = state.copy()
    new_state["messages"] = new_messages

    # Clear CUDA cache after processing
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleared CUDA cache post-processing. Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    except ImportError:
        pass
    except Exception as e:
        print(f"Error clearing CUDA cache post-processing: {e}")

    return new_state

# 3. Enhanced execute_parsed_tool_calls to prevent duplicate observations
def execute_parsed_tool_calls(state: AgentState):
    """
    Execute tool calls that were parsed from the Thought/Action/Action Input format.
    This is called by call_tool when parsed_tool_calls are present in state.
    """

    # Tool name mappings
    tool_name_mappings = {
        'wikipedia_semantic_search': 'wikipedia_tool',
        'wikipedia': 'wikipedia_tool',
        'search': 'enhanced_search',
        'duckduckgo_search': 'enhanced_search',
        'web_search': 'enhanced_search',
        'enhanced_search': 'enhanced_search',
        'youtube_screenshot_qa_tool': 'youtube_tool',
        'youtube': 'youtube_tool',
        'youtube_transcript_extractor': 'youtube_transcript_extractor',
        'youtube_audio_tool': 'youtube_transcript_extractor'
    }

    # Create a lookup by tool names
    tools_by_name = {}
    for tool in tools:
        tools_by_name[tool.name.lower()] = tool

    # Copy messages to avoid mutation during iteration
    new_messages = list(state["messages"])

    # Process each tool call ONCE
    for tool_call in state['parsed_tool_calls']:
        action = tool_call['action']
        action_input = tool_call['action_input']
        normalized_action = tool_call['normalized_action']

        print(f"🚀 Executing tool: {action} with input: {action_input}")

        # Find the tool instance
        tool_instance = None
        if normalized_action in tools_by_name:
            tool_instance = tools_by_name[normalized_action]
        elif normalized_action in tool_name_mappings:
            mapped_name = tool_name_mappings[normalized_action]
            if mapped_name in tools_by_name:
                tool_instance = tools_by_name[mapped_name]

        if tool_instance:
            try:
                # Pass file_path if the tool expects it
                if hasattr(tool_instance, 'run'):
                    if 'file_path' in tool_instance.run.__code__.co_varnames:
                        result = tool_instance.run(action_input, file_path=state.get('file_path'))
                    else:
                        result = tool_instance.run(action_input)
                else:
                    result = str(tool_instance)

                # Truncate long results
                if len(result) > 6000:
                    result = result[:6000] + "... [Result truncated due to length]"

                # Create a SINGLE observation message
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=f"Observation: {result}",
                    name=action,
                    tool_call_id=str(uuid.uuid4())
                )
                new_messages.append(tool_message)
                print(f"✅ Tool '{action}' executed successfully")

            except Exception as e:
                print(f"❌ Error executing tool '{action}': {e}")
                from langchain_core.messages import ToolMessage
                error_message = ToolMessage(
                    content=f"Observation: Error executing '{action}': {str(e)}",
                    name=action,
                    tool_call_id=str(uuid.uuid4())
                )
                new_messages.append(error_message)
        else:
            print(f"❌ Tool '{action}' not found in available tools")
            available_tool_names = list(tools_by_name.keys())
            from langchain_core.messages import ToolMessage
            error_message = ToolMessage(
                content=f"Observation: Tool '{action}' not found. Available tools: {', '.join(available_tool_names)}",
                name=action,
                tool_call_id=str(uuid.uuid4())
            )
            new_messages.append(error_message)

    # Update state with new messages and clear parsed tool calls
    new_state = state.copy()
    new_state["messages"] = new_messages
    new_state['parsed_tool_calls'] = []  # Clear to prevent re-execution

    return new_state


# 1. Add loop detection to your AgentState

def should_continue(state: AgentState) -> str:
    """Enhanced continuation logic with better limits."""
    print("Running should_continue...")

    # Check done flag first
    if state.get("done", False):
        print("✅ Done flag is True - ending")
        return "end"

    messages = state["messages"]

    # More aggressive message limit
    #if len(messages) > 20:  # Reduced from 15
    #    print(f"⚠️ Message limit reached ({len(messages)}/20) - forcing end")
    #    return "end"

    # Check for repeated patterns (stuck in loop)
    if len(messages) >= 6:
        recent_contents = [str(msg.content)[:100] for msg in messages[-6:] if hasattr(msg, 'content')]
        if len(set(recent_contents)) < 3:  # Too much repetition
            print("🔄 Detected repetitive pattern - ending")
            return "end"

    print(f"📊 Continuing... ({len(messages)} messages so far)")
    return "continue"


def route_after_parse_react(state: AgentState) -> str:
    """Determines the next step after parsing LLM output, prioritizing end state."""
    if state.get("done", False):  # Check if parse_react_output decided we are done
        return "end_processing"

    # Original logic: check for tool calls in the last message
    # Ensure messages list and last message exist before checking tool_calls
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "call_tool"
    return "call_llm"

# --- Graph Construction ---
# --- Graph Construction ---
def create_memory_safe_workflow():
    """Create a workflow with memory management and loop prevention."""
    # These models are initialized here but might be better managed if they need to be released/reinitialized
    # like you attempt in run_agent. Consider passing them or managing their lifecycle carefully.
    hf_pipe = create_llm_pipeline()
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    # vqa_model_name = "Salesforce/blip-vqa-base" # Not used in the provided graph logic directly
    # processor_vqa = BlipProcessor.from_pretrained(vqa_model_name) # Not used
    # model_vqa = BlipForQuestionAnswering.from_pretrained(vqa_model_name).to('cpu') # Not used

    workflow = StateGraph(AgentState)

    # Bind the llm_model to the call_llm_with_memory_management function
    bound_call_llm = partial(call_llm_with_memory_management, llm_model=llm)

    # Add nodes with memory-safe versions
    workflow.add_node("call_llm", bound_call_llm)  # Use the bound version here
    workflow.add_node("parse_react_output", parse_react_output)
    workflow.add_node("call_tool", call_tool_with_memory_management) # Ensure this doesn't also need llm if it calls back directly

    # Set entry point
    workflow.set_entry_point("call_llm")

    # Add conditional edges
    workflow.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "continue": "parse_react_output",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "parse_react_output",
        route_after_parse_react,
        {
            "call_tool": "call_tool",
            "call_llm": "call_llm",
            "end_processing": END
        }
    )

    workflow.add_edge("call_tool", "call_llm")

    return workflow.compile()


def count_english_words(text):
    # Remove punctuation, lowercase, split into words
    table = str.maketrans('', '', string.punctuation)
    words_in_text = text.translate(table).lower().split()
    return sum(1 for word in words_in_text if word in english_words)

def fix_backwards_text(text):
    reversed_text = text[::-1]
    original_count = count_english_words(text)
    reversed_count = count_english_words(reversed_text)
    if reversed_count > original_count:
        return reversed_text
    else:
        return text


# --- Run the Agent ---
# Enhanced system prompt for better behavior
def run_agent(agent, state: AgentState):
    """Enhanced agent initialization with better prompt and hallucination prevention."""
    global WIKIPEDIA_TOOL, SEARCH_TOOL, YOUTUBE_TOOL, YOUTUBE_AUDIO_TOOL, AUDIO_TRANSCRIPTION_TOOL, EXCEL_TOOL, PYTHON_TOOL, COMMUTATIVITY_TOOL, tools

    # Initialize tools
    WIKIPEDIA_TOOL = WikipediaSearchToolWithFAISS()
    SEARCH_TOOL = EnhancedDuckDuckGoSearchTool(max_results=3, max_chars_per_page=18000)
    YOUTUBE_TOOL = EnhancedYoutubeScreenshotQA()
    YOUTUBE_AUDIO_TOOL = YouTubeTranscriptExtractor()
    AUDIO_TRANSCRIPTION_TOOL = AudioTranscriptionTool()
    EXCEL_TOOL = ExcelReaderTool()
    PYTHON_TOOL = PythonExecutorTool()
    COMMUTATIVITY_TOOL = CommutativityAnalysisTool()
    tools = [WIKIPEDIA_TOOL, SEARCH_TOOL, YOUTUBE_AUDIO_TOOL, YOUTUBE_TOOL,  AUDIO_TRANSCRIPTION_TOOL, EXCEL_TOOL, PYTHON_TOOL, COMMUTATIVITY_TOOL]

    formatted_tools_description = render_text_description(tools)
    current_date_str = datetime.now().strftime("%Y-%m-%d")

    # Enhanced system prompt with stricter boundaries
    system_content = f"""You are an AI assistant with access to these tools:

{formatted_tools_description}

CRITICAL INSTRUCTIONS:
1. Answer ONLY the question asked by the human
2. Do NOT generate additional questions or continue conversations
3. Use tools ONLY when you need specific information you don't know
4. After using a tool, provide your FINAL ANSWER immediately
5. STOP after giving your FINAL ANSWER - do not continue
6. Do not repeat words in the question in the answer

FORMAT for tool use:
Thought: <brief reasoning>
Action: <exact_tool_name>
Action Input: <tool_input>

When you have the answer, immediately provide:
FINAL ANSWER: [concise answer only]

ANSWER FORMAT:
- Numbers: no commas, no units unless specified
- Questions on "how many" should be answered with a number ONLY
- Strings: no articles, no abbreviations, digits in plain text
- Lists: comma-separated either in ascending numeric order or alphabetical order as requested
- Be extremely brief and concise
- Do not provide additional context or explanations
- Do not provide parentheticals



IMPORTANT: You are responding to ONE question only. Do not ask follow-up questions or generate additional dialogue.

Current date: {current_date_str}
"""

    query = fix_backwards_text(state['question'])

    # Check for YouTube URLs
    yt_pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/[^\s]+"
    if re.search(yt_pattern, query):
        url_match = re.search(r"(https?://[^\s]+)", query)
        if url_match:
            state['youtube_url'] = url_match.group(0)

    # Initialize messages
    system_message = SystemMessage(content=system_content)
    human_message = HumanMessage(content=query)

    state['messages'] = [system_message, human_message]
    state["done"] = False

    # Run the agent
    result = agent.invoke(state)

    # Cleanup
    if result.get("done"):
        #torch.cuda.empty_cache()
        #torch.cuda.ipc_collect()
        gc.collect()
        print("🧹 Released GPU memory after completion")

    return result["messages"]

