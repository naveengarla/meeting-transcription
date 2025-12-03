"""
Robust Transcription Engine with Automatic Fallbacks
Handles errors gracefully and automatically falls back to smaller models on MemoryError.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .transcription import WhisperTranscriptionEngine, AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustTranscriptionEngine:
    """
    Robust wrapper around WhisperTranscriptionEngine with automatic fallback.
    
    Features:
    - Automatic fallback to smaller models on MemoryError
    - User-friendly error messages with specific solutions
    - Logging for diagnostics
    - Graceful degradation: large → medium → small → base → tiny
    """
    
    # Model fallback chain (in order of decreasing size)
    MODEL_FALLBACK_CHAIN = ['large', 'medium', 'small', 'base', 'tiny']
    
    # Model memory requirements (approximate)
    MODEL_MEMORY_REQUIREMENTS = {
        'large': 10.0,   # GB
        'medium': 5.0,
        'small': 2.0,
        'base': 1.0,
        'tiny': 0.5,
    }
    
    def __init__(self, 
                 model: str = 'base',
                 language: Optional[str] = None,
                 task: str = 'transcribe',
                 num_workers: int = 4,
                 enable_auto_fallback: bool = True):
        """
        Initialize robust transcription engine.
        
        Args:
            model: Initial model to try ('tiny', 'base', 'small', 'medium', 'large')
            language: Language code (e.g., 'en', 'es'). None for auto-detection
            task: 'transcribe' or 'translate'
            num_workers: Number of CPU threads for transcription
            enable_auto_fallback: Enable automatic fallback on errors
        """
        self.requested_model = model
        self.language = language
        self.task = task
        self.num_workers = num_workers
        self.enable_auto_fallback = enable_auto_fallback
        
        self.engine: Optional[WhisperTranscriptionEngine] = None
        self.current_model: Optional[str] = None
        self.fallback_count = 0
        
        # Initialize engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize transcription engine with error handling."""
        logger.info(f"Initializing transcription engine with model: {self.requested_model}")
        
        try:
            self.engine = WhisperTranscriptionEngine(
                model=self.requested_model,
                language=self.language,
                task=self.task,
                num_workers=self.num_workers
            )
            self.current_model = self.requested_model
            logger.info(f"✅ Successfully initialized with model: {self.current_model}")
            
        except MemoryError as e:
            logger.error(f"❌ MemoryError during initialization: {str(e)}")
            
            if self.enable_auto_fallback:
                logger.info("🔄 Attempting automatic fallback to smaller model...")
                self._fallback_to_smaller_model()
            else:
                raise MemoryError(
                    f"Insufficient memory to load model '{self.requested_model}'.\n"
                    f"Estimated requirement: {self.MODEL_MEMORY_REQUIREMENTS.get(self.requested_model, 'unknown')} GB RAM\n\n"
                    "Solutions:\n"
                    "1. Close other applications to free up memory\n"
                    "2. Use a smaller model (e.g., 'base' or 'tiny')\n"
                    "3. Enable auto-fallback in settings\n"
                    "4. Upgrade your system memory"
                )
        
        except FileNotFoundError as e:
            logger.error(f"❌ Model not found: {str(e)}")
            raise FileNotFoundError(
                f"Model '{self.requested_model}' not found.\n\n"
                "Solutions:\n"
                "1. Run: python download_models.py\n"
                "2. Or run: python src/model_downloader.py {self.requested_model} --engine faster-whisper\n"
                "3. Check your internet connection\n"
                "4. Verify model cache directory exists"
            )
        
        except Exception as e:
            logger.error(f"❌ Unexpected error during initialization: {str(e)}")
            raise RuntimeError(
                f"Failed to initialize transcription engine: {str(e)}\n\n"
                "Solutions:\n"
                "1. Run: python check_system.py\n"
                "2. Run: python check_dependencies.py\n"
                "3. Check logs/diagnostic_report_*.json for details\n"
                "4. Restart the application"
            )
    
    def _fallback_to_smaller_model(self):
        """Attempt to fall back to a smaller model."""
        # Find current position in fallback chain
        try:
            current_index = self.MODEL_FALLBACK_CHAIN.index(self.requested_model)
        except ValueError:
            # Model not in chain, default to base
            current_index = self.MODEL_FALLBACK_CHAIN.index('base') - 1
        
        # Try each smaller model
        for i in range(current_index + 1, len(self.MODEL_FALLBACK_CHAIN)):
            fallback_model = self.MODEL_FALLBACK_CHAIN[i]
            
            logger.info(f"🔄 Trying fallback model: {fallback_model}")
            logger.info(f"   Estimated memory requirement: {self.MODEL_MEMORY_REQUIREMENTS[fallback_model]} GB")
            
            try:
                self.engine = WhisperTranscriptionEngine(
                    model=fallback_model,
                    language=self.language,
                    task=self.task,
                    num_workers=self.num_workers
                )
                self.current_model = fallback_model
                self.fallback_count += 1
                
                logger.warning(
                    f"⚠️ Fell back to smaller model: {fallback_model}\n"
                    f"   Original model: {self.requested_model}\n"
                    f"   Quality may be reduced, but transcription will work."
                )
                
                return  # Success!
                
            except MemoryError:
                logger.error(f"❌ MemoryError with {fallback_model}, trying smaller model...")
                continue
            
            except Exception as e:
                logger.error(f"❌ Error with {fallback_model}: {str(e)}")
                continue
        
        # All fallbacks failed
        raise MemoryError(
            f"Unable to load any model. Tried: {self.requested_model} → {', '.join(self.MODEL_FALLBACK_CHAIN[current_index+1:])}\n\n"
            "Your system does not have enough available memory.\n\n"
            "Solutions:\n"
            "1. Close ALL other applications\n"
            "2. Restart your computer to clear memory\n"
            "3. Disable browser tabs and background apps\n"
            "4. Consider upgrading your RAM (minimum 4GB recommended)"
        )
    
    def transcribe(self, audio_file: str) -> List[AudioSegment]:
        """
        Transcribe audio file with robust error handling.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of AudioSegment objects with transcription results
            
        Raises:
            Various exceptions with user-friendly error messages
        """
        if not self.engine:
            raise RuntimeError("Transcription engine not initialized")
        
        audio_path = Path(audio_file)
        
        # Validate audio file exists
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_file}\n\n"
                "Solutions:\n"
                "1. Check the file path is correct\n"
                "2. Ensure the file was saved properly\n"
                "3. Check file permissions"
            )
        
        # Validate audio file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 1000:  # > 1GB
            logger.warning(f"⚠️ Large audio file: {file_size_mb:.1f} MB. Transcription may take a long time.")
        
        try:
            logger.info(f"🎵 Transcribing: {audio_file}")
            logger.info(f"   Model: {self.current_model}")
            logger.info(f"   Language: {self.language or 'auto-detect'}")
            logger.info(f"   Workers: {self.num_workers}")
            
            segments = self.engine.transcribe(audio_file)
            
            logger.info(f"✅ Transcription complete: {len(segments)} segments")
            
            # Warn if fallback was used
            if self.fallback_count > 0:
                logger.warning(
                    f"⚠️ Note: Used fallback model '{self.current_model}' instead of '{self.requested_model}'\n"
                    f"   For better quality, close other applications or use a smaller model."
                )
            
            return segments
        
        except MemoryError as e:
            logger.error(f"❌ MemoryError during transcription: {str(e)}")
            
            if self.enable_auto_fallback and self.current_model != 'tiny':
                logger.info("🔄 Attempting fallback to smaller model...")
                self._fallback_to_smaller_model()
                
                # Retry with smaller model
                logger.info("🔄 Retrying transcription with smaller model...")
                return self.transcribe(audio_file)
            else:
                raise MemoryError(
                    f"Insufficient memory to transcribe audio.\n"
                    f"Current model: {self.current_model}\n\n"
                    "Solutions:\n"
                    "1. Close other applications\n"
                    "2. Use a smaller model (Settings → Model Size → tiny/base)\n"
                    "3. Split audio into smaller chunks\n"
                    "4. Reduce num_workers in settings"
                )
        
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            # Specific error handling
            if 'cuda' in error_msg or 'gpu' in error_msg:
                raise RuntimeError(
                    f"GPU/CUDA error: {str(e)}\n\n"
                    "This application uses CPU only (no GPU required).\n"
                    "If you see CUDA errors, they can usually be ignored.\n\n"
                    "Solutions:\n"
                    "1. Ignore CUDA warnings (they don't affect functionality)\n"
                    "2. Reinstall faster-whisper: pip install --force-reinstall faster-whisper"
                )
            
            elif 'audio' in error_msg or 'format' in error_msg:
                raise RuntimeError(
                    f"Audio format error: {str(e)}\n\n"
                    "Solutions:\n"
                    "1. Ensure audio file is valid WAV/MP3/FLAC format\n"
                    "2. Try converting with: ffmpeg -i input.mp3 output.wav\n"
                    "3. Check audio is not corrupted"
                )
            
            else:
                raise RuntimeError(
                    f"Transcription error: {str(e)}\n\n"
                    "Solutions:\n"
                    "1. Run: python diagnostic_report.py\n"
                    "2. Check logs for details\n"
                    "3. Try a different model\n"
                    "4. Restart the application"
                )
        
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise RuntimeError(
                f"Unexpected error during transcription: {str(e)}\n\n"
                "Solutions:\n"
                "1. Run: python diagnostic_report.py\n"
                "2. Check logs/diagnostic_report_*.json\n"
                "3. Restart the application\n"
                "4. Report this error with the diagnostic report"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current engine status.
        
        Returns:
            Dictionary with engine status information
        """
        return {
            'requested_model': self.requested_model,
            'current_model': self.current_model,
            'fallback_count': self.fallback_count,
            'language': self.language,
            'task': self.task,
            'num_workers': self.num_workers,
            'auto_fallback_enabled': self.enable_auto_fallback,
            'initialized': self.engine is not None,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "initialized" if self.engine else "not initialized"
        fallback_info = f", {self.fallback_count} fallback(s)" if self.fallback_count > 0 else ""
        return f"RobustTranscriptionEngine(model={self.current_model}, {status}{fallback_info})"


def create_robust_engine(config: Optional[Dict[str, Any]] = None) -> RobustTranscriptionEngine:
    """
    Factory function to create robust transcription engine from config.
    
    Args:
        config: Configuration dictionary (if None, loads from environment)
        
    Returns:
        Initialized RobustTranscriptionEngine
    """
    if config is None:
        # Load from environment
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        config = {
            'model': os.getenv('WHISPER_MODEL', 'base'),
            'language': os.getenv('WHISPER_LANGUAGE'),
            'task': os.getenv('WHISPER_TASK', 'transcribe'),
            'num_workers': int(os.getenv('NUM_WORKERS', '4')),
            'enable_auto_fallback': True,
        }
    
    logger.info(f"Creating robust engine with config: {config}")
    
    return RobustTranscriptionEngine(**config)
