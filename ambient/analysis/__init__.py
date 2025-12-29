"""
Ambient Analysis Package - Gemini File Management System

A comprehensive Python package for managing file uploads to Google Gemini AI with intelligent caching,
batch processing, and gait analysis capabilities.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

# Import core components with defensive imports
try:
    from .gait_analyzer import GaitAnalyzer, GeminiAnalyzer
    from .gait_app import GaitAnalysisApplication
    from .upload_manager import AmbientGeminiFileManager

    # Legacy imports for backward compatibility
    def analyze_video(video_path: str, record_id: str = None):
        """Legacy function for backward compatibility."""
        app = GaitAnalysisApplication()
        return app.analyze_single_video(video_path, record_id)

    def analyze_all_videos():
        """Legacy function for backward compatibility."""
        app = GaitAnalysisApplication()
        app.analyze_all_videos()

    def upload_all_files():
        """Legacy function for backward compatibility."""
        manager = AmbientGeminiFileManager()
        # This would need to be implemented based on the original functionality
        pass

    def list_cached_files():
        """Legacy function for backward compatibility."""
        manager = AmbientGeminiFileManager()
        return manager.cache

    __all__ = [
        # Core components
        "GeminiAnalyzer",
        "GaitAnalyzer",
        "GaitAnalysisApplication",
        "AmbientGeminiFileManager",
        # Legacy functions
        "analyze_video",
        "analyze_all_videos",
        "upload_all_files",
        "list_cached_files",
    ]

except ImportError as e:
    # Handle import errors gracefully
    print(f"Warning: Could not import Ambient analysis components: {e}")
    __all__ = []
