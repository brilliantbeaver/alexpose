"""
Gait analysis module for the Ambient system.

This module provides the main analysis functionality for gait assessment using
Google Gemini AI and pose estimation data.

For calling the Gemini API, we use the `google.generativeai` library.
We use exponential backoff for retries of the API call if it fails.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import time
from typing import Any, List, Optional

import google.generativeai as genai
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from ambient.core.interfaces import IAnalyzer, IConfigurationManager, IOutputManager
from ambient.exceptions import AmbientError


def should_retry_exception(exception):
    """Custom retry condition that excludes ValueError."""
    return isinstance(exception, Exception) and not isinstance(exception, ValueError)


class GeminiError(Exception):
    """Exception raised for Gemini API errors."""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class AnalysisError(Exception):
    """Exception raised for analysis errors."""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class GeminiAnalyzer(IAnalyzer):
    """
    Handles Gemini API interactions for gait analysis.

    This class manages the connection to the Gemini API, handles file uploads,
    and processes analysis requests.
    """

    def __init__(
        self,
        api_key: str,
        file_manager: Any,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        config_manager: Optional[IConfigurationManager] = None,
    ):
        """
        Initialize the Gemini analyzer.

        Args:
            api_key: The Gemini API key
            file_manager: The file manager instance for handling uploads
            model_name: The Gemini model name to use
            temperature: The temperature setting for generation
            config_manager: Optional configuration manager for accessing prompt templates
        """
        self.api_key = api_key
        self.file_manager = file_manager
        self.model_name = model_name
        self.temperature = temperature
        self.config_manager = config_manager
        self.model = None

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )

    def wait_for_active(
        self, file_obj: Any, timeout: int = 300, poll_interval: int = 5
    ) -> Any:
        """
        Wait for a Gemini file to become ACTIVE.

        File States (Google Generative AI SDK):
        - STATE_UNSPECIFIED = 0: Initial state
        - PROCESSING = 1: File is being processed
        - ACTIVE = 2: File is ready for use
        - FAILED = 10: File processing failed

        Args:
            file_obj: Gemini file object
            timeout: Maximum time to wait in seconds (default: 300)
            poll_interval: Time between checks in seconds (default: 5)

        Returns:
            The file object once it becomes ACTIVE

        Raises:
            RuntimeError: If file processing fails
            TimeoutError: If file doesn't become ACTIVE within timeout
        """
        start = time.time()

        while True:
            current_state = getattr(file_obj, "state", None)

            # Check for ACTIVE state (enum value 2 or string 'ACTIVE')
            if (
                current_state == "ACTIVE"
                or current_state == 2
                or (hasattr(current_state, "name") and current_state.name == "ACTIVE")
            ):
                break

            # Check for FAILED state (enum value 10 or string 'FAILED')
            if (
                current_state == "FAILED"
                or current_state == 10
                or (hasattr(current_state, "name") and current_state.name == "FAILED")
            ):
                file_name = getattr(
                    file_obj, "display_name", getattr(file_obj, "name", str(file_obj))
                )
                raise RuntimeError(f"File {file_name} failed to process.")

            # Check timeout
            if time.time() - start > timeout:
                file_name = getattr(
                    file_obj, "display_name", getattr(file_obj, "name", str(file_obj))
                )
                raise TimeoutError(f"File {file_name} did not become ACTIVE in time.")

            # Refresh the file object
            try:
                file_obj = genai.get_file(file_obj.name)
            except Exception:
                # If we can't refresh, assume the file is ready
                break

            time.sleep(poll_interval)

        return file_obj

    def get_file_references(self, video_path: str, csv_paths: List[str]) -> tuple:
        """
        Get cached file references or upload if needed, and wait for ACTIVE state.

        Args:
            video_path: Path to the video file
            csv_paths: List of paths to CSV files

        Returns:
            Tuple of (video_ref, csv_refs) where video_ref is the video reference
            and csv_refs is a list of CSV references
        """
        # Get video reference using upload method (handles caching)
        video_ref = self.file_manager.upload_video(video_path)
        if not video_ref:
            raise GeminiError(
                message=f"Failed to get video reference for: {video_path}",
                details="Please check the video file and try again.",
            )
        video_ref = self.wait_for_active(video_ref)

        # Get CSV references using upload method (handles caching)
        csv_refs = []
        for csv_path in csv_paths:
            csv_ref = self.file_manager.upload_csv(csv_path)
            if csv_ref:
                csv_ref = self.wait_for_active(csv_ref)
                csv_refs.append(csv_ref)

        return video_ref, csv_refs

    def _get_content_items(
        self,
        content_config: List[str],
        video_ref: Any,
        csv_refs: List[Any],
        stage_text: str = None,
    ) -> List[Any]:
        """
        Create content items for Gemini analysis based on configuration.

        This method determines which content items to include based on the configuration.
        It supports different analysis modes (video, pose, video+pose, pose-video, etc.)
        through the 'content_items' configuration parameter. The order of items in the
        returned list respects the order specified in the YAML configuration.

        Args:
            video_ref: Gemini file reference for the video
            csv_refs: List of Gemini file references for CSV files
            stage_text: Optional previous response for multi-stage prompts

        Returns:
            List of content items to pass to Gemini generate_content() in the order
            specified by the YAML configuration

        Raises:
            AmbientError: If required configuration is missing
        """
        content_items = []
        i = 0

        # Process content items in the order specified by the configuration
        while i < len(content_config):
            item_type = content_config[i]
            if item_type.startswith("prompt"):  # prompt1, prompt2, etc.
                prompt_text = self.get_analysis_prompt(item_type)
                prompt_text = prompt_text.format(
                    video=str(video_ref), pose=str(csv_refs)
                )
                # Check if this is a 2nd-stage prompt that needs previous response
                if item_type == "prompt2" and stage_text:
                    prompt_text = prompt_text.format(stage1=stage_text)

                content_items.append(prompt_text)
            elif item_type == "video" and video_ref:
                content_items.append(video_ref)
            elif item_type == "pose" and csv_refs:
                content_items.extend(csv_refs)
            i += 1

        # Validate that we have at least a prompt
        if not content_items:
            raise AmbientError(
                "No content items available for analysis",
                details="At least a prompt is required for analysis",
            )
        print(f"Content items: {content_items}", flush=True)

        return content_items

    def _get_content_config(self, content_items_name: str = "content_items") -> list:
        """
        Get content configuration from config manager with sensible defaults.
        `content_items` could be a list of strings or a dictionary with boolean values.

        Returns:
            Ordered list of content items to include. The order respects
            the order specified in the YAML file.  If the field is not
            found, return an empty list.
        """
        if not self.config_manager or not hasattr(
            self.config_manager, "get_config_value"
        ):
            return []

        # Try to get explicit content configuration
        content_config = self.config_manager.get_config_value(content_items_name)
        if content_config:
            # Handle both list and dictionary formats
            if isinstance(content_config, list):
                # Already in list format, return as-is
                return content_config
            elif isinstance(content_config, dict):
                # Convert dict => ordered list, only include items that are True
                return [item for item, enabled in content_config.items() if enabled]

        # Fallback to default configuration: just []
        return []

    def _get_stage_response(self, content_items: List[Any]) -> tuple:
        """
        Generate the response for this stage of the analysis for the given
        content_items that the VLM needs to take in.

        This method handles the generation of the response from the Gemini model
        for the first stage of analysis.

        Args:
            content_items: List of content items to pass to the model

        Returns:
            Tuple of (stage_response, stage1_text) where:
            - stage_response: The complete Gemini response object
            - stage_text: The extracted analysis text from the response
        """
        stage_text = ""
        try:
            stage_response = self.model.generate_content(
                content_items,
                generation_config=genai.GenerationConfig(temperature=self.temperature),
            )
        except Exception as e:
            print(f"ERROR generating content:::::: {e}", flush=True)
            raise GeminiError(
                message=f"Error generating content: {e.message}", details=f"{e.details}"
            )

        try:
            stage_text = stage_response.candidates[0].content.parts[0].text.strip()
        except Exception:
            try:
                stage_text = stage_response.text.strip()
            except AttributeError:
                stage_text = str(stage_response).strip()

        return stage_response, stage_text

    def analyze_video(self, video_path: str, csv_paths: List[str]) -> tuple:
        """
        Analyze a video with associated CSV files using Gemini.

        Args:
            video_path: Path to the video file
            csv_paths: List of paths to CSV files

        Returns:
            Tuple of (raw_response, generated_text) where:
            - raw_response: The complete Gemini response object
            - generated_text: The extracted analysis text

        Raises:
            GeminiError: If the analysis fails
        """
        try:
            # Get file references
            video_ref, csv_refs = self.get_file_references(video_path, csv_paths)

            if not video_ref:
                raise GeminiError(f"Failed to get video reference for: {video_path}")

            if not csv_refs:
                raise GeminiError(f"Failed to get pose CSV references")

            # ---------------- first stage -----------------
            stage_response = None
            stage_text = ""
            content_config = self._get_content_config(
                content_items_name="content_items1"
            )
            if content_config:
                content_items = self._get_content_items(
                    content_config, video_ref, csv_refs
                )
                stage_response, stage_text = self._get_stage_response(content_items)

                # ------------ second stage (if any) ------------
                content_config = self._get_content_config(
                    content_items_name="content_items2"
                )
                if content_config:
                    content_items = self._get_content_items(
                        content_config, video_ref, csv_refs, stage_text
                    )
                    stage_response, stage_text = self._get_stage_response(content_items)

            return stage_response, stage_text

        except Exception as e:
            raise GeminiError(f"Analysis failed: {str(e)}")

    def get_analysis_prompt(self, prompt_name: str = "prompt1") -> str:
        """
        Get the analysis prompt template with key in the input
        parameter `prompt_name`.

        Args:
            prompt_name: The name of the prompt to get
        Returns:
            The prompt template from configuration.
        Raises:
            AmbientError: If the prompt is not provided in the configuration.
        """
        # Try to get prompt from configuration first with the given prompt_name
        if self.config_manager and hasattr(self.config_manager, "get_config_value"):
            config_prompt = self.config_manager.get_config_value(prompt_name)
            if config_prompt:
                return config_prompt
        raise AmbientError(
            f"The '{prompt_name}' field is required in the YAML configuration file.",
            details=f"Please provide a '{prompt_name}' key in your --config YAML file.",
        )


class GaitAnalyzer:
    """
    Main business logic class for gait analysis.

    This class orchestrates the analysis process, handling file discovery,
    record identification, and coordinating between different components.
    """

    def __init__(
        self,
        config_manager: IConfigurationManager,
        analyzer: IAnalyzer,
        file_manager: Any,
        output_manager: Any,
    ):
        """
        Initialize the gait analyzer.

        Args:
            config_manager: Configuration manager instance
            analyzer: Analyzer instance for AI analysis
            file_manager: File manager instance
            output_manager: Output manager instance
        """
        self.config_manager = config_manager
        self.analyzer = analyzer
        self.file_manager = file_manager
        self.output_manager = output_manager

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(should_retry_exception),
    )
    def analyze_single_video(
        self, video_path: str, record_id: Optional[str] = None
    ) -> bool:
        """
        Analyze a single video file with exponential retry logic.

        Args:
            video_path: Path to the video file
            record_id: Optional record ID, will be extracted from filename if not provided

        Returns:
            True if analysis was successful

        Raises:
            Exception: If analysis fails (will trigger retry)
        """
        try:
            # Extract record ID from filename if not provided
            if record_id is None:
                from pathlib import Path

                base_name = Path(video_path).name
                if not base_name.endswith("-bottom.mp4"):
                    raise ValueError(
                        f"Video {video_path} does not follow expected naming pattern"
                    )
                record_id = base_name.replace("-bottom.mp4", "")

            # Find associated CSV files
            csv_paths = self._find_csv_files(record_id)
            if not csv_paths:
                raise ValueError(f"No OpenPose CSVs found for {record_id}")

            print(f"Analyzing {record_id}...", flush=True)
            print(f"Video: {video_path}", flush=True)
            print(f"CSVs: {csv_paths}", flush=True)

            # Perform analysis
            raw_response, analysis_text = self.analyzer.analyze_video(
                video_path, csv_paths
            )

            # Save results
            self.output_manager.save_analysis_text(record_id, analysis_text)
            self.output_manager.save_raw_response(record_id, raw_response)

            # Display results
            print(f"\n===== Analysis for {record_id} =====")
            print(analysis_text, flush=True)
            print("====================================\n", flush=True)

            return True

        except ValueError:
            # Re-raise ValueError immediately (no retry for validation errors)
            raise
        except Exception as e:
            # Re-raise other exceptions for retry
            raise

    def analyze_all_videos(self) -> None:
        """Analyze all bottom-view videos in the videos directory."""
        videos_dir = self.config_manager.get_videos_directory()

        # Find all bottom videos
        bottom_videos = list(videos_dir.glob("*-bottom.mp4"))

        if not bottom_videos:
            print("No matching videos found.", flush=True)
            return

        print(f"Found {len(bottom_videos)} videos to analyze", flush=True)

        for video_path in bottom_videos:
            base_name = video_path.name
            if not base_name.endswith("-bottom.mp4"):
                continue
            record_id = base_name.replace("-bottom.mp4", "")

            try:
                self.analyze_single_video(str(video_path), record_id)
            except Exception as e:
                print(
                    f"Failed to analyze {record_id} after retries: {str(e)}", flush=True
                )

    def _find_csv_files(self, record_id: str) -> List[str]:
        """
        Find CSV files associated with a record ID.

        Args:
            record_id: The record ID to find CSV files for

        Returns:
            List of CSV file paths
        """
        openpose_dir = self.config_manager.get_openpose_directory()
        csv_pattern = openpose_dir / record_id / f"{record_id}-bottom-gait.csv"

        if csv_pattern.exists():
            return [str(csv_pattern)]
        else:
            return []
