"""
YouTube caching utilities.

This module provides high-level helpers to:
- Extract unique YouTube URLs from a list of row dictionaries
- Download those URLs to a local cache directory using yt-dlp

Design goals:
- Keep business logic in small, composable functions
- Be resilient to partial failures and network hiccups
- Produce clear logs and return a structured result summary

Note: yt-dlp is an optional dependency. Install with:
    pip install yt-dlp

On Windows, install ffmpeg (for muxing bestvideo+bestaudio) via one of:
    winget install --id=Gyan.FFmpeg -e --source winget
    choco install ffmpeg -y
"""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse


YOUTUBE_HOST_HINTS: Tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "m.youtube.com",
    "music.youtube.com",
)


def extract_unique_youtube_urls(rows: List[Dict[str, Any]]) -> Set[str]:
    """Extract a set of unique YouTube URLs from a list of row dicts.

    A row is expected to potentially contain a key named "url".

    Args:
        rows: List of dictionaries representing CSV rows.

    Returns:
        Set of unique YouTube URLs (strings), filtered for plausible hosts.
    """
    unique_urls: Set[str] = set()
    for row in rows:
        value = row.get("url")
        if not value or not isinstance(value, str):
            continue
        lowered = value.lower()
        if any(host in lowered for host in YOUTUBE_HOST_HINTS):
            unique_urls.add(value.strip())
    return unique_urls


def _find_project_root(start: Optional[Path] = None) -> Path:
    """Find the project root by looking for pyproject.toml or .git.

    Falls back to current working directory.
    """
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current


def _ensure_output_dir(output_dir: Path | str, use_project_root: bool = True) -> Path:
    """Resolve and create the output directory if needed."""
    output_path = Path(output_dir)
    if not output_path.is_absolute() and use_project_root:
        output_path = _find_project_root() / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


@dataclass
class YtDlpConfig:
    """Configuration for YouTube downloads with yt-dlp.

    Fields correspond to yt-dlp options and high-level behavior toggles.
    """

    output_dir: Path | str = "data/youtube"
    cookies_file: Optional[Path | str] = None
    ffmpeg_location: Optional[Path | str] = None
    format_selector: Optional[str] = None
    max_concurrent_fragments: int = 3
    max_retries: int = 10
    rate_limit: Optional[str] = None
    filename_scheme: str = "id"  # 'id' (recommended) or 'legacy'
    # Workarounds for recent YouTube SABR streaming issues
    extractor_args: Optional[Dict[str, Any]] = None
    prefer_non_hls: bool = True


def _build_output_template(target_dir: Path, filename_scheme: str) -> str:
    """Return the yt-dlp output template for the chosen scheme."""
    if filename_scheme == "id":
        return str(target_dir / "%(id)s.%(ext)s")
    if filename_scheme == "legacy":
        return str(target_dir / "%(uploader)s/%(title).200B [%(id)s].%(ext)s")
    return str(target_dir / "%(id)s.%(ext)s")


def _build_ydl_options(template: str, cfg: YtDlpConfig) -> Dict[str, Any]:
    """Construct yt-dlp options from configuration."""
    # Build a robust default format string; prefer non-HLS to avoid SABR m3u8 fragment errors
    if cfg.prefer_non_hls:
        default_format = (
            "bv*[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]"
            "+ba[ext=m4a][protocol!=m3u8][protocol!=m3u8_native]/"
            "bv*[protocol!=m3u8][protocol!=m3u8_native]"
            "+ba[protocol!=m3u8][protocol!=m3u8_native]/"
            "best[protocol!=m3u8][protocol!=m3u8_native]/best"
        )
    else:
        default_format = "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/bestvideo+bestaudio/best"
    ydl_opts: Dict[str, Any] = {
        "outtmpl": template,
        "format": cfg.format_selector or default_format,
        "merge_output_format": "mp4",
        "windowsfilenames": True,
        "restrictfilenames": False,
        "concurrent_fragment_downloads": max(1, int(cfg.max_concurrent_fragments)),
        "retries": max(1, int(cfg.max_retries)),
        "fragment_retries": max(1, int(cfg.max_retries)),
        "noprogress": False,
        "ignoreerrors": True,
        "overwrites": False,
        "continuedl": True,
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [
            {"key": "FFmpegMetadata", "add_metadata": True, "add_chapters": True},
        ],
    }

    if cfg.rate_limit:
        ydl_opts["ratelimit"] = cfg.rate_limit
    if cfg.cookies_file:
        ydl_opts["cookiefile"] = str(cfg.cookies_file)
    if cfg.ffmpeg_location:
        ydl_opts["ffmpeg_location"] = str(cfg.ffmpeg_location)
    # Apply extractor args; default to android client to avoid SABR for web client
    extractor_args = cfg.extractor_args
    if extractor_args is None:
        extractor_args = {"youtube": {"player_client": ["android"]}}
    ydl_opts["extractor_args"] = extractor_args

    return ydl_opts


def extract_video_id(url: str) -> Optional[str]:
    """Extract a YouTube video ID from watch and youtu.be URLs.

    Returns None when the ID cannot be determined.
    """
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").strip("/")
        if host.endswith("youtu.be"):
            # Short link: /<id>[?...]
            vid = path.split("/")[0]
            return vid or None
        # Handle /shorts/<id>, /embed/<id>, /live/<id>
        for prefix in ("shorts/", "embed/", "live/"):
            if path.startswith(prefix):
                seg = path[len(prefix) :].split("/")[0]
                return seg or None
        # Standard watch URL
        query = parse_qs(parsed.query)
        vid_list = query.get("v")
        if vid_list and vid_list[0]:
            return vid_list[0]
        return None
    except Exception:
        return None


def _is_nonfinal_artifact(candidate: Path) -> bool:
    """Return True for temporary/metadata files that are not final media outputs."""
    suffix = candidate.suffix.lower()
    if suffix in {".part", ".ytdl", ".temp", ".json"}:
        return True
    name = candidate.name
    return name.endswith(".info.json") or name.endswith(".metadata.json")


def find_existing_download(target_dir: Path, video_id: str) -> Optional[Path]:
    """Find an existing non-zero-sized file for a given video ID in target_dir."""
    for candidate in target_dir.glob(f"{video_id}.*"):
        if not candidate.is_file():
            continue
        if _is_nonfinal_artifact(candidate):
            continue
        try:
            if candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def _prepare_download_list(
    urls: List[str], target_dir: Path, filename_scheme: str, results: Dict[str, Any]
) -> List[str]:
    """Apply pre-checks (skip if already downloaded) and return URLs to fetch."""
    if filename_scheme != "id":
        return list(urls)

    to_download: List[str] = []
    for url in urls:
        video_id = extract_video_id(url)
        if not video_id:
            to_download.append(url)
            continue
        existing = find_existing_download(target_dir, video_id)
        if existing is None:
            to_download.append(url)
            continue
        logger.info("Already cached (skipping): {}", existing)
        results["skipped"] += 1
        results["details"][url] = {"status": "skipped_existing", "path": str(existing)}
    return to_download

def download_youtube_urls(
    urls: Iterable[str],
    output_dir: Path | str = "data/youtube",
    *,
    cookies_file: Optional[Path | str] = None,
    ffmpeg_location: Optional[Path | str] = None,
    format_selector: Optional[str] = None,
    max_concurrent_fragments: int = 3,
    max_retries: int = 10,
    rate_limit: Optional[str] = None,
    filename_scheme: str = "id",  # 'id' (recommended) or 'legacy'
) -> Dict[str, Any]:
    """Download YouTube videos to a local cache using yt-dlp.

    Args:
        urls: Iterable of YouTube URLs.
        output_dir: Target directory (relative to project root by default).
        cookies_file: Path to cookies.txt (Netscape format) for authenticated downloads.
        ffmpeg_location: Optional path to ffmpeg binary directory or executable.
        format_selector: Custom yt-dlp format string. If None, a robust default is used.
        max_concurrent_fragments: Fragment concurrency for faster downloads.
        max_retries: Number of retries for network/fragment errors.
        rate_limit: Optional rate limit like "5M" (bytes per second) to throttle.

    Returns:
        Dictionary with summary statistics and per-URL results.
    """
    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except Exception as import_error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "yt-dlp is required. Install with `pip install yt-dlp`."
        ) from import_error

    target_dir = _ensure_output_dir(output_dir, use_project_root=True)
    cfg = YtDlpConfig(
        output_dir=target_dir,
        cookies_file=cookies_file,
        ffmpeg_location=ffmpeg_location,
        format_selector=format_selector,
        max_concurrent_fragments=max_concurrent_fragments,
        max_retries=max_retries,
        rate_limit=rate_limit,
        filename_scheme=filename_scheme,
    )
    out_template = _build_output_template(target_dir, cfg.filename_scheme)
    ydl_opts = _build_ydl_options(out_template, cfg)

    results: Dict[str, Any] = {
        "attempted": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "details": {},  # url -> {status, error?}
        "output_dir": str(target_dir),
    }

    url_list = [u for u in urls if isinstance(u, str) and u.strip()]
    results["attempted"] = len(url_list)

    if not url_list:
        logger.info("No URLs to download.")
        return results

    # Pre-skip if using id-based scheme and file already present
    to_download = _prepare_download_list(url_list, target_dir, filename_scheme, results)

    if not to_download:
        return results

    with YoutubeDL(ydl_opts) as ydl:
        for url in to_download:
            try:
                logger.info("Downloading: {}", url)
                retcode = ydl.download([url])
                # yt-dlp returns 0 on success, 1 on errors; but also may skip existing
                if retcode == 0:
                    results["downloaded"] += 1
                    results["details"][url] = {"status": "downloaded"}
                else:
                    # Could be skipped or partial; yt-dlp API doesn't clearly expose skip
                    # status here, so record as failed unless output indicates otherwise.
                    results["failed"] += 1
                    results["details"][url] = {
                        "status": "failed",
                        "error": f"yt-dlp returned code {retcode}",
                    }
            except Exception as e:  # pragma: no cover - network dependent
                logger.warning("Failed to download {}: {}", url, e)
                results["failed"] += 1
                results["details"][url] = {"status": "failed", "error": str(e)}

    return results


def cache_youtube_videos_from_rows(
    rows: List[Dict[str, Any]],
    output_dir: Path | str = "data/youtube",
    *,
    cookies_file: Optional[Path | str] = "config/yt_cookies.txt",
    ffmpeg_location: Optional[Path | str] = None,
    format_selector: Optional[str] = None,
    filename_scheme: str = "id",
) -> Dict[str, Any]:
    """High-level convenience: extract URLs from rows and download them.

    Args:
        rows: List of row dictionaries containing at least a "url" field.
        output_dir: Target directory (relative to project root by default).
        cookies_file: Optional path to cookies.txt for Premium access.
        ffmpeg_location: Optional path to ffmpeg binary.
        format_selector: Optional yt-dlp format string override.
        logger: Optional logger instance.

    Returns:
        Summary dictionary returned by download_youtube_urls.
    """
    urls = extract_unique_youtube_urls(rows)
    return download_youtube_urls(
        urls=sorted(urls),
        output_dir=output_dir,
        cookies_file=cookies_file,
        ffmpeg_location=ffmpeg_location,
        format_selector=format_selector,
        filename_scheme=filename_scheme,
    )


