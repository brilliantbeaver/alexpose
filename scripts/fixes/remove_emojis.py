#!/usr/bin/env python3
"""
Remove emoji characters from Python files to fix Windows encoding issues.

This script replaces emoji characters with plain text equivalents.
"""

import re
from pathlib import Path

# Emoji replacements
EMOJI_REPLACEMENTS = {
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    'ℹ️': '[INFO]',
    '🎯': '[TARGET]',
    '👁️': '[EYE]',
    '📊': '[CHART]',
    '📋': '[CLIPBOARD]',
    '📍': '[PIN]',
    '📹': '[VIDEO]',
    '🔍': '[SEARCH]',
    '🔧': '[WRENCH]',
    '💡': '[BULB]',
}

def remove_emojis_from_file(file_path: Path) -> int:
    """Remove emojis from a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Replace each emoji
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)
        
        # Count changes
        changes = sum(1 for e in EMOJI_REPLACEMENTS.keys() if e in original_content)
        
        if changes > 0:
            file_path.write_text(content, encoding='utf-8')
            print(f"✓ {file_path}: {changes} emoji(s) replaced")
        
        return changes
        
    except Exception as e:
        print(f"✗ {file_path}: {e}")
        return 0


def main():
    """Remove emojis from all Python files in ambient/."""
    project_root = Path(__file__).parent.parent
    ambient_dir = project_root / "ambient"
    
    total_files = 0
    total_changes = 0
    
    print("Removing emoji characters from Python files...")
    print("=" * 60)
    
    for py_file in ambient_dir.rglob("*.py"):
        changes = remove_emojis_from_file(py_file)
        if changes > 0:
            total_files += 1
            total_changes += changes
    
    print("=" * 60)
    print(f"Summary: {total_changes} emoji(s) removed from {total_files} file(s)")


if __name__ == "__main__":
    main()
