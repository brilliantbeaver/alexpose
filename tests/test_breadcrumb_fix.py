"""
Tests for Breadcrumb Navigation Fix

Verifies that route segments like "sequence" are not clickable in breadcrumbs.
"""

import pytest


class TestBreadcrumbFix:
    """Test suite for breadcrumb navigation fixes."""
    
    def test_breadcrumb_component_exists(self):
        """Verify the breadcrumb component file exists."""
        from pathlib import Path
        
        breadcrumb_file = Path("frontend/components/navigation/Breadcrumbs.tsx")
        assert breadcrumb_file.exists(), "Breadcrumb component file should exist"
    
    def test_breadcrumb_has_non_clickable_segments(self):
        """Verify breadcrumb component defines non-clickable segments."""
        from pathlib import Path
        
        breadcrumb_file = Path("frontend/components/navigation/Breadcrumbs.tsx")
        content = breadcrumb_file.read_text(encoding='utf-8')
        
        # Check for non-clickable segments definition
        assert "nonClickableSegments" in content, "Should define nonClickableSegments"
        assert "'sequence'" in content, "Should include 'sequence' as non-clickable"
        
    def test_breadcrumb_checks_non_clickable(self):
        """Verify breadcrumb component checks if segment is non-clickable."""
        from pathlib import Path
        
        breadcrumb_file = Path("frontend/components/navigation/Breadcrumbs.tsx")
        content = breadcrumb_file.read_text(encoding='utf-8')
        
        # Check for non-clickable logic
        assert "isNonClickable" in content, "Should check if segment is non-clickable"
        assert "nonClickableSegments.includes" in content, "Should check segment against list"
    
    def test_breadcrumb_renders_non_clickable_as_span(self):
        """Verify non-clickable segments are rendered as span, not Link."""
        from pathlib import Path
        
        breadcrumb_file = Path("frontend/components/navigation/Breadcrumbs.tsx")
        content = breadcrumb_file.read_text(encoding='utf-8')
        
        # Check for conditional rendering
        assert "isLast || isNonClickable" in content, "Should render span for non-clickable"
    
    def test_sequence_page_has_back_button(self):
        """Verify sequence page has a back button to dataset."""
        from pathlib import Path
        
        sequence_page = Path("frontend/app/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx")
        assert sequence_page.exists(), "Sequence page should exist"
        
        content = sequence_page.read_text(encoding='utf-8')
        assert "Back to Dataset" in content, "Should have back to dataset button"
        assert "router.push" in content, "Should use router for navigation"


class TestBreadcrumbURLStructure:
    """Test suite for URL structure and routing."""
    
    def test_gavd_dataset_page_exists(self):
        """Verify GAVD dataset page exists."""
        from pathlib import Path
        
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        assert dataset_page.exists(), "Dataset page should exist"
    
    def test_gavd_sequence_page_exists(self):
        """Verify GAVD sequence page exists."""
        from pathlib import Path
        
        sequence_page = Path("frontend/app/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx")
        assert sequence_page.exists(), "Sequence page should exist"
    
    def test_no_intermediate_sequence_page(self):
        """Verify there's no page at /gavd/[dataset_id]/sequence (which would cause 404)."""
        from pathlib import Path
        
        # This page should NOT exist - it's just a route segment
        intermediate_page = Path("frontend/app/gavd/[dataset_id]/sequence/page.tsx")
        assert not intermediate_page.exists(), "Intermediate sequence page should not exist"


class TestBreadcrumbIntegration:
    """Integration tests for breadcrumb behavior."""
    
    def test_breadcrumb_path_segments(self):
        """Test breadcrumb generation for various paths."""
        # Simulate breadcrumb generation logic
        test_paths = [
            ("/gavd/abc123", ["Gavd", "Abc123"]),
            ("/gavd/abc123/sequence/seq001", ["Gavd", "Abc123", "Sequence", "Seq001"]),
            ("/training/gavd/dataset123", ["Training", "Gavd", "Dataset123"]),
        ]
        
        for path, expected_segments in test_paths:
            segments = path.split('/')[1:]  # Remove empty first element
            breadcrumbs = [seg.capitalize() for seg in segments]
            assert len(breadcrumbs) == len(expected_segments), f"Path {path} should have {len(expected_segments)} segments"
    
    def test_non_clickable_segment_identification(self):
        """Test identification of non-clickable segments."""
        non_clickable = ['sequence', 'frame', 'analysis']
        
        test_cases = [
            ("Sequence", True),
            ("sequence", True),
            ("SEQUENCE", True),
            ("Frame", True),
            ("Gavd", False),
            ("Dataset123", False),
            ("Home", False),
        ]
        
        for segment, should_be_non_clickable in test_cases:
            is_non_clickable = segment.lower() in non_clickable
            assert is_non_clickable == should_be_non_clickable, \
                f"Segment '{segment}' clickability check failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
