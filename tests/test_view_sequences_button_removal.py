"""
Tests for View Sequences Button Removal

Verifies that the unnecessary "View Sequences" button has been removed
from the GAVD dataset detail page.
"""

import pytest
from pathlib import Path


class TestViewSequencesButtonRemoval:
    """Test suite for View Sequences button removal."""
    
    def test_gavd_dataset_page_exists(self):
        """Verify GAVD dataset page exists."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        assert dataset_page.exists(), "Dataset page should exist"
    
    def test_view_sequences_button_removed(self):
        """Verify 'View Sequences' button has been removed."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Button should not exist
        assert "View Sequences" not in content, "'View Sequences' button should be removed"
        assert "🎥 View Sequences" not in content, "'View Sequences' button with emoji should be removed"
    
    def test_viewer_route_not_referenced(self):
        """Verify /viewer route is not referenced."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Should not link to non-existent viewer route
        assert "/viewer" not in content, "Should not reference /viewer route"
    
    def test_back_to_dashboard_button_exists(self):
        """Verify 'Back to Dashboard' button still exists."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Back button should still exist
        assert "Back to Dashboard" in content, "'Back to Dashboard' button should exist"
    
    def test_sequences_list_exists(self):
        """Verify sequences list is still displayed on the page."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Sequences list should exist
        assert "Sequences List" in content or "sequences.map" in content, \
            "Sequences list should be displayed on the page"
    
    def test_individual_sequence_view_buttons_exist(self):
        """Verify individual sequence 'View' buttons still exist."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Individual sequence view buttons should exist
        assert "View →" in content, "Individual sequence view buttons should exist"
        assert f"/gavd/${{dataset_id}}/sequence/${{sequence.sequence_id}}" in content, \
            "Should link to individual sequence pages"
    
    def test_no_viewer_directory_exists(self):
        """Verify that /viewer directory doesn't exist (confirming it was a 404)."""
        viewer_dir = Path("frontend/app/gavd/[dataset_id]/viewer")
        assert not viewer_dir.exists(), "Viewer directory should not exist"
        
        viewer_page = Path("frontend/app/gavd/[dataset_id]/viewer/page.tsx")
        assert not viewer_page.exists(), "Viewer page should not exist"


class TestGAVDNavigationFlow:
    """Test suite for GAVD navigation flow."""
    
    def test_dataset_to_sequence_navigation(self):
        """Verify navigation flow from dataset to individual sequences."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Should have links to individual sequences
        assert "/sequence/" in content, "Should link to sequence pages"
    
    def test_sequence_page_exists(self):
        """Verify individual sequence page exists."""
        sequence_page = Path("frontend/app/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx")
        assert sequence_page.exists(), "Individual sequence page should exist"
    
    def test_proper_route_structure(self):
        """Verify proper route structure exists."""
        # Dataset page should exist
        assert Path("frontend/app/gavd/[dataset_id]/page.tsx").exists()
        
        # Sequence page should exist
        assert Path("frontend/app/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx").exists()
        
        # Viewer page should NOT exist
        assert not Path("frontend/app/gavd/[dataset_id]/viewer/page.tsx").exists()


class TestUIConsistency:
    """Test suite for UI consistency."""
    
    def test_header_structure(self):
        """Verify header has proper structure after button removal."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Header should have title and back button
        assert "GAVD Dataset Details" in content
        assert "Back to Dashboard" in content
    
    def test_no_conditional_button_rendering(self):
        """Verify no conditional rendering for View Sequences button."""
        dataset_page = Path("frontend/app/gavd/[dataset_id]/page.tsx")
        content = dataset_page.read_text(encoding='utf-8')
        
        # Should not have conditional rendering for View Sequences
        # This pattern was used: {metadata.status === 'completed' && ( <Button>View Sequences</Button> )}
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'View Sequences' in line:
                pytest.fail(f"Found 'View Sequences' on line {i+1}: {line.strip()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
