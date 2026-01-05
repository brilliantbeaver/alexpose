"""
Integration tests for GAVD dataset upload and processing.

Tests the complete workflow from CSV upload to pose estimation.
"""

import pytest
import tempfile
import csv
from pathlib import Path
from fastapi.testclient import TestClient
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from server.main import app
from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_gavd_csv():
    """Create a sample GAVD CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'seq', 'frame_num', 'cam_view', 'gait_event', 'dataset',
            'gait_pat', 'bbox', 'vid_info', 'id', 'url'
        ])
        writer.writeheader()
        
        # Write sample rows
        for i in range(10):
            writer.writerow({
                'seq': 'test_seq_001',
                'frame_num': str(1757 + i),
                'cam_view': 'right side',
                'gait_event': 'Right initial contact' if i == 5 else '',
                'dataset': 'Abnormal Gait',
                'gait_pat': 'parkinsons',
                'bbox': "{'top': 125.0, 'left': 156.0, 'height': 497.0, 'width': 228.0}",
                'vid_info': "{'height': 720, 'width': 1280, 'mime_type': 'video/mp4'}",
                'id': 'B5hrxKe2nP8',
                'url': 'https://www.youtube.com/watch?v=B5hrxKe2nP8'
            })
        
        return Path(f.name)


def test_gavd_csv_validation(client, sample_gavd_csv):
    """Test GAVD CSV file validation."""
    with open(sample_gavd_csv, 'rb') as f:
        response = client.post(
            '/api/v1/gavd/upload',
            files={'file': ('test_gavd.csv', f, 'text/csv')},
            data={'process_immediately': 'false'}
        )
    
    assert response.status_code == 200
    result = response.json()
    assert result['success'] is True
    assert 'dataset_id' in result
    assert result['row_count'] > 0
    assert result['sequence_count'] > 0


def test_gavd_invalid_file_type(client):
    """Test rejection of non-CSV files."""
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        f.write(b'This is not a CSV file')
        f.seek(0)
        
        response = client.post(
            '/api/v1/gavd/upload',
            files={'file': ('test.txt', f, 'text/plain')}
        )
    
    assert response.status_code == 400


def test_gavd_missing_columns(client):
    """Test rejection of CSV with missing required columns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seq', 'frame_num'])  # Missing required columns
        writer.writeheader()
        writer.writerow({'seq': 'test', 'frame_num': '1'})
        temp_path = Path(f.name)
    
    try:
        with open(temp_path, 'rb') as f:
            response = client.post(
                '/api/v1/gavd/upload',
                files={'file': ('invalid.csv', f, 'text/csv')}
            )
        
        assert response.status_code == 400
        assert 'Missing required columns' in response.json()['detail']
    finally:
        temp_path.unlink()


def test_gavd_dataset_status(client, sample_gavd_csv):
    """Test retrieving dataset status."""
    # Upload dataset
    with open(sample_gavd_csv, 'rb') as f:
        upload_response = client.post(
            '/api/v1/gavd/upload',
            files={'file': ('test_gavd.csv', f, 'text/csv')},
            data={'process_immediately': 'false'}
        )
    
    assert upload_response.status_code == 200
    dataset_id = upload_response.json()['dataset_id']
    
    # Get status
    status_response = client.get(f'/api/v1/gavd/status/{dataset_id}')
    assert status_response.status_code == 200
    
    status = status_response.json()
    assert status['success'] is True
    assert status['metadata']['dataset_id'] == dataset_id
    assert status['metadata']['status'] == 'uploaded'


def test_gavd_list_datasets(client, sample_gavd_csv):
    """Test listing GAVD datasets."""
    # Upload a dataset
    with open(sample_gavd_csv, 'rb') as f:
        client.post(
            '/api/v1/gavd/upload',
            files={'file': ('test_gavd.csv', f, 'text/csv')},
            data={'process_immediately': 'false'}
        )
    
    # List datasets
    response = client.get('/api/v1/gavd/list')
    assert response.status_code == 200
    
    result = response.json()
    assert result['success'] is True
    assert result['count'] > 0
    assert len(result['datasets']) > 0


def test_gavd_delete_dataset(client, sample_gavd_csv):
    """Test deleting a GAVD dataset."""
    # Upload dataset
    with open(sample_gavd_csv, 'rb') as f:
        upload_response = client.post(
            '/api/v1/gavd/upload',
            files={'file': ('test_gavd.csv', f, 'text/csv')},
            data={'process_immediately': 'false'}
        )
    
    dataset_id = upload_response.json()['dataset_id']
    
    # Delete dataset
    delete_response = client.delete(f'/api/v1/gavd/{dataset_id}')
    assert delete_response.status_code == 200
    assert delete_response.json()['success'] is True
    
    # Verify deletion
    status_response = client.get(f'/api/v1/gavd/status/{dataset_id}')
    assert status_response.status_code == 404


def test_gavd_service_metadata_operations():
    """Test GAVD service metadata operations."""
    config_manager = ConfigurationManager()
    service = GAVDService(config_manager)
    
    # Test save and retrieve metadata
    test_metadata = {
        'dataset_id': 'test_123',
        'filename': 'test.csv',
        'status': 'uploaded'
    }
    
    service.save_dataset_metadata('test_123', test_metadata)
    retrieved = service.get_dataset_metadata('test_123')
    
    assert retrieved is not None
    assert retrieved['dataset_id'] == 'test_123'
    assert retrieved['filename'] == 'test.csv'
    
    # Test update metadata
    service.update_dataset_metadata('test_123', {'status': 'processing'})
    updated = service.get_dataset_metadata('test_123')
    assert updated['status'] == 'processing'
    
    # Cleanup
    service.delete_dataset('test_123')


def test_gavd_processor_integration(sample_gavd_csv):
    """Test GAVD processor with sample CSV."""
    from ambient.gavd.gavd_processor import create_gavd_processor
    
    processor = create_gavd_processor()
    
    # Process the sample CSV (without pose estimation to keep test fast)
    results = processor.process_gavd_file(
        csv_file_path=str(sample_gavd_csv),
        max_sequences=1,
        include_metadata=True,
        verbose=False
    )
    
    assert results['total_sequences'] > 0
    assert 'sequences' in results
    assert 'summary' in results
    assert results['summary']['total_frames'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
