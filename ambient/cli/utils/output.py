"""
Output formatting utilities for CLI results.
"""

import json
import csv
import io
from typing import Dict, Any, List
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


class OutputFormatter:
    """
    Format analysis results in various output formats.
    """
    
    def format(self, data: Dict[str, Any], format: str) -> str:
        """
        Format data in the specified format.
        
        Args:
            data: Data to format
            format: Output format (json, csv, xml, text)
            
        Returns:
            Formatted string
        """
        if format == 'json':
            return self._format_json(data)
        elif format == 'csv':
            return self._format_csv(data)
        elif format == 'xml':
            return self._format_xml(data)
        elif format == 'text':
            return self._format_text(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format as JSON."""
        return json.dumps(data, indent=2, default=str)
    
    def _format_csv(self, data: Dict[str, Any]) -> str:
        """Format as CSV."""
        output = io.StringIO()
        
        # Handle single result
        if 'video' in data and 'gait_metrics' in data:
            writer = csv.writer(output)
            
            # Write header
            headers = ['video', 'is_normal', 'confidence']
            if 'gait_metrics' in data:
                headers.extend([
                    'stride_length', 'stride_time', 'cadence',
                    'step_width', 'symmetry_index'
                ])
            writer.writerow(headers)
            
            # Write data
            row = [
                data.get('video', ''),
                data.get('classification', {}).get('is_normal', ''),
                data.get('classification', {}).get('confidence', '')
            ]
            
            if 'gait_metrics' in data:
                metrics = data['gait_metrics']
                row.extend([
                    metrics.get('stride_length', ''),
                    metrics.get('stride_time', ''),
                    metrics.get('cadence', ''),
                    metrics.get('step_width', ''),
                    metrics.get('symmetry_index', '')
                ])
            
            writer.writerow(row)
        
        # Handle batch results
        elif 'results' in data and isinstance(data['results'], list):
            writer = csv.writer(output)
            
            # Write header
            headers = ['video', 'is_normal', 'confidence', 'output_file']
            writer.writerow(headers)
            
            # Write data
            for result in data['results']:
                writer.writerow([
                    result.get('video', ''),
                    result.get('is_normal', ''),
                    result.get('confidence', ''),
                    result.get('output_file', '')
                ])
        
        return output.getvalue()
    
    def _format_xml(self, data: Dict[str, Any]) -> str:
        """Format as XML."""
        root = Element('analysis')
        
        self._dict_to_xml(data, root)
        
        # Pretty print
        xml_str = tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")
    
    def _dict_to_xml(self, data: Any, parent: Element):
        """Convert dictionary to XML elements."""
        if isinstance(data, dict):
            for key, value in data.items():
                child = SubElement(parent, str(key))
                self._dict_to_xml(value, child)
        elif isinstance(data, list):
            for item in data:
                item_elem = SubElement(parent, 'item')
                self._dict_to_xml(item, item_elem)
        else:
            parent.text = str(data)
    
    def _format_text(self, data: Dict[str, Any]) -> str:
        """Format as human-readable text."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("AlexPose Gait Analysis Results")
        lines.append("=" * 60)
        
        # Video information
        if 'video' in data:
            lines.append(f"\nVideo: {data['video']}")
        
        # Analysis information
        if 'analysis' in data:
            analysis = data['analysis']
            lines.append("\nAnalysis:")
            lines.append(f"  Frame Count: {analysis.get('frame_count', 'N/A')}")
            lines.append(f"  Duration: {analysis.get('duration', 'N/A')}s")
            lines.append(f"  Frame Rate: {analysis.get('frame_rate', 'N/A')} fps")
            lines.append(f"  Pose Estimator: {analysis.get('pose_estimator', 'N/A')}")
        
        # Gait metrics
        if 'gait_metrics' in data:
            metrics = data['gait_metrics']
            lines.append("\nGait Metrics:")
            lines.append(f"  Stride Length: {metrics.get('stride_length', 'N/A'):.2f}")
            lines.append(f"  Stride Time: {metrics.get('stride_time', 'N/A'):.2f}s")
            lines.append(f"  Cadence: {metrics.get('cadence', 'N/A'):.2f} steps/min")
            lines.append(f"  Step Width: {metrics.get('step_width', 'N/A'):.2f}")
            lines.append(f"  Symmetry Index: {metrics.get('symmetry_index', 'N/A'):.2f}")
        
        # Classification
        if 'classification' in data:
            classification = data['classification']
            lines.append("\nClassification:")
            
            is_normal = classification.get('is_normal')
            if is_normal is not None:
                status = "NORMAL" if is_normal else "ABNORMAL"
                lines.append(f"  Status: {status}")
            
            confidence = classification.get('confidence')
            if confidence is not None:
                lines.append(f"  Confidence: {confidence:.2%}")
            
            explanation = classification.get('explanation')
            if explanation:
                lines.append(f"\n  Explanation:")
                lines.append(f"  {explanation}")
            
            conditions = classification.get('conditions', [])
            if conditions:
                lines.append(f"\n  Identified Conditions:")
                for condition in conditions:
                    if isinstance(condition, dict):
                        lines.append(f"    - {condition.get('name', 'Unknown')}: {condition.get('confidence', 0):.2%}")
                    else:
                        lines.append(f"    - {condition}")
        
        # Batch summary
        if 'total_videos' in data:
            lines.append(f"\nBatch Summary:")
            lines.append(f"  Total Videos: {data.get('total_videos', 0)}")
            lines.append(f"  Successful: {data.get('successful', 0)}")
            lines.append(f"  Failed: {data.get('failed', 0)}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
