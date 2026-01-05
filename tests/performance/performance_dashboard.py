"""Performance dashboard generator for visualizing performance trends and reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging

from .performance_reporter import PerformanceReporter

logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Generate HTML performance dashboards with charts and trend analysis."""
    
    def __init__(self, reporter: PerformanceReporter):
        self.reporter = reporter
        self.dashboard_dir = Path("tests/performance/dashboards")
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dashboard(self, 
                          dashboard_type: str = "comprehensive",
                          days_back: int = 30) -> Path:
        """Generate HTML performance dashboard."""
        
        # Get dashboard data
        dashboard_data = self.reporter.get_performance_dashboard_data()
        
        # Generate HTML content based on type
        if dashboard_type == "comprehensive":
            html_content = self._generate_comprehensive_dashboard(dashboard_data, days_back)
        elif dashboard_type == "summary":
            html_content = self._generate_summary_dashboard(dashboard_data, days_back)
        elif dashboard_type == "trends":
            html_content = self._generate_trends_dashboard(dashboard_data, days_back)
        else:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")
        
        # Save dashboard
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_filename = f"performance_dashboard_{dashboard_type}_{timestamp_str}.html"
        dashboard_path = self.dashboard_dir / dashboard_filename
        
        # Ensure directory exists
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save as latest
        latest_path = self.dashboard_dir / f"latest_{dashboard_type}_dashboard.html"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Performance dashboard generated: {dashboard_path}")
        return dashboard_path
    
    def _generate_comprehensive_dashboard(self, data: Dict[str, Any], days_back: int) -> str:
        """Generate comprehensive performance dashboard."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlexPose Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .alert-info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .test-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .test-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .test-item:last-child {
            border-bottom: none;
        }
        .test-name {
            font-weight: 500;
        }
        .test-value {
            color: #667eea;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }
        .recommendations {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        .recommendations h4 {
            margin-top: 0;
            color: #856404;
        }
        .recommendations ul {
            margin-bottom: 0;
        }
        .recommendations li {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AlexPose Performance Dashboard</h1>
        <p>Comprehensive performance monitoring and trend analysis</p>
        <p>Generated: {timestamp} | Period: Last {days_back} days</p>
    </div>

    <div class="metric-grid">
        <div class="metric">
            <div class="metric-value">{total_tests}</div>
            <div class="metric-label">Tests Tracked</div>
        </div>
        <div class="metric">
            <div class="metric-value">{total_trends}</div>
            <div class="metric-label">Trends Analyzed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{regressions}</div>
            <div class="metric-label">Regressions Detected</div>
        </div>
        <div class="metric">
            <div class="metric-value">{data_points}</div>
            <div class="metric-label">Total Data Points</div>
        </div>
    </div>

    {alerts_section}

    <div class="dashboard-grid">
        <div class="card">
            <h3>📈 Performance Trends</h3>
            <div class="chart-container">
                <canvas id="trendsChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>⚡ Top Performing Tests</h3>
            <div class="test-list">
                {top_performing_tests}
            </div>
        </div>

        <div class="card">
            <h3>🐌 Slowest Tests</h3>
            <div class="test-list">
                {slowest_tests}
            </div>
        </div>

        <div class="card">
            <h3>📊 Execution Time Distribution</h3>
            <div class="chart-container">
                <canvas id="distributionChart"></canvas>
            </div>
        </div>
    </div>

    {recommendations_section}

    <div class="footer">
        <p>AlexPose Performance Monitoring System | Last updated: {timestamp}</p>
    </div>

    <script>
        // Performance trends chart
        const trendsCtx = document.getElementById('trendsChart').getContext('2d');
        new Chart(trendsCtx, {{
            type: 'line',
            data: {trends_chart_data},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Execution Time (seconds)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Time'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    title: {{
                        display: true,
                        text: 'Performance Trends Over Time'
                    }}
                }}
            }}
        }});

        // Distribution chart
        const distributionCtx = document.getElementById('distributionChart').getContext('2d');
        new Chart(distributionCtx, {{
            type: 'bar',
            data: {distribution_chart_data},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Number of Tests'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Execution Time Range (seconds)'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: true,
                        text: 'Test Execution Time Distribution'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """
        
        # Prepare data for template
        summary = data.get('summary', {})
        
        # Generate alerts section
        alerts_html = self._generate_alerts_section(data.get('regression_alerts', []))
        
        # Generate test lists
        top_tests_html = self._generate_test_list(data.get('top_performing_tests', []), 'time')
        slow_tests_html = self._generate_test_list(data.get('slowest_tests', []), 'time')
        
        # Generate chart data
        trends_chart_data = self._generate_trends_chart_data(data.get('trend_charts', {}))
        distribution_chart_data = self._generate_distribution_chart_data(data)
        
        # Generate recommendations
        recommendations_html = self._generate_recommendations_section()
        
        # Fill template - use double braces for JavaScript
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            days_back=days_back,
            total_tests=summary.get('total_tests_tracked', 0),
            total_trends=summary.get('total_trends_analyzed', 0),
            regressions=summary.get('regressions_detected', 0),
            data_points=summary.get('data_points_total', 0),
            alerts_section=alerts_html,
            top_performing_tests=top_tests_html,
            slowest_tests=slow_tests_html,
            trends_chart_data=trends_chart_data,
            distribution_chart_data=distribution_chart_data,
            recommendations_section=recommendations_html
        )
        
        # Replace double braces with single braces for JavaScript
        html_content = html_content.replace('{{', '{').replace('}}', '}')
        
        return html_content
    
    def _generate_summary_dashboard(self, data: Dict[str, Any], days_back: int) -> str:
        """Generate summary performance dashboard."""
        
        summary = data.get('summary', {})
        regression_count = summary.get('regressions_detected', 0)
        
        # Determine regression status
        if regression_count == 0:
            regression_status = 'status-good'
        elif regression_count <= 2:
            regression_status = 'status-warning'
        else:
            regression_status = 'status-danger'
        
        # Generate quick alerts
        alerts = data.get('regression_alerts', [])
        quick_alerts = ""
        if alerts:
            quick_alerts = f"""
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; 
                        padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center;">
                <strong>⚠️ {len(alerts)} Performance Alert(s)</strong><br>
                <a href="latest_comprehensive_dashboard.html">View Details</a>
            </div>
            """
        
        # Build HTML content directly
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlexPose Performance Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .summary-value {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .summary-label {{
            color: #666;
            font-size: 1.1em;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-good {{ background-color: #28a745; }}
        .status-warning {{ background-color: #ffc107; }}
        .status-danger {{ background-color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Performance Summary</h1>
        <p>Quick overview of system performance health</p>
        <p>Period: Last {days_back} days</p>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-value">{summary.get('total_tests_tracked', 0)}</div>
            <div class="summary-label">
                <span class="status-indicator status-good"></span>
                Tests Monitored
            </div>
        </div>
        
        <div class="summary-card">
            <div class="summary-value">{regression_count}</div>
            <div class="summary-label">
                <span class="status-indicator {regression_status}"></span>
                Regressions Detected
            </div>
        </div>
        
        <div class="summary-card">
            <div class="summary-value">{summary.get('avg_execution_time', 0):.2f}s</div>
            <div class="summary-label">
                <span class="status-indicator status-good"></span>
                Avg Execution Time
            </div>
        </div>
        
        <div class="summary-card">
            <div class="summary-value">{summary.get('data_points_total', 0)}</div>
            <div class="summary-label">
                <span class="status-indicator status-good"></span>
                Data Points Collected
            </div>
        </div>
    </div>

    {quick_alerts}

    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><a href="latest_comprehensive_dashboard.html">View Detailed Dashboard</a></p>
    </div>
</body>
</html>"""
        
        return html_content
    
    def _generate_trends_dashboard(self, data: Dict[str, Any], days_back: int) -> str:
        """Generate trends-focused dashboard."""
        # Implementation for trends-specific dashboard
        # This would focus on detailed trend analysis and charts
        return self._generate_comprehensive_dashboard(data, days_back)  # Simplified for now
    
    def _generate_alerts_section(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate HTML for alerts section."""
        if not alerts:
            return """
            <div class="alert alert-success">
                <strong>✅ All Clear!</strong> No performance regressions detected.
            </div>
            """
        
        alerts_html = ""
        for alert in alerts:
            severity_class = "alert-danger" if alert.get('severity') == 'high' else "alert-info"
            alerts_html += f"""
            <div class="alert {severity_class}">
                <strong>⚠️ Performance Regression:</strong> 
                {alert['test_name']} - {alert['metric']} 
                (Trend: {alert['trend_direction']}, Strength: {alert['trend_strength']:.2f})
            </div>
            """
        
        return alerts_html
    
    def _generate_test_list(self, tests: List[Dict[str, Any]], value_type: str) -> str:
        """Generate HTML for test lists."""
        if not tests:
            return "<p>No data available</p>"
        
        html = ""
        for test in tests:
            test_name = test.get('test_name', 'Unknown')
            if value_type == 'time':
                value = f"{test.get('execution_time', 0):.3f}s"
            else:
                value = str(test.get('value', 'N/A'))
            
            html += f"""
            <div class="test-item">
                <span class="test-name">{test_name}</span>
                <span class="test-value">{value}</span>
            </div>
            """
        
        return html
    
    def _generate_trends_chart_data(self, trend_charts: Dict[str, Any]) -> str:
        """Generate Chart.js data for trends chart."""
        if not trend_charts:
            return json.dumps({
                'labels': [],
                'datasets': []
            })
        
        # Take first few trends for visualization
        selected_trends = list(trend_charts.items())[:5]  # Limit to 5 trends
        
        datasets = []
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        
        for i, (trend_key, trend_data) in enumerate(selected_trends):
            color = colors[i % len(colors)]
            datasets.append({
                'label': f"{trend_data['test_name']} - {trend_data['metric_name']}",
                'data': trend_data['values'],
                'borderColor': color,
                'backgroundColor': color + '20',
                'tension': 0.1
            })
        
        # Use timestamps from first trend as labels
        labels = []
        if selected_trends:
            first_trend = selected_trends[0][1]
            labels = [ts.split('T')[0] for ts in first_trend.get('timestamps', [])]  # Date only
        
        chart_data = {
            'labels': labels,
            'datasets': datasets
        }
        
        return json.dumps(chart_data)
    
    def _generate_distribution_chart_data(self, data: Dict[str, Any]) -> str:
        """Generate Chart.js data for execution time distribution."""
        # Create execution time buckets
        buckets = ['<1s', '1-5s', '5-10s', '10-30s', '30s+']
        bucket_counts = [0, 0, 0, 0, 0]
        
        # Analyze slowest tests to create distribution
        slowest_tests = data.get('slowest_tests', [])
        for test in slowest_tests:
            exec_time = test.get('execution_time', 0)
            if exec_time < 1:
                bucket_counts[0] += 1
            elif exec_time < 5:
                bucket_counts[1] += 1
            elif exec_time < 10:
                bucket_counts[2] += 1
            elif exec_time < 30:
                bucket_counts[3] += 1
            else:
                bucket_counts[4] += 1
        
        chart_data = {
            'labels': buckets,
            'datasets': [{
                'label': 'Number of Tests',
                'data': bucket_counts,
                'backgroundColor': [
                    '#28a745',
                    '#17a2b8',
                    '#ffc107',
                    '#fd7e14',
                    '#dc3545'
                ]
            }]
        }
        
        return json.dumps(chart_data)
    
    def _generate_recommendations_section(self) -> str:
        """Generate recommendations section."""
        # Get latest report for recommendations
        try:
            latest_report = self.reporter.generate_performance_report(include_trends=False)
            recommendations = latest_report.recommendations
            
            if not recommendations:
                return ""
            
            recommendations_html = """
            <div class="recommendations">
                <h4>💡 Recommendations</h4>
                <ul>
            """
            
            for rec in recommendations:
                recommendations_html += f"<li>{rec}</li>"
            
            recommendations_html += """
                </ul>
            </div>
            """
            
            return recommendations_html
        
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ""
    
    def generate_all_dashboards(self, days_back: int = 30) -> Dict[str, Path]:
        """Generate all dashboard types."""
        dashboards = {}
        
        try:
            dashboards['comprehensive'] = self.generate_dashboard('comprehensive', days_back)
            dashboards['summary'] = self.generate_dashboard('summary', days_back)
            dashboards['trends'] = self.generate_dashboard('trends', days_back)
            
            logger.info(f"Generated {len(dashboards)} performance dashboards")
        
        except Exception as e:
            logger.error(f"Failed to generate dashboards: {e}")
        
        return dashboards