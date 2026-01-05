#!/usr/bin/env python3
"""
Configuration validation script for AlexPose.

This script validates the current configuration and provides detailed
error reporting and recommendations.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambient.core.config import ConfigurationManager


def main():
    """Main validation function."""
    print("🔧 AlexPose Configuration Validation")
    print("=" * 50)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigurationManager()
        
        print(f"Environment: {config_manager.environment}")
        print(f"Config Directory: {config_manager.config_dir}")
        print()
        
        # Run validation
        print("Running comprehensive validation...")
        is_valid = config_manager.validate_configuration()
        
        # Get detailed report
        report = config_manager.get_validation_report()
        
        print()
        print("📊 Validation Results:")
        print(f"  • Valid: {'✅ Yes' if report['valid'] else '❌ No'}")
        print(f"  • Errors: {report['error_count']}")
        print(f"  • Warnings: {report['warning_count']}")
        
        if report['errors']:
            print()
            print("❌ Errors:")
            for i, error in enumerate(report['errors'], 1):
                print(f"  {i}. {error}")
        
        if report['warnings']:
            print()
            print("⚠️  Warnings:")
            for i, warning in enumerate(report['warnings'], 1):
                print(f"  {i}. {warning}")
        
        # Show recommendations
        recommendations = config_manager.get_configuration_recommendations()
        if any(recommendations.values()):
            print()
            print("💡 Recommendations:")
            for category, recs in recommendations.items():
                if recs:
                    print(f"  {category.title()}:")
                    for rec in recs:
                        print(f"    • {rec}")
        
        # Show configuration summary
        print()
        print("📋 Configuration Summary:")
        print("-" * 30)
        summary = config_manager.generate_configuration_summary()
        print(summary)
        
        # Show environment info
        print()
        print("🌍 Environment Information:")
        print("-" * 30)
        env_info = config_manager.get_environment_info()
        
        print(f"Environment: {env_info['environment']}")
        print(f"Config Directory: {env_info['config_directory']}")
        print(f"Main Config Exists: {'✅' if env_info['config_files']['main_exists'] else '❌'}")
        print(f"Environment Config Exists: {'✅' if env_info['config_files']['environment_exists'] else '❌'}")
        
        print()
        print("API Keys Configured:")
        for provider, configured in env_info['api_keys_configured'].items():
            print(f"  • {provider.title()}: {'✅' if configured else '❌'}")
        
        print()
        print("Enabled Features:")
        for feature, enabled in env_info['enabled_features'].items():
            print(f"  • {feature.replace('_', ' ').title()}: {'✅' if enabled else '❌'}")
        
        print()
        print("Pose Estimators:")
        for estimator, enabled in env_info['pose_estimators'].items():
            print(f"  • {estimator}: {'✅' if enabled else '❌'}")
        
        # Exit with appropriate code
        sys.exit(0 if is_valid else 1)
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()