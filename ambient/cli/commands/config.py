"""
Configuration command for managing system settings.
"""

import click
from pathlib import Path
import yaml


@click.command(name='config')
@click.argument('action', type=click.Choice(['show', 'set', 'get', 'validate']))
@click.argument('key', required=False)
@click.argument('value', required=False)
@click.option('--file', '-f', type=click.Path(), help='Configuration file path')
@click.pass_context
def config_cmd(ctx, action, key, value, file):
    """
    Manage system configuration.
    
    Actions:
        show     - Display current configuration
        set      - Set a configuration value
        get      - Get a configuration value
        validate - Validate configuration
    
    Examples:
    
        # Show current configuration
        alexpose config show
        
        # Get a specific value
        alexpose config get pose_estimators.mediapipe.model_complexity
        
        # Set a configuration value
        alexpose config set default_frame_rate 60.0
        
        # Validate configuration
        alexpose config validate
    """
    config_manager = ctx.obj['config']
    logger = ctx.obj['logger']
    
    try:
        if action == 'show':
            _show_config(config_manager, file)
        elif action == 'get':
            if not key:
                raise click.ClickException("Key is required for 'get' action")
            _get_config_value(config_manager, key)
        elif action == 'set':
            if not key or not value:
                raise click.ClickException("Key and value are required for 'set' action")
            _set_config_value(config_manager, key, value, file)
        elif action == 'validate':
            _validate_config(config_manager)
        
    except Exception as e:
        logger.error(f"Configuration command failed: {str(e)}")
        raise click.ClickException(str(e))


def _show_config(config_manager, file):
    """Display current configuration."""
    if file:
        config_path = Path(file)
        if not config_path.exists():
            raise click.ClickException(f"Configuration file not found: {file}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = config_manager.config.__dict__
    
    click.echo("Current Configuration:")
    click.echo("=" * 60)
    click.echo(yaml.dump(config_data, default_flow_style=False, indent=2))


def _get_config_value(config_manager, key):
    """Get a specific configuration value."""
    keys = key.split('.')
    value = config_manager.config
    
    try:
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise AttributeError(f"Key not found: {k}")
        
        click.echo(f"{key}: {value}")
        
    except AttributeError as e:
        raise click.ClickException(f"Configuration key not found: {key}")


def _set_config_value(config_manager, key, value, file):
    """Set a configuration value."""
    if not file:
        file = config_manager.config_dir / 'alexpose.yaml'
    
    config_path = Path(file)
    
    # Load existing configuration
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}
    
    # Set value
    keys = key.split('.')
    current = config_data
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Try to convert value to appropriate type
    try:
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.replace('.', '').replace('-', '').isdigit():
            value = float(value) if '.' in value else int(value)
    except:
        pass
    
    current[keys[-1]] = value
    
    # Save configuration
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    click.echo(f"Configuration updated: {key} = {value}")
    click.echo(f"Saved to: {config_path}")


def _validate_config(config_manager):
    """Validate configuration."""
    click.echo("Validating configuration...")
    
    is_valid = config_manager.validate_configuration()
    
    if is_valid:
        click.echo("✓ Configuration is valid")
    else:
        click.echo("✗ Configuration validation failed")
        click.echo("Check logs for details")
