# Logging in PyART

PyART has been updated to use Python's built-in `logging` module instead of `print()` statements. This provides better control over output verbosity and formatting.

## Quick Start

To enable logging output in your scripts:

```python
from PyART.logging_config import setup_logging

# Setup logging with default INFO level
setup_logging()

# Now all PyART modules will log to console
```

## Customizing Log Level

You can customize the logging level:

```python
from PyART.logging_config import setup_logging

# Show all debug messages
setup_logging(level='DEBUG')

# Show only warnings and errors
setup_logging(level='WARNING')
```

## Available Log Levels

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems
- `INFO`: Confirmation that things are working as expected (default)
- `WARNING`: An indication that something unexpected happened
- `ERROR`: A serious problem
- `CRITICAL`: A very serious error

## Custom Format

You can also customize the log format:

```python
setup_logging(
    level='INFO',
    format_string='%(levelname)s - %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
```

## Without Configuration

If you don't call `setup_logging()`, Python's logging will still work but no messages will be displayed by default (unless you configure logging yourself).

## Migration from print()

Most `print()` statements in PyART have been replaced with:
- `logging.info()` - for informational messages
- `logging.warning()` - for warnings
- `logging.error()` - for errors
- `logging.debug()` - for debug information

## Example

```python
from PyART.logging_config import setup_logging
from PyART.catalogs.sxs import Waveform_SXS

# Enable logging
setup_logging(level='INFO')

# Now you'll see informational messages
wave = Waveform_SXS(path='./data/', ID='0001')
```
