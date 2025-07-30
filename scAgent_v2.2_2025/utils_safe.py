#!/usr/bin/env python3
"""
Safe string utility functions for scAgent to handle None values
"""

from typing import Any, Optional, Union

def safe_lower(value: Any) -> str:
    """Safely convert value to lowercase string, handling None values."""
    if value is None:
        return ''
    return str(value).lower()

def safe_str(value: Any) -> str:
    """Safely convert value to string, handling None values."""
    if value is None:
        return ''
    return str(value)

def safe_get_field(record: dict, field: str, default: str = '') -> str:
    """Safely get field from record, ensuring string return."""
    value = record.get(field, default)
    if value is None:
        return default
    return str(value)

def safe_get_field_lower(record: dict, field: str, default: str = '') -> str:
    """Safely get field from record and convert to lowercase."""
    return safe_lower(safe_get_field(record, field, default))

def safe_split_words(text: Any) -> set:
    """Safely split text into words, handling None values."""
    if text is None:
        return set()
    return set(str(text).lower().split())

def safe_contains(haystack: Any, needle: str) -> bool:
    """Safely check if haystack contains needle, handling None values."""
    if haystack is None:
        return False
    return needle.lower() in str(haystack).lower()

def safe_startswith(text: Any, prefix: str) -> bool:
    """Safely check if text starts with prefix, handling None values."""
    if text is None:
        return False
    return str(text).lower().startswith(prefix.lower())

def safe_endswith(text: Any, suffix: str) -> bool:
    """Safely check if text ends with suffix, handling None values."""
    if text is None:
        return False
    return str(text).lower().endswith(suffix.lower())

def safe_join_fields(record: dict, fields: list, separator: str = ' ') -> str:
    """Safely join multiple fields from record."""
    values = []
    for field in fields:
        value = safe_get_field(record, field)
        if value:
            values.append(value)
    return separator.join(values)

def safe_multi_field_lower(record: dict, fields: list) -> str:
    """Get multiple fields and return as lowercase concatenated string."""
    return safe_lower(safe_join_fields(record, fields))

# Patch function to fix existing utils.py
def patch_utils_safe_strings():
    """Apply safe string handling to existing utils functions."""
    import scAgent.utils as utils_module
    
    # Replace unsafe .lower() calls with safe versions
    original_functions = {}
    
    def make_safe_wrapper(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AttributeError as e:
                if "'NoneType' object has no attribute 'lower'" in str(e):
                    # Handle the specific error by returning safe defaults
                    return {'decision': 'reject', 'confidence': 0.0, 'reasons': ['Data processing error']}
                raise
        return wrapper
    
    # Apply wrapper to assessment functions
    for attr_name in dir(utils_module):
        if 'assess' in attr_name.lower() and callable(getattr(utils_module, attr_name)):
            original_func = getattr(utils_module, attr_name)
            safe_func = make_safe_wrapper(original_func)
            setattr(utils_module, attr_name, safe_func)
    
    return original_functions

if __name__ == "__main__":
    # Test safe functions
    test_record = {'title': None, 'summary': 'Test summary', 'organism': None}
    
    print("Testing safe string functions:")
    print(f"safe_lower(None): '{safe_lower(None)}'")
    print(f"safe_get_field_lower(test_record, 'title'): '{safe_get_field_lower(test_record, 'title')}'")
    print(f"safe_get_field_lower(test_record, 'summary'): '{safe_get_field_lower(test_record, 'summary')}'")
    print(f"safe_contains(None, 'test'): {safe_contains(None, 'test')}")
    print(f"safe_split_words(None): {safe_split_words(None)}") 