from surfquakecore.data_processing.validators.constants import ALGEBRA_FUNCTIONS, ALGEBRA_CONSTANTS


def require_keys(config: object, required_keys: object) -> object:
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

def require_type(config, key, expected_type):
    if not isinstance(config[key], expected_type):
        raise ValueError(
            f"Key '{key}' must be of type {expected_type.__name__}, got {type(config[key]).__name__}"
        )

def optional_type(config, key, expected_types):
    if key in config and not isinstance(config[key], expected_types):
        type_names = (
            expected_types.__name__
            if isinstance(expected_types, type)
            else ', '.join(t.__name__ for t in expected_types)
        )
        raise ValueError(
            f"Optional key '{key}' must be of type {type_names}, got {type(config[key]).__name__}"

        )

def _validate_equation(equation: str) -> None:
    """
    Validate that the equation string:
      - Is a non-empty string
      - Contains only safe characters (no shell/code injection)
      - Does not reference forbidden Python builtins
      - Only calls functions from ALGEBRA_FUNCTIONS and constants from ALGEBRA_CONSTANTS
      - Treats everything else as a trace reference (tr1, tr2, Z, N, E, ch1, ...)
    """
    import re
    _ALLOWED_TOKENS = set(ALGEBRA_FUNCTIONS) | set(ALGEBRA_CONSTANTS)

    _SAFE_EQUATION_RE = re.compile(r'^[a-zA-Z0-9_\s\+\-\*\/\(\)\.\,\^\%]+$')

    _BLOCKED_BUILTINS = {
        'exec', 'eval', 'import', 'open', 'compile', 'globals',
        'locals', 'getattr', 'setattr', 'delattr', '__import__',
        'breakpoint', 'input', 'print', 'vars', 'dir',
    }

    if not isinstance(equation, str) or not equation.strip():
        raise ValueError("'equation' must be a non-empty string.")

    if not _SAFE_EQUATION_RE.match(equation):
        raise ValueError(
            f"'equation' contains invalid characters: '{equation}'"
        )

    identifiers = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', equation))
    unknown = identifiers - _ALLOWED_TOKENS
    dangerous = unknown & _BLOCKED_BUILTINS

    if dangerous:
        raise ValueError(
            f"'equation' references forbidden names: {sorted(dangerous)}"
        )
