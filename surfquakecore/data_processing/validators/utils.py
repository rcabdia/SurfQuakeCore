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