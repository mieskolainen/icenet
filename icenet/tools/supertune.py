# Replace arbitrary (nested) dictionary parameters based on a command line string
# 
# From commandline, use e.g. with --supertune "key1.key2=new_value"
#
# m.mieskolainen@imperial.ac.uk, 2024

import re

def parse_config_string(config_string):
    config_dict = {}
    pairs = re.findall(r'(\S+?\s*=\s*\[.*?\]|\S+?\s*=\s*\S+)', config_string)
    for pair in pairs:
        try:
            keys, value = pair.split('=', 1)
            keys = keys.strip().split('.')
            value = value.strip()
            current_dict = config_dict
            for key in keys[:-1]:
                current_dict = current_dict.setdefault(key.strip(), {})
            current_dict[keys[-1].strip()] = convert_value(value)
        except ValueError:
            print(f"supertune: Error parsing pair '{pair}'. Expected format 'key1.key2=value'. Skipping this pair.")
    return config_dict

def convert_value(value):
    # Check if the value represents a list
    if re.match(r'^\[.*\]$', value.strip()):
        value = value.strip('[]').strip()
        return [convert_value(v.strip()) for v in value.split(',')]
    
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            return value

def supertune(cfg: dict, config_string: str):
    """
    Replace arbitrary (nested) dictionary parameters
    based on a command line string
    
    Args:
        cfg:            config dictionary
        config_string:  config string
    """
    def recursive_update(cfg_node, data_node, path=""):
        for key, value in cfg_node.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                if key not in data_node:
                    data_node[key] = {}
                recursive_update(value, data_node[key], current_path)
            else:
                if key in data_node:
                    old_value = data_node.get(key, None)
                    data_node[key] = value
                    print(f"supertune: Replaced '{current_path}' from '{old_value}' to '{value}'")
                else:
                    data_node[key] = value
                    print(f"supertune: Added new key '{current_path}' with value '{value}'")

    config_dict = parse_config_string(config_string)
    recursive_update(config_dict, cfg)

# Test function using pytest
def test_config():
    
    import pytest

    cfg = {
        "NN": {
            "optim": {
                "lr": 0.001
            },
            "model": {
                "layers": [64, 64]
            }
        },
        "XGB": {
            "model": {
                "trees": 100
            }
        }
    }

    # Arbitrary test case. Keep (extra) white spaces here to check robustness
    config_string = "NN.optim.lr = 1.0e-4  NN.model.layers=[128, 256,   512] XGB.model.trees=500"
    supertune(cfg, config_string)
    
    expected_cfg = {
        "NN": {
            "optim": {
                "lr": 0.0001
            },
            "model": {
                "layers": [128, 256, 512]
            }
        },
        "XGB": {
            "model": {
                "trees": 500
            }
        }
    }

    assert cfg == expected_cfg, f"Expected {expected_cfg} but got {cfg}"
    print(cfg)
