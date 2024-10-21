# Replace arbitrary (nested) dictionary parameters based on a command line string
# 
# From commandline, use e.g. with --supertune "key1.key2=new_value"
#
# m.mieskolainen@imperial.ac.uk, 2024

import pytest
import re
import ast

# ------------------------------------------
from icenet import print
# ------------------------------------------

def set_nested_dict_value(d, keys, value, indices=None):
    """
    Helper function to set a value in a nested dictionary or list and print the old value being replaced.
    """
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    
    if indices is not None:
        target = d[keys[-1]]
        for idx in indices[:-1]:
            target = target[idx]
        old_value = target[indices[-1]]
        if isinstance(old_value, dict):
            old_value = old_value.get(keys[-1], "Key did not exist")
        print(f"Old value at '{keys[-1]}[{','.join(map(str, indices))}]' was: {old_value}")
        if isinstance(target[indices[-1]], dict):
            target[indices[-1]][keys[-1]] = value
        else:
            target[indices[-1]] = value
    else:
        old_value = d.get(keys[-1], "Key did not exist")
        print(f"Old value at '{keys[-1]}' was: {old_value}")
        d[keys[-1]] = value

def parse_value(value_str: str):
    """
    Parse the value string and return the appropriate type.
    """
    value_str = value_str.strip()
    if value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
    elif value_str.lower() == 'none':
        return None
    try:
        value = ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        value = value_str
    return value

def supertune(d, config_string):
    """
    Parse a replacement string and update the dictionary accordingly,
    printing the old values before updating.
    
    Args:
        d:              dictionary to be updated
        config_string:  replacement string
        
    Returns:
        updated dictionary
    """
    
    pattern = re.compile(r'(\S+(?:\[\d+(?:,\s*\d+)*\])?)\s*=\s*([\s\S]+?)(?=\s+\S+(?:\[\d+(?:,\s*\d+)*\])?\s*=|\s*$)')
    
    matches = pattern.findall(config_string)
    
    for match in matches:
        keys_str, value_str = match
        
        # Check if we're dealing with a nested list element
        if '[' in keys_str:
            keys_part, indices_part = keys_str.split('[')
            keys = keys_part.split('.')
            indices = [int(i.strip()) for i in indices_part[:-1].split(',')]
        else:
            keys = keys_str.split('.')
            indices = None
        
        value = parse_value(value_str)
        
        print(f"Replacing '{keys_str}' with a value = {value}")
        set_nested_dict_value(d, keys, value, indices)
    
    return d

@pytest.mark.parametrize("initial, replacement, expected", [

    # Basic test cases
    ({'key1': {'key2': 3.14, 'key3': [1, 2, 3]}, 'key5': 10},
     "key1.key2=6.28 key1.key3 = [4, 5, 6] key5 = 42",
     {'key1': {'key2': 6.28, 'key3': [4, 5, 6]}, 'key5': 42}),
    
    ({'a': {'b': {'c': 1}}},
     "a.b.c = 2 a.b.d = 3",
     {'a': {'b': {'c': 2, 'd': 3}}}),
    
    ({'x': 1, 'y': 2},
     "x = 10 y = [20, 30]",
     {'x': 10, 'y': [20, 30]}),
    
    ({'nested': {'list': [1, 2, 3]}},
     "nested.list = [4, 5, 6]",
     {'nested': {'list': [4, 5, 6]}}),
    
    # String tests
    ({},
     "new.key = 'value' another.new.key = 123",
     {'new': {'key': 'value'}, 'another': {'new': {'key': 123}}}),
    
    ({'nested': {'list': [[1, 2, 3], [4, 5, 6]]}},
     "nested.list = [4, 5, 6]",
     {'nested': {'list': [4, 5, 6]}}),
    
    ({'nested': {'list': [[1, 2, 3], [4, 5, 6]]}},
     "nested.list = [[3,4,5],[0,0,1]]",
     {'nested': {'list': [[3, 4, 5], [0, 0, 1]]}}),
    
    ({'name': 'John'},
     "name = 'Jane Doe'",
     {'name': 'Jane Doe'}),
    
    ({'person': {'name': 'John', 'age': 30}},
     "person.name = 'Jane Doe' person.age = 28",
     {'person': {'name': 'Jane Doe', 'age': 28}}),
    
    ({'quote': ''},
     "quote = 'To be or not to be, that is the question'",
     {'quote': 'To be or not to be, that is the question'}),
    
    ({'nested': {'string': 'old'}},
     "nested.string = 'new string with spaces'",
     {'nested': {'string': 'new string with spaces'}}),
    
    ({'data': {'values': [1, 2, 3], 'label': 'old'}},
     "data.values = ['a', 'b', 'c'] data.label = 'new label'",
     {'data': {'values': ['a', 'b', 'c'], 'label': 'new label'}}),
    
    ({'complex': {'nested': {'string': 'old', 'list': [1, 2]}}},
     "complex.nested.string = 'new' complex.nested.list = ['a', 'b', 'c']",
     {'complex': {'nested': {'string': 'new', 'list': ['a', 'b', 'c']}}}),
    
    # Nested list element updates
    ({'matrix': [[1, 2], [3, 4]]},
     "matrix[0,1] = 5",
     {'matrix': [[1, 5], [3, 4]]}),
    
    ({'data': {'values': [[1.0, 2.0], [3.0, 4.0]]}},
     "data.values[1,0] = 3.5",
     {'data': {'values': [[1.0, 2.0], [3.5, 4.0]]}}),
    
    ({'nested': {'matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}},
     "nested.matrix[0,2] = 10 nested.matrix[2,0] = 20",
     {'nested': {'matrix': [[1, 2, 10], [4, 5, 6], [20, 8, 9]]}}),
    
    ({'complex': {'data': {'matrix': [[1.5, 2.5], [3.5, 4.5]]}}},
     "complex.data.matrix[0,0] = 1.0 complex.data.matrix[1,1] = 5.0",
     {'complex': {'data': {'matrix': [[1.0, 2.5], [3.5, 5.0]]}}}),
    
    ({'mixed': {'list': [1, 2, 3], 'matrix': [[4, 5], [6, 7]]}},
     "mixed.list[1] = 10 mixed.matrix[0,1] = 20",
     {'mixed': {'list': [1, 10, 3], 'matrix': [[4, 20], [6, 7]]}}),
    
    # Boolean test cases
    ({'flags': {'active': False, 'enabled': True}},
     "flags.active = True flags.enabled = False",
     {'flags': {'active': True, 'enabled': False}}),

    ({'config': {'debug': False}},
     "config.debug = true",  # Test lowercase 'true'
     {'config': {'debug': True}}),

    ({'settings': {'verbose': True}},
     "settings.verbose = false",  # Test lowercase 'false'
     {'settings': {'verbose': False}}),

    # None test case
    ({'user': {'name': 'John', 'email': 'john@example.com'}},
     "user.email = None",
     {'user': {'name': 'John', 'email': None}}),

    # Mixed types test case
    ({'mixed': {'num': 10, 'str': 'hello', 'bool': False, 'list': [1, 2, 3]}},
     "mixed.num = 20 mixed.str = 'world' mixed.bool = true mixed.list = [4, 5, 6]",
     {'mixed': {'num': 20, 'str': 'world', 'bool': True, 'list': [4, 5, 6]}}),

    # Nested boolean update
    ({'deep': {'nested': {'flag1': True, 'flag2': False}}},
     "deep.nested.flag1 = false deep.nested.flag2 = true",
     {'deep': {'nested': {'flag1': False, 'flag2': True}}}),

    # Boolean in list
    ({'data': [True, False, True]},
     "data[1] = true",
     {'data': [True, True, True]}),

    # None in nested structure
    ({'user': {'profile': {'name': 'Alice', 'age': None}}},
     "user.profile.age = 30",
     {'user': {'profile': {'name': 'Alice', 'age': 30}}}),

    # Edge case: empty string
    ({'config': {'api_key': 'abc123'}},
     "config.api_key = ''",
     {'config': {'api_key': ''}}),

    # Edge case: space in string
    ({'message': {'greeting': 'Hello'}},
     "message.greeting = 'Hello World'",
     {'message': {'greeting': 'Hello World'}}),

    # Edge case: very large integer
    ({'stats': {'max_value': 1000}},
     "stats.max_value = 9999999999999999",
     {'stats': {'max_value': 9999999999999999}}),

    # Edge case: scientific notation
    ({'science': {'avogadro': 6.022e23}},
     "science.avogadro = 6.0221409e+23",
     {'science': {'avogadro': 6.0221409e+23}}),

    # Edge case: Unicode string
    ({'text': {'welcome': 'Hello'}},
     "text.welcome = '你好'",
     {'text': {'welcome': '你好'}}),

    # Combination of different types in nested structure
    ({'complex': {
        'num': 42,
        'str': 'test',
        'bool': True,
        'list': [1, 'two', False],
        'dict': {'a': 1, 'b': '2'}
    }},
     "complex.num = 43 complex.str = 'changed' complex.bool = false complex.list[1] = 2 complex.dict.b = true",
     {'complex': {
         'num': 43,
         'str': 'changed',
         'bool': False,
         'list': [1, 2, False],
         'dict': {'a': 1, 'b': True}
     }}),

    # Edge case: empty dictionary
    ({},
     "new_key = 'new_value'",
     {'new_key': 'new_value'}),

    # Edge case: deeply nested dictionaries
    ({'a': {'b': {'c': {'d': {'e': {'f': 1}}}}}},
     "a.b.c.d.e.f = 2",
     {'a': {'b': {'c': {'d': {'e': {'f': 2}}}}}}),

    # Edge case: dictionary with boolean values
    ({'flag1': True, 'flag2': False},
     "flag1 = false flag2 = true",
     {'flag1': False, 'flag2': True}),

    # Edge case: dictionary with None value
    ({'key1': None},
     "key1 = 'value'",
     {'key1': 'value'}),

    # Nested dictionary with a list of dictionaries (this does not work atm)
    #({'data': {'list': [{'key1': 1}, {'key2': 2}]}},
    # "data.list[0].key1 = 10 data.list[1].key2 = 20",
    # {'data': {'list': [{'key1': 10}, {'key2': 20}]}}),

    # Multiple replacements with mixed types
    ({'a': 1, 'b': 'text', 'c': [1, 2, 3]},
     "a = 2 b = 'new_text' c = [4, 5, 6]",
     {'a': 2, 'b': 'new_text', 'c': [4, 5, 6]}),

    # Replacement involving float values
    ({'x': 1.1, 'y': 2.2},
     "x = 3.3 y = 4.4",
     {'x': 3.3, 'y': 4.4}),

    # Update a list of booleans
    ({'list': [True, False, True]},
     "list[1] = true",
     {'list': [True, True, True]}),

    # Nested dictionary with None value update
    ({'user': {'profile': {'name': 'Alice', 'age': None}}},
     "user.profile.age = 30 user.profile.name = 'Bob'",
     {'user': {'profile': {'name': 'Bob', 'age': 30}}}),

    # Update involving scientific notation
    ({'constants': {'pi': 3.14, 'e': 2.71}},
     "constants.pi = 3.14159 constants.e = 2.71828",
     {'constants': {'pi': 3.14159, 'e': 2.71828}}),

    # Edge case: update with a negative number
    ({'count': 5},
     "count = -10",
     {'count': -10}),

    # Edge case: update with an empty list
    ({'items': [1, 2, 3]},
     "items = []",
     {'items': []}),

    # Update a nested list within a dictionary
    ({'matrix': {'data': [[1, 2], [3, 4]]}},
     "matrix.data[1,1] = 5",
     {'matrix': {'data': [[1, 2], [3, 5]]}}),

    # Edge case: updating to None value
    ({'key': 'value'},
     "key = None",
     {'key': None}),

    # Edge case: multiple spaces in replacement string
    ({'a': 1, 'b': 2},
     "a = 10   b = 20",
     {'a': 10, 'b': 20}),
])

def test_parse_and_replace(initial, replacement, expected):
    result = supertune(initial.copy(), replacement)
    assert result == expected, f"Expected {expected}, but got {result}"
