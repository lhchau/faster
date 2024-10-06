"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
import yaml
from ast import literal_eval

def exec_configurator():
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            yaml_config_file = arg
            with open(yaml_config_file, "r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.Loader)
            print(f"Overriding config with {yaml_config_file}:")
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            idx = arg.find('=')
            key, val = arg[:idx], arg[idx+1:]
            key = key[2:]
            parent_key, child_key = key.split('.')
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            # assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            cfg[parent_key][child_key] = attempt
    return cfg