import os
import yaml


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)