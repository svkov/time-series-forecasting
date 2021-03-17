from collections import Iterable
from typing import Dict

import yaml


class Config:

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.data: Dict = self.read_yaml()

    def read_yaml(self):
        with open(self.path, encoding='utf-8') as file:
            data = yaml.load(file)
        return data

    def get(self, param, default):
        try:
            data = self.data[self.name]
        except:
            raise KeyError(f'В workflow/labels.yaml нет ключа {self.name}')
        try:
            return data.get(param, default)
        except:
            raise KeyError(f'В workflow/labels.yaml в {self.name} нет параметра {param}')
