"""
File: sc_data_generator.py
Author: Jeff Martin
Date: 12/28/23

This file generates SuperCollider data structures from Python data structures,
to make generating SuperCollider data files easy.
"""

import re

class Array:
    def __init__(self, name):
        """
        Creates an Array
        :param name: The name of the array (to be used with SuperCollider)
        """
        self.array = []
        self.name = name

    def make_arr_sc(self):
        """
        Turns the array into a SuperCollider array string
        :return: The SuperCollider array string
        """
        string = f"~{self.name} = [\n"
        for i, val in enumerate(self.array):
            if i < len(self.array) - 1:
                if type(val) == str:
                    string += f"    \"{val}\",\n"
                elif type(val) == int or type(val) == float:
                    string += f"    {val},\n"
            elif i < len(self.array) - 1:
                if type(val) == str:
                    string += f"    \"{val}\"\n"
                elif type(val) == int or type(val) == float:
                    string += f"    {val}\n"
        return string + "];\n"
    

class Dictionary:
    def __init__(self, name):
        """
        Creates a Dictionary
        :param name: The name of the dictionary (to be used with SuperCollider)
        """
        self.dict = {}
        self.name = name

    def make_dict_sc(self):
        """
        Turns the dictionary into a SuperCollider dictionary string
        :return: The SuperCollider dictionary string
        """
        string = f"~{self.name} = Dictionary.new;\n"
        for key, val in self.dict.items():
            if type(val) == str:
                string += f"~{self.name}.add(\{key} -> \"{val}\");\n"
            elif type(val) == int or type(val) == float:
                string += f"~{self.name}.add(\{key} -> {val});\n"
        return string


class List:
    def __init__(self, name):
        """
        Creates a List
        :param name: The name of the list (to be used with SuperCollider)
        """
        self.list = []
        self.name = name

    def make_list_sc(self):
        """
        Turns the list into a SuperCollider list string
        :return: The SuperCollider list string
        """
        string = f"~{self.name} = List[\n"
        for i, val in enumerate(self.list):
            if i < len(self.list) - 1:
                if type(val) == str:
                    string += f"    \"{val}\",\n"
                elif type(val) == int or type(val) == float:
                    string += f"    {val},\n"
            elif i < len(self.list) - 1:
                if type(val) == str:
                    string += f"    \"{val}\"\n"
                elif type(val) == int or type(val) == float:
                    string += f"    {val}\n"
        return string + "];\n"

    
def make_sc_from_nested_objects(data, level=0):
    """
    Makes SC data structures from nested Python data structures, recursively
    :param data: The data structure to turn into SuperCollider format
    :param level: The level of indentation (handled automatically)
    :return: A string with the SuperCollider code
    """
    content = ""

    if len(data) == 0:
        content = " " * (level * 4)
        if type(data) == list:    
            content += "List.new"
        elif type(data) == tuple:
            content += "Array.new"
        elif type(data) == dict:
            content += "Dictionary.new"
        if level == 0:
            content += ';\n'
    
    else:
        if type(data) == list:
            content += "List[\n" + " " * ((level + 1) * 4)
            for item in data:
                if type(item) == list or type(item) == dict or type(item) == tuple:
                    content += make_sc_from_nested_objects(item, level + 1) + ', '
                elif type(item) == str:
                    content += '\"' + re.sub(r'\\', '/', item) + '\", '
                else:
                    content += f"{item}, "
            content += "\n" + " " * (level * 4) + "]"
            if level == 0:
                content += ';\n'
        if type(data) == tuple:
            content += "["
            for item in data:
                if type(item) == list or type(item) == dict or type(item) == tuple:
                    content += make_sc_from_nested_objects(item, level + 1) + ', '
                elif type(item) == str:
                    content += '\"' + re.sub(r'\\', '/', item) + '\", '
                else:
                    content += f"{item}, "
            content += "]"
            if level == 0:
                content += ';\n'
        elif type(data) == dict:
            content += "Dictionary.newFrom([\n" + " " * ((level + 1) * 4)
            for key, item in data.items():
                content += f"\"{key}\", "
                if type(item) == list or type(item) == dict or type(item) == tuple:
                    content += make_sc_from_nested_objects(item, level + 1) + ', '
                elif type(item) == str and key != "buffer":
                    content += '\"' + re.sub(r'\\', '/', item) + '\", '
                else:
                    content += f"{item}, "
            content += "\n" + " " * (level * 4) + "])"
            if level == 0:
                content += ';'
    
    return content
