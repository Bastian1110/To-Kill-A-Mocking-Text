"""
By : Harimi Manzano (@HarumiManz)
Project : To-Kill-A-Mocking-Bird
"""

import os
import re

def clean_content(content):
    """
    Cleans the content of a file by removing special characters and excessive spaces.
    
    Args:
    content (str): Content to be cleaned.
    
    Returns:
    str: Cleaned content with no more than one space between words.
    """
    content = content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    content = re.sub(' +', ' ', content)
    return content.strip()

def read_directory_files(directory):
    """
    Reads all text files in the specified directory, cleans their content, and
    collects them into a list.
    
    Args:
    directory (str): The path to the directory containing text files.
    
    Returns:
    list[str, str]: A list of all the strings containing the cleaned contents of all files.
    """
    combined_content = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    cleaned_content = clean_content(content)
                    combined_content.append([filename, cleaned_content])
            except FileNotFoundError:
                print(f"The file {filename} was not found.")
            except Exception as e:
                print(f"An error occurred while reading {filename}: {e}")
    return combined_content

def build_validation_dataset(directory):
    """
    Constructs a validation dataset from a specified directory containing text files.

    This function reads text files within the given directory, processes them to clean the content, 
    and organizes them into a dataset. It assumes that files prefixed with 'org' are originals (label 0)
    and all others are considered fakes (label 1).

    Parameters:
    - directory (str): The path to the directory where text files are stored.

    Returns:
    - contents (list of str): A list of cleaned contents of the text files.
    - y (list of int): A list of labels corresponding to the text files where '0' denotes an original
      document and '1' denotes a fake document.

    The function handles FileNotFoundError if a file is missing, and a general Exception for other
    issues that may arise during file reading, logging the specific error to the console.

    Example usage:
    >>> directory_path = 'path/to/text/files'
    >>> contents, labels = build_validation_dataset(directory_path)
    >>> print(f'Loaded {len(contents)} documents.')
    """
    contents = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    cleaned_content = clean_content(content)  # Assumes existence of a cleaning function
                    contents.append(cleaned_content)
                    y.append(0 if filename[:3] == "org" else 1)
            except FileNotFoundError:
                print(f"The file {filename} was not found.")
            except Exception as e:
                print(f"An error occurred while reading {filename}: {e}")
    return contents, y
