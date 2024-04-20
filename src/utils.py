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