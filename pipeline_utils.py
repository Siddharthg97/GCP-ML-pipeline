import argparse
from pathlib import Path
import os
from google.cloud import storage
import sys
import yaml

class PipelineUtils:
    def __init__(self, storage_path, file_name):
        self.storage_path = storage_path
        self.file_name = file_name
        
    def store_pipeline(self) -> None:
        """Uploads a file to GCS bucket.
        Args: 
            no arguments
        Returns:
            None
        """
        print(self.file_name)
        blob = storage.blob.Blob.from_string(self.storage_path, client=storage.Client())
        blob.upload_from_filename(self.file_name)
        print(
            "contents {} uploaded to {}.".format(
                self.file_name, self.storage_path
            )
        )

class MarkdownArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
    def get_args(self):
        """Get arguments for python script
        Args:
            no arguments
        Returns:
            args object of having commit-SHA hash ID, branch name, and is_prod of Github.
        """
        # Import arguments to local variables
        self.parser.add_argument('--COMMIT_ID', required=True, type=str)
        self.parser.add_argument('--BRANCH', required=True, type=str)
        self.parser.add_argument("--is_prod", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
        
        args = self.parser.parse_args()
        return args

class YamlImport:
    def __init__(self, file_name):
        self.file_name = file_name
        
    def yaml_import(self):
        # Import yaml to local variables
        file_path = Path().resolve().parent.parent
        file_path = os.path.join(file_path, self.file_name)
        file_path_ = Path().resolve().parent
        file_path_ = os.path.join(file_path_, self.file_name) 

        try:
            with open(file_path, 'r') as stream:
                try:
                    dict_ = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print("Unable to load yaml")
        except:
            with open(file_path_, "r") as stream:
                try:
                    dict_ = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print("Unable to load yaml")
            
        return(dict_)
    
    