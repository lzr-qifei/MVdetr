import os
import sys
import yaml

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
    def log_yaml(self, yaml_path):
        """Log the contents of a YAML file to the console and the log file."""
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                yaml_str = yaml.dump(yaml_content, default_flow_style=False)
                self.write("\nYAML content:\n")
                self.write(yaml_str + "\n")
        else:
            self.write(f"\nYAML file not found: {yaml_path}\n")
