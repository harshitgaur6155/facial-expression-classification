import os
from pathlib import Path
import logging
import yaml


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# project_name = 'facial-expression-classify'
project_name = 'facialExpressionClassify'

list_of_files = [
    '.github/workflows/.gitkeep',
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/configuration.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/constants/__init__.py',
    f'src/{project_name}/exception/__init__.py',
    f'templates/index.html',
    f'static/.placeholder',
    'app.py',
    'main.py',
    'config/config.yaml',
    'dvc.yaml',
    'params.yaml',
    'requirements.txt',
    'setup.py',
    'research/trials.ipynb'
]



# Default YAML content to add to yaml files
yaml_content = {
    "key1": "val1"
}

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as wr:
            logging.info(f"Creating empty file: {filepath}")
            
            # If the file is yaml, write YAML content into it
            if filepath in (Path('config/config.yaml'), Path('params.yaml'), Path('dvc.yaml')):
                yaml.dump(yaml_content, wr, default_flow_style=False)
                logging.info(f"Added YAML content to {filepath}")
    else:
        logging.info(f"{filename} already exists")