import argparse

import mteb
import yaml


def load_tasks_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['tasks']

def download_datasets(tasks):
    for task_name in tasks:
        try:
            print(f"Downloading dataset: {task_name}")
            
            mteb_tasks = mteb.get_tasks(tasks=[task_name])
            evaluation = mteb.MTEB(tasks=mteb_tasks)
            evaluation.load_tasks_data()
            
            print(f"Dataset {task_name} downloaded successfully")
        except Exception as e:
            print(f"Error downloading {task_name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Download MTEB datasets')
    parser.add_argument('--config', type=str, 
                      required=True,
                      help='Path to config file containing tasks list')
    args = parser.parse_args()
    
    tasks = load_tasks_from_config(args.config)
    
    print(f"Total tasks to download: {len(tasks)}")
    download_datasets(tasks)

if __name__ == "__main__":
    main()