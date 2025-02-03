import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.experiments.classification import run_classification_experiment
import json

def main():
    print("Loading config...")
    with open('config/classification.json') as f:
        config = json.load(f)
    
    print("Starting classification experiment...")
    try:
        results = run_classification_experiment(config)
        print('Results:', results)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main()
