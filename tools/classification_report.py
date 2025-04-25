import json
import argparse
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    classification_report
)
from tqdm import tqdm


def evaluate_prompt_response_scores(file_path):
    """
    Evaluates the binary classification performance based on the structure of prompt_score and response_score fields 
    in a JSONL file.
    """
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)

    y_true = []
    y_predict = []

    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Processing lines"):
            data = json.loads(line)
            question_score = data['prompt_score'].split('\n')
            response_score = data['response_score'].split('\n')

            y_true.append(1 if len(question_score) == 2 else 0)

            if len(question_score) == 1 and len(response_score) == 1:
                y_predict.append(0)
            elif (len(question_score) == 1 and len(response_score) == 2) or \
                 (len(question_score) == 2 and len(response_score) == 1):
                y_predict.append(1)
            else:
                y_predict.append(0)

    results = {
        "precision": precision_score(y_true, y_predict),
        "recall": recall_score(y_true, y_predict),
        "f1_score": f1_score(y_true, y_predict),
        "accuracy": accuracy_score(y_true, y_predict),
        "classification_report": classification_report(y_true, y_predict)
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running classification report...")
    parser.add_argument('--filepath', type=str, help='Path to llama-3 result.')
    args = parser.parse_args()

    results = evaluate_prompt_response_scores(args.filepath)
    print('\n')
    print('F1_SCORE: ', results['f1_score'])
    print('ACCURACY: ', results['accuracy'])
    print('\n')