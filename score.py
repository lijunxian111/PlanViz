import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a simple argparse example")
    parser.add_argument('--input_path', type=str, help="path to the evaluation json file", default="")
    args = parser.parse_args()
    with open(args.input_path, 'r') as f:
        data = json.load(f)

    score = 0.
    cnt = 0.
    for line in data:
        if isinstance(line['evaluation_out'], list):
            line['evaluation_out'] = line['evaluation_out'][0]
        try:
            if 'Correctness' in line['evaluation_out']:
                if 'correctness_score_points' in line:
                    score += int(line['evaluation_out'][-1]) / len(line['correctness_score_points'])
                elif 'correctness_key_points' in line:
                    score += int(line['evaluation_out'][-1]) / len(line['correctness_key_points'])
            else:
                score += int(line['evaluation_out'][-1]) / 5
            cnt += 1
        except:
            score += 0.0

    score = score / cnt
    print(score)
