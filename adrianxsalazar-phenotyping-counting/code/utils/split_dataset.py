import numpy as np
import argparse
import json
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-i',
        metavar='dataset_json',
        required=True,
        help='The dataset JSON');
    
    parser.add_argument('-t',
        metavar='split_t',
        required=True,
        default=0.7,
        type=float,
        help='The split for training data');
    
    parser.add_argument('-v',
        metavar='split_v',
        required=True,
        default=0.1,
        type=float,
        help='The split for validation data');
    
    parser.add_argument('-test',
        metavar='split_test',
        required=True,
        default=0.2,
        type=float,
        help='The split for testing data');

    args = parser.parse_args()

    json_file = open(args.i, "r");
    dataset = json.load(json_file);
    json_file.close();

    split = [args.test, args.t, args.v];
    names = ["json_test_set.json", "json_train_set.json", "json_val_set.json"];

    if (not math.isclose(np.sum(split), 1.0)): raise RuntimeError(f"Testing, training and validation splits must sum to 1.0 (they sum to {np.sum(split)}, {split})");

    images   = dataset["images"];
    imgcount = len(images);
    np.random.shuffle(images);
    print(f"Splitting the {imgcount} shuffled images where {names} {split}");

    last_img = 0;
    for i, name in enumerate(names):
        count=math.ceil(split[i] * imgcount);

        with open(name, 'w') as out:
            simages = [img["file_name"] for img in images[ last_img:(last_img+(count))]];

            json.dump(
                simages,
                out
            );
            
            out.write('\n');
            print(f"Saved images (exclusive larger bound) {last_img}:{(last_img+(count))} (count of {len(simages)}) to {name}");
        
        last_img += count;



