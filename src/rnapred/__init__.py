import json
import os
import random
import numpy as np
from datetime import datetime
import pandas as pd
import shutil
import pickle
import torch as tr

from torch.utils.data import DataLoader
from .data import Data, pad_batch
from .model import rnapred
from .utils import write_ct, validate_file, ct_to_dot
from .rnapred_parser import parser
from .utils import dot2png, ct2svg


nu_dict = {
            "R": ["G","A"],
            "Y": ["C","U"],
            "K": ["G","U"],
            "M": ["A","C"],
            "S": ["G","C"],
            "W": ["A","U"],
            "B": ["G","U","C"],
            "D": ["G","A","U"],
            "H": ["A","C","U"],
            "V": ["G","C","A"],
            "N": ["A","G","C","U"]
        }


def train(training_data_file, configuration={}, results_directory=None, validation_data_file=None, worker_count=2, display_output=True):
    if results_directory is None:
        results_directory = f"results_{str(datetime.today()).replace(' ', '-')}/"
    else:
        results_directory = results_directory

    if display_output:
        print(f"Results will be saved in: {results_directory}")

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    if "cache_directory" not in configuration:
        configuration["cache_directory"] = "cache/"

    if validation_data_file is not None:
        training_data_file = training_data_file
        validation_data_file = validation_data_file

    else:
        data = pd.read_csv(training_data_file)
        validation_data_fraction = configuration["validation_data_fraction"] if "validation_data_fraction" in configuration else 0.1
        training_data_file = os.path.join(results_directory, "train.csv")
        validation_data_file = os.path.join(results_directory, "valid.csv")

        validation_data = data.sample(frac=validation_data_fraction)
        validation_data.to_csv(validation_data_file, index=False)
        data.drop(validation_data.index).to_csv(training_data_file, index=False)

    batch_size = configuration["batch_size"] if "batch_size" in configuration else 4
    training_data_loader = DataLoader(
        Data(training_data_file, training=True, **configuration),
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        collate_fn=pad_batch
    )
    validation_data_loader = DataLoader(
        Data(validation_data_file, **configuration),
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=pad_batch,
    )

    model = rnapred(train_len=len(training_data_loader), **configuration)
    highest_f1, patience_counter = -1, 0
    patience_limit = configuration["patience_limit"] if "patience_limit" in configuration else 30
    if display_output:
        print("Training started...")
    maximum_epochs = configuration["maximum_epochs"] if "maximum_epochs" in configuration else 1000
    log_file = os.path.join(results_directory, "training_log.csv")

    for epoch in range(maximum_epochs):
        training_metrics = model.fit(training_data_loader)

        validation_metrics = model.test(validation_data_loader)

        if validation_metrics["f1"] > highest_f1:
            highest_f1 = validation_metrics["f1"]
            tr.save(model.state_dict(), os.path.join(results_directory, "model_weights.pmt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience_limit:
                break

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                log_message = ','.join(['epoch'] + [f"train_{k}" for k in sorted(training_metrics.keys())] + [f"valid_{k}" for k in sorted(validation_metrics.keys())]) + "\n"
                f.write(log_message)
                f.flush()
                if display_output:
                    print(log_message)

        with open(log_file, "a") as f:
            log_message = ','.join([str(epoch)] + [f'{training_metrics[k]:.4f}' for k in sorted(training_metrics.keys())] + [
                f'{validation_metrics[k]:.4f}' for k in sorted(validation_metrics.keys())]) + "\n"
            f.write(log_message)
            f.flush()
            if display_output:
                print(log_message)

def test(input_file, model_weights=None, result_file=None, config={}, worker_count=2, display_output=True):
    input_file = validate_file(input_file)
    if display_output not in config:
        config["verbose"] = display_output

    test_data_loader = DataLoader(
        Data(input_file, **config),
        batch_size=config["batch_size"] if "batch_size" in config else 4,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=pad_batch,
    )

    if model_weights is not None:
        model = rnapred(weights=model_weights, **config)
    else:
        model = rnapred(pretrained=True, **config)

    if display_output:
        print(f"Beginning test of {input_file}")
    test_results = model.test(test_data_loader)
    result_summary = ",".join([key for key in sorted(test_results.keys())]) + "\n" + ",".join([f"{test_results[key]:.3f}" for key in sorted(test_results.keys())])+ "\n"
    if result_file is not None:
        with open(result_file, "w") as file:
            file.write(result_summary)
    if display_output:
        print(result_summary)


def predict(input_data, seq_id='pred_id', model_weights=None, output_dir=None, return_logits=False, config={},
                    worker_count=2, generate_image=False, image_resolution=10, display_messages=True):
    if output_dir is None:
        output_type = "text"
    else:
        _, file_extension = os.path.splitext(output_dir)
        if file_extension == "":
            if os.path.isdir(output_dir):
                raise ValueError(f"Output directory {output_dir} already exists")
            os.makedirs(output_dir)
            output_type = "ct"
        elif file_extension != ".csv":
            raise ValueError(f"Output directory must be a .csv file or a folder, not {file_extension}")
        else:
            output_type = "csv"

    is_file_input = os.path.isfile(input_data)
    if is_file_input:
        prediction_file = validate_file(input_data)
    else:
        input_data = input_data.upper().strip()
        nucleotide_set = set([i for item in list(nu_dict.values()) for i in item] + list(nu_dict.keys()))
        if set(input_data).issubset(nucleotide_set):
            prediction_file = f"{seq_id}.csv"
            with open(prediction_file, "w") as f:
                f.write("id,sequence\n")
                f.write(f"{seq_id},{input_data}\n")
        else:
            raise ValueError(
                f"Invalid input nucleotide {set(input_data)}, either the file is missing or the sequence have invalid nucleotides (should be any of {nucleotide_set})")
    prediction_loader = DataLoader(
        Data(prediction_file, prediction=True, **config),
        batch_size=config["batch_size"] if "batch_size" in config else 4,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=pad_batch,
    )

    if model_weights is not None:
        weights = model_weights
        model = rnapred(weights=weights, **config)
    else:
        model = rnapred(pretrained=True, **config)

    if display_messages:
        print(f"Start prediction of {prediction_file}")

    predictions, logits_list = model.pred(prediction_loader, logits=return_logits)
    if generate_image:
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            ctfile = "tmp.ct"
            write_ct(ctfile, item.id, item.sequence, item.base_pairs)
            dotbracket = ct_to_dot(ctfile)

            image_file = item.id + ".png"
            if output_dir is not None and os.path.isdir(output_dir):
                image_file = os.path.join(output_dir, image_file)
            if dotbracket:
                dot2png(image_file, item.sequence, dotbracket, resolution=image_resolution)
            ct2svg("tmp.ct", image_file.replace(".png", ".svg"))

    if not is_file_input:
        os.remove(prediction_file)

    if output_type == "text":
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            ctfile = "tmp.ct"
            write_ct(ctfile, item.id, item.sequence, item.base_pairs)
            dotbracket = ct_to_dot(ctfile)
            print(item.id)
            print(item.sequence)
            print(dotbracket)
            print()
    elif output_type == "csv":
        predictions.to_csv(output_dir, index=False)
    else:  # ct
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            write_ct(os.path.join(output_dir, item.id + ".ct"), item.id, item.sequence, item.base_pairs)
    if return_logits:
        base_dir = os.path.split(output_dir)[0] if not os.path.isdir(output_dir) else output_dir
        if len(base_dir) == 0:
            base_dir = "."
        logits_dir = base_dir + "/logits/"
        os.mkdir(logits_dir)
        for id, pred, pred_post in logits_list:
            pickle.dump((pred, pred_post), open(os.path.join(logits_dir, id + ".pk"), "wb"))
def main():
    args = parser()
    if not args.no_cache and args.command == "train":
        cache_path = "cache/"
    else:
        cache_path = None
    config= {"device": args.d, "batch_size": args.batch,
             "valid_split": 0.1, "max_len": args.max_length, "verbose": not args.quiet, "cache_path": cache_path}
    if "max_epochs" in args:
        config["max_epochs"] = args.max_epochs
    if args.config is not None:
        config.update(json.load(open(args.config)))
    if config["cache_path"] is not None:
        shutil.rmtree(config["cache_path"], ignore_errors=True)
        os.makedirs(config["cache_path"])
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if args.command == "train":
        train(args.train_file, config, args.out_path,  args.valid_file, args.j)
    if args.command == "test":
        test(args.test_file, args.model_weights, args.out_path, config, args.j)
    if args.command == "pred":
        predict(input_data=args.pred_file,
                model_weights=args.model_weights,
                output_dir=args.out_path,
                return_logits=args.logits,
                config=config,
                worker_count=args.j,
                generate_image=args.draw,
                image_resolution=args.draw_resolution)

