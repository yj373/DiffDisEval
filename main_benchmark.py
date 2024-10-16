
import os
import time
import shutil
import nltk
import argparse
import numpy as np

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune import ResultGrid
from ray.tune import CLIReporter
from ray.train import RunConfig
from function import stable_diffusion_func


def parse_args():
    parser = argparse.ArgumentParser(description='DiffDisEval')
    parser.add_argument(
        "--diffusion_model",
        type=str,
        default="/dev/shm/alexJiang/source/stable-diffusion-v1-4",
        required=True,
        help="directory containing the diffusion model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dev/shm/alexJiang/output/DiffDisEval",
        help="directory containing the output result",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/dev/shm/alexJiang/eval_dataset/large_500_300_150_50",
        help="directory containing the evaluation dataset",
    )
    parser.add_argument(
        "--BLIP",
        type=str,
        default="/dev/shm/alexJiang/source/BLIP",
        help="directory containing the BLIP model",
    )
    parser.add_argument(
        "--negative_token",
        action='store_true',
        help="whether to use negative token",
    )
    parser.add_argument(
        "--num_cpu",
        type=int,
        default=4,
        required=True,
        help="number of CPU cores",
    )
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=1,
        required=True,
        help="number of GPUs",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        required=True,
        help="number of triles for tuning",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="random seed"
    )
    args = parser.parse_args()
    return args


class ExperimentTerminationReporter(CLIReporter):
    def should_report(self, trials, done=False):
        """Reports only on experiment termination."""
        return done


class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


def main():
    args = parse_args()
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    # nltk.download()
    search_space = {
        't': tune.uniform(50, 150),
        'map_weight1': tune.uniform(0, 1),
        'map_weight2': tune.uniform(0, 1),
        'map_weight3': tune.uniform(0, 1),
        'map_weight4': tune.uniform(0, 1),
        'alpha': tune.uniform(1, 10),
        'beta': tune.uniform(0, 1),
    }
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    # reporter = CLIReporter()
    # reporter.update_interval = 60
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(stable_diffusion_func, args=args), resources={"cpu": args.num_cpu, "gpu": args.num_gpu}),
        param_space=search_space,
        tune_config=tune.TuneConfig(search_alg=OptunaSearch(), metric="f1_auc", mode="max", num_samples=args.num_runs),
        run_config=RunConfig(progress_reporter=ExperimentTerminationReporter()),
        # run_config=RunConfig(progress_reporter=reporter),
    )
    results = tuner.fit()
    best_result = results.get_best_result("f1_auc", "max")
    print(">>> best config: ", best_result.config)
    print(">>> f1_auc: {:.4f}, f1_optim: {:.4f}, iou_auc: {:.4f}, iou_optim: {:.4f}, pixel_auc: {:.4f}, pixel_optim: {:.4f}".format(
        best_result["f1_auc"], best_result["f1_optim"], best_result["iou_auc"], best_result["iou_optim"], best_result["pixel_auc"], best_result["pixel_optim"]
    ))

if __name__ == '__main__':
    main()