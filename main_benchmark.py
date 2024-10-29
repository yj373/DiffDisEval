
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
from function_sc import stable_cascade_func


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
    # parser.add_argument(
    #     "--negative_token",
    #     action='store_true',
    #     help="whether to use negative token",
    # )
    parser.add_argument(
        "--single_run",
        action='store_true',
        help="whether to only run a single trial",
    )
    parser.add_argument(
        "--single_image",
        action='store_true',
        help="whether to only run a single trial",
    )
    parser.add_argument(
        "--all_masks",
        action='store_true',
        help="whether to try all the tokens",
    )
    parser.add_argument(
        "--num_cpu",
        type=int,
        default=4,
        help="number of CPU cores",
    )
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=1,
        help="number of GPUs",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=2,
        help="number of triles for tuning",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="random seed"
    )
    parser.add_argument(
        "--t",
        type=int,
        default=100,
        help="hyperparameter t"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=8.0,
        help="hyperparameter alpha"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
        help="hyperparameter beta"
    )
    parser.add_argument(
        "--neg_weight",
        type=float,
        default=0.0,
        help="hyperparameter neg_weight"
    )
    parser.add_argument(
        "--alpha_prior",
        type=float,
        default=0.4,
        help="hyperparameter alpha_prior"
    )
    parser.add_argument(
        "--beta_prior",
        type=float,
        default=0.4,
        help="hyperparameter beta_prior"
    )
    parser.add_argument(
        "--lr_flip",
        action='store_true',
        help="whether to flip the input image left to right",
    )
    parser.add_argument(
        "--ud_flip",
        action='store_true',
        help="whether to flip the input image up to down",
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


def main(args):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    # nltk.download()
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    if args.diffusion_model.endswith("stable-cascade"):
        search_space = {
            't': tune.uniform(90, 110),
            'neg_weight': tune.uniform(0, 0),
            'alpha_prior': tune.uniform(5, 10),
            'beta_prior': tune.uniform(0.2, 1),
        }
        reporter = CLIReporter()
        reporter.max_report_frequency = 60
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(stable_cascade_func, args=args), resources={"cpu": args.num_cpu, "gpu": args.num_gpu}),
            param_space=search_space,
            tune_config=tune.TuneConfig(search_alg=OptunaSearch(), metric="f1_auc", mode="max", num_samples=args.num_runs),
            run_config=RunConfig(progress_reporter=reporter),
        )
        # results = tuner.fit()
        # best_result = results.get_best_result("f1_auc", "max")
        # print(">>> best config: ", best_result.config)
        # print(">>> f1_auc: {:.4f}, f1_optim: {:.4f}, iou_auc: {:.4f}, iou_optim: {:.4f}, pixel_auc: {:.4f}, pixel_optim: {:.4f}".format(
        #     best_result["f1_auc"], best_result["f1_optim"], best_result["iou_auc"], best_result["iou_optim"], best_result["pixel_auc"], best_result["pixel_optim"]
        # ))
    else:
        search_space = {
            't': tune.uniform(80, 120),
            'map_weight1': tune.uniform(0.3, 0.3),
            'map_weight2': tune.uniform(0.5, 0.5),
            'map_weight3': tune.uniform(0.1, 0.1),
            'map_weight4': tune.uniform(0.1, 0.1),
            'neg_weight': tune.uniform(-2, 2),
            'alpha': tune.uniform(10, 20),
            'beta': tune.uniform(0.5, 0.9),
        }
    
        reporter = CLIReporter()
        reporter.max_report_frequency = 60
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(stable_diffusion_func, args=args), resources={"cpu": args.num_cpu, "gpu": args.num_gpu}),
            param_space=search_space,
            tune_config=tune.TuneConfig(search_alg=OptunaSearch(), metric="f1_auc", mode="max", num_samples=args.num_runs),
            # run_config=RunConfig(progress_reporter=reporter),
        )
    results = tuner.fit()
    best_result = results.get_best_result("f1_auc", "max")
    print(">>> best config: ", best_result.config)
    print(">>> f1_auc: {:.4f}, f1_optim: {:.4f}, iou_auc: {:.4f}, iou_optim: {:.4f}, pixel_auc: {:.4f}, pixel_optim: {:.4f}".format(
        best_result["f1_auc"], best_result["f1_optim"], best_result["iou_auc"], best_result["iou_optim"], best_result["pixel_auc"], best_result["pixel_optim"]
    ))


def main_single(args):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')

    if args.diffusion_model.endswith("stable-cascade"):
        config = {
            "t": args.t,
            "neg_weight": args.neg_weight,
            "alpha_prior": args.alpha_prior,
            "beta_prior": args.beta_prior,
        }
        stable_cascade_func(config, args)
    else:
        config = {
            "t": args.t,
            'map_weight1': 0.3,
            "map_weight2": 0.5,
            "map_weight3": 0.1,
            "map_weight4": 0.1,
            "neg_weight": args.neg_weight,
            'alpha': args.alpha,
            'beta': args.beta,
        }
        stable_diffusion_func(config, args)

if __name__ == '__main__':
    args = parse_args()
    if args.single_run:
        main_single(args)
    else:
        main(args)