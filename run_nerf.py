import os
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
from typing_extensions import Annotated

from model.feature_embedding import PositionalEmbedding
from training.inference.infer import infer
from training.train import train
from volume_handling.data_handling import NeRF_Data_Loader
from volume_handling.sampling import NeRF_Stratified_Sampler

app = typer.Typer()


def config_parser_training() -> dict:
    import configargparse

    # Create a function to convert the argument to a boolean
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise configargparse.ArgumentTypeError("Boolean value expected.")

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, required=True, help="name of your experiment; is required")
    parser.add_argument("--data_path", type=str, required=True, help="path to the trained dataset; is required")
    parser.add_argument("--out_path", type=str, default="experiments/", help="output destination of your experiment")
    parser.add_argument(
        "--tensorboard_log_dir", type=str, default="tensorboard_runs/", help="output destination of tensorboard logs"
    )

    parser.add_argument("--near", type=float, default=2.0, help="Near clipping distance")
    parser.add_argument("--far", type=float, default=6.0, help="Far clipping distance")
    parser.add_argument("--n_training", type=int, default=100, help="Far clipping distance")

    # Encoders
    parser.add_argument("--d_input", type=int, default=3, help="Number of input dimensions")
    parser.add_argument("--n_freqs", type=int, default=10, help="Number of encoding functions for samples")
    parser.add_argument("--log_space", type=str2bool, default="True", help="If set, frequencies scale in log space")
    parser.add_argument("--use_viewdirs", type=str2bool, default="True", help="If set, use view direction as input")
    parser.add_argument("--n_freqs_views", type=int, default=4, help="Number of encoding functions for views")

    # Stratified sampling
    parser.add_argument("--n_samples", type=int, default=64, help="Number of spatial samples per ray")
    parser.add_argument("--perturb", type=str2bool, default="True", help="If set, applies noise to sample positions")
    parser.add_argument(
        "--inverse_depth", type=str2bool, default="False", help="If set, samples points linearly in inverse depth"
    )

    # Model
    parser.add_argument("--d_Weights", type=int, default=128, help="Dimensions of linear layer filters")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in network bottleneck")
    parser.add_argument("--skip", nargs="*", type=int, default=[], help="Layers at which to apply input residual")
    parser.add_argument("--use_fine_model", type=str2bool, default="True", help="If set, creates a fine model")
    parser.add_argument(
        "--d_Weights_fine", type=int, default=128, help="Dimensions of linear layer filters of fine network"
    )
    parser.add_argument("--n_layers_fine", type=int, default=6, help="Number of layers in fine network bottleneck")

    # Hierarchical sampling
    parser.add_argument("--n_samples_hierarchical", type=int, default=64, help="Number of samples per ray")
    parser.add_argument(
        "--perturb_hierarchical", type=str2bool, default="False", help="If set, applies noise to sample positions"
    )

    # Optimizer
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # Training
    parser.add_argument("--n_iters", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=2**14, help="Number of rays per gradient step (power of 2)")
    parser.add_argument(
        "--one_image_per_step", type=str2bool, default="False", help="One image per gradient step (disables batching)"
    )
    parser.add_argument("--chunksize", type=int, default=2**14, help="Modify as needed to fit in GPU memory")
    parser.add_argument(
        "--center_crop", type=str2bool, default="True", help="Debugging: Crop the center of image (one_image_per_)"
    )
    parser.add_argument(
        "--center_crop_iters", type=int, default=50, help="Debugging: Stop cropping center after this many epochs"
    )
    parser.add_argument("--display_rate", type=int, default=50, help="Debugging: Display test output every X epochs")
    parser.add_argument(
        "--model_storage_rate", type=int, default=5001, help="Debugging: Store model params every X epochs"
    )

    # Early Stopping
    parser.add_argument("--warmup_iters", type=int, default=100, help="Number of iterations during warmup phase")
    parser.add_argument(
        "--warmup_min_fitness", type=float, default=10.0, help="Min val PSNR to continue training at warmup_iters"
    )
    parser.add_argument("--n_restarts", type=int, default=10, help="Number of times to restart if training stalls")

    return parser.parse_args()


@app.command()
def training(config: Annotated[str, typer.Argument(help="The path to the config file")]):
    args = vars(config_parser_training())
    success = train(args)


@app.command()
def inference():
    args = vars(config_parser_training())
    success = infer(args)


if __name__ == "__main__":
    app()
