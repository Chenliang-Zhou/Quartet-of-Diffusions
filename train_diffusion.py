# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used to train a ATISS."""
import argparse
import json
import logging
import os
import sys

import numpy as np

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DiT3D.utils.visualize import visualize_pointcloud_batch
from DiffuScene.scripts.utils import rotate_point_cloud
from training_utils import load_config, save_checkpoints

from scene_diffusion.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from scene_diffusion.stats_logger import StatsLogger, WandB

sys.path.append('..')
from DiT3D.datasets.shapenet_data_pc import ShapeNet15kPointClouds
from DiT3D.models.point_cloud_encoder_and_sampler import get_pretrained_encoder_and_sampler


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on transformations of parts of 3D shapes"
    )

    parser.add_argument("-t", "--test", help="Run the test suite", action="store_true")

    parser.add_argument(
        "--config_file",
        help="Path to the file that contains the experiment configuration",
        default="config/diffusion/diffusion_structure_aware_gen.yaml"
    )
    parser.add_argument(
        "--output_directory",
        help="Path to the output directory",
        default="/local/scratch/cz363/phd-research/pretrained/DiffusionAssembler/"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=12,  # 0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )

    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )
    parser.add_argument('--distribution_type', default='multi', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Parse the config file and Save the parameters of this run to a file
    config = load_config(args.config_file)
    experiment_directory = os.path.join(args.output_directory, "test" if args.test else config["dataset"]["category"])
    os.makedirs(experiment_directory, exist_ok=True)
    num_labels = {"airplane": 4}[config["dataset"]["category"]]
    latent_dim = 128
    config["network"]["angle_dim"] = config["network"]["size_dim"] = config["network"]["translation_dim"] = \
        config["network"]["net_kwargs"]["angle_dim"] = config["network"]["net_kwargs"]["size_dim"] = \
        config["network"]["net_kwargs"]["translation_dim"] = 3 * num_labels
    config["network"]["point_dim"] = config["network"]["net_kwargs"]["channels"] = config["network"]["angle_dim"] + \
                                                                                   config["network"]["size_dim"] + \
                                                                                   config["network"]["translation_dim"]
    config["network"]["net_kwargs"]["context_dim"] = (1 + num_labels) * latent_dim
    # config["network"]["net_kwargs"]["instanclass_dim"] = num_labels

    json_path = os.path.join(experiment_directory, "options.json")
    with open(json_path, "w") as f:
        json.dump({**vars(args), **config}, f, indent=4)
    print("Save experiment options in {}".format(json_path))

    train_dataset = ShapeNet15kPointClouds(root_dir=config["dataset"]["dataroot"],
                                           categories=config["dataset"]["category"].split(','), split=None,
                                           tr_sample_size=config["dataset"]["max_npoints"],
                                           te_sample_size=0,
                                           normalize_per_shape=True,
                                           normalize_std_per_axis=True,
                                           random_subsample=True,
                                           for_assembler=True, max_label=num_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        # collate_fn=train_dataset.collate_fn,
        shuffle=True
    )

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(0, 1,
                                                               config, args.weight_file, device=device)
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, network.parameters())]))
    print(f"Number of parameters in {network.__class__.__name__}:  {n_trainable_params} / {n_all_params}")
    pc_encoder, latent_dpm = get_pretrained_encoder_and_sampler(config["dataset"]["category"])
    pc_encoder.to(device)
    latent_dpm.to(device)

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], filter(lambda p: p.requires_grad, network.parameters()))
    # optimizer = optimizer_factory(config["training"], network.parameters() )

    # Load the checkpoints if they exist in the experiment directory
    # load_checkpoints(network, optimizer, experiment_directory, args, device)
    # Load the learning rate scheduler 
    lr_scheduler = schedule_factory(config["training"])

    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get(
                "project", "autoregressive_transformer"
            ),
            name=config["dataset"]["category"],
            watch=False,
            log_frequency=10
        )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(os.path.join(experiment_directory, "stats.txt"), "w"))

    num_epochs = 3 if args.test else config["training"]["epochs"]
    num_vis = config["training"]["num_vis"]
    gt_for_vis_saved = False

    # Do the training
    epochs = []
    losses = []
    for i in range(1, num_epochs + 1):
        epochs.append(i)
        adjust_learning_rate(lr_scheduler, optimizer, i - 1)
        network.train()
        epoch_loss = 0
        for b, x in enumerate(train_loader):
            B = x['train_points'].shape[0]
            sample = {}
            sample["translations"] = x["translation"].to(device).reshape(B, -1)
            sample["sizes"] = x["scale"].to(device).reshape(B, -1)
            sample["angles"] = x["rotation"].to(device).reshape(B, -1)
            part_latent = pc_encoder.encode(x['train_points'].to(device).flatten(end_dim=1)).reshape(B, -1)
            full_pc_latent = pc_encoder.encode(x['full_pc'].to(device))
            sample["condition"] = torch.cat([part_latent, full_pc_latent], dim=1)

            batch_loss = train_on_batch(network, optimizer, sample, config)
            if args.test:
                StatsLogger.instance().print_progress(i, b + 1, batch_loss)
            epoch_loss += batch_loss * len(x)
            if not gt_for_vis_saved:
                gt_translations = x["translation"].to(device)[:num_vis]
                gt_sizes = x["scale"].to(device)[:num_vis]
                gt_angles = x["rotation"].to(device)[:num_vis]
                gt_part_latent = part_latent[:num_vis]
                gt_full_pc = x['full_pc'].to(device)[:num_vis]
                gt_full_pc_latent = full_pc_latent[:num_vis]
                gt_condition = torch.cat([gt_part_latent, gt_full_pc_latent], dim=1)
                gt_parts = x['train_points'].to(device)[:num_vis]
                visualize_pointcloud_batch(os.path.join(experiment_directory, 'gt_full_pc.png'), gt_full_pc)
                gt_full_pc_path = os.path.join(experiment_directory, 'gt_full_pc.pt')
                torch.save(gt_full_pc, gt_full_pc_path)
                print("Groundtruth full pointclouds saved at", gt_full_pc_path)
                gt_transformation_path = os.path.join(experiment_directory, 'gt_transformation.pt')
                torch.save(torch.cat([gt_translations, gt_sizes, gt_angles], dim=-2), gt_transformation_path)
                print("Groundtruth transformations saved at", gt_transformation_path)
                gt_parts_path = os.path.join(experiment_directory, 'gt_parts.pt')
                torch.save(gt_parts, gt_parts_path)
                print("Groundtruth parts saved at", gt_parts_path)
                # gt_assembled_parts = (rotate_point_cloud(gt_parts, gt_angles) * gt_sizes + gt_translations).reshape(num_vis, -1, 3)
                gt_assembled_parts = (gt_parts * gt_sizes + gt_translations).reshape(num_vis, -1, 3)
                visualize_pointcloud_batch(os.path.join(experiment_directory, 'gt_assembled_parts.png'),
                                           gt_assembled_parts)
                gt_for_vis_saved = True

        epoch_loss /= len(train_dataset)
        losses.append(epoch_loss)
        StatsLogger.instance().print_progress(i, None, epoch_loss)
        StatsLogger.instance().clear()

        if i == num_epochs or not i % config["training"]["save_frequency"]:
            save_checkpoints(i, network, optimizer, experiment_directory)
        # if i == 1 or i == epochs or not i % val_every:
        #     print("====> Validation Epoch ====>")
        #     network.eval()
        #     for b, sample in enumerate(val_loader):
        #         # Move everything to device
        #         for k, v in sample.items():
        #             if not isinstance(v, list):
        #                 sample[k] = v.to(device)
        #         batch_loss = validate_on_batch(network, sample, config)
        #         StatsLogger.instance().print_progress(-1, b + 1, batch_loss)
        #     StatsLogger.instance().clear()
        #     print("====> Validation Epoch ====>")
        if i == num_epochs or not i % config["training"]["vis_frequency"]:
            print('Start visualization.')

            network.eval()
            with torch.no_grad():
                transformations = network.sample(num_vis, gt_condition).reshape(num_vis, num_labels, 3, 3)
                translations = transformations[:, :, 0:1, :]
                sizes = transformations[:, :, 1:2, :]
                angles = transformations[:, :, 2:3, :]
            gen_transformation_path = os.path.join(experiment_directory, f'gen_transformation_epoch{i}.pt')
            torch.save(transformations, gen_transformation_path)
            print("Generated transformations saved at", gen_transformation_path)
            # gen_assembled_parts = (rotate_point_cloud(gt_parts, angles) * sizes + translations).reshape(num_vis, -1, 3)
            gen_assembled_parts = (gt_parts * sizes + translations).reshape(num_vis, -1, 3)
            visualize_pointcloud_batch(os.path.join(experiment_directory, f'gen_assembled_parts_epoch{i}.png'),
                                       gen_assembled_parts)

            # plot loss
            plt.plot(epochs, losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            loss_plot_path = os.path.join(experiment_directory, "loss_plot.png")
            plt.savefig(loss_plot_path)
            plt.close()
            print("Loss plot saved at", loss_plot_path)
            print("Finished visualization.")
    print("Finished training.")


if __name__ == "__main__":
    # main(["-t"])
    main([])