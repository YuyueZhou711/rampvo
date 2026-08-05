import os

import json
import time
import torch
import argparse
import traceback
from tqdm import tqdm
from functools import partial
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils.seed_everything import seed_everything
from evaluate import evaluate as validate
from ramp.data_readers.TartanEvent import TartanEvent
from ramp.lietorch import SE3
from ramp.net import VONet
from ramp.checkpoints import load_ramp_state_dict
from ramp.utils import kabsch_umeyama

try:
    import wandb
    log = True
except ImportError:
    print("WARNING: wandb is not installed, cannot log results")
    log = False


def normalize_checkpoint_state(checkpoint):
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    return OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in state_dict.items()
        if "update.lmbda" not in key
    )


def load_checkpoint(path, net, optimizer, scheduler):
    """Load a legacy weights-only checkpoint or resume a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_ramp_state_dict(net, normalize_checkpoint_state(checkpoint))

    resume_keys = {
        "batch_idx",
        "total_idx",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    if not isinstance(checkpoint, dict) or not resume_keys.issubset(checkpoint):
        return 1, 0, 0, False

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        checkpoint["batch_idx"],
        checkpoint.get("epoch", 0),
        checkpoint["total_idx"],
        True,
    )


def compute_losses(traj, so, train_config, patch_size):
    loss = 0.0
    for i, (v, x, y, P1, P2) in enumerate(traj):
        e = (x - y).norm(dim=-1)
        e = e.reshape(-1, patch_size**2)[(v > 0.5).reshape(-1)].min(dim=-1).values

        N = P1.shape[1]
        ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))

        ii = ii.reshape(-1).cuda()
        jj = jj.reshape(-1).cuda()

        k = ii != jj
        ii = ii[k]
        jj = jj[k]

        P1 = P1.inv()
        P2 = P2.inv()

        t1 = P1.matrix()[..., :3, 3]
        t2 = P2.matrix()[..., :3, 3]

        s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
        P1 = P1.scale(s.view(1, 1))

        dP = P1[:, ii].inv() * P1[:, jj]  # relative predicted pose
        dG = P2[:, ii].inv() * P2[:, jj]  # ground truth

        e1 = (dP * dG.inv()).log()
        tr = e1[..., 0:3].norm(dim=-1)
        ro = e1[..., 3:6].norm(dim=-1)

        loss += train_config["flow_weight"] * e.mean()
        if not so and i >= 2:
            loss += train_config["pose_weight"] * (tr.mean() + ro.mean())

    return loss, e, ro, tr

def train(args):
    """main training loop"""
    seed_everything(seed=args.seed)
    with open(args.config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    train_cfg = config["data_loader"]["train"]["args"]
    log_results = args.log_results and log

    # Initialize network, optimizer and scheduler
    net = VONet(cfg=train_cfg)
    patch_size = net.P
    net.train()
    net.to("cuda")
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["lr"],
        total_steps=train_cfg["steps"],
        pct_start=train_cfg["pct_start"],
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    # Import either legacy weights or a resumable training checkpoint.
    batch_idx, epoch, step = 1, 0, 0
    resume_train = False
    if args.ckpt is not None:
        batch_idx, epoch, step, resume_train = load_checkpoint(
            args.ckpt, net, optimizer, scheduler
        )

    # prefetch_factor is only valid when multiprocessing workers are enabled.
    dataloader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "shuffle": train_cfg["shuffle"],
        "num_workers": args.workers,
    }
    if args.workers > 0:
        dataloader_kwargs["prefetch_factor"] = 1
    Dataloader = partial(DataLoader, **dataloader_kwargs)

    # Name your experiment through name argument or with config file
    run_name = args.name if args.name is not None else config["experiment_name"]
    if log_results:
        wandb.init(
            project="RAMP-VO",
            name=run_name,
            config=config,
            id=run_name,
            resume=resume_train,
        )
        wandb.watch(net, log="all")

    steps_run = 0
    stop_training = False
    for curr_epoch in range(10):
        if curr_epoch < epoch:
            continue
        db = TartanEvent(
            path=args.data_path, config=config, step=step, workers_n=args.workers
            )
        pbar = tqdm(
            Dataloader(dataset=db), mininterval=10, ncols=50
            )

        for batch_idx, data_blob in enumerate(pbar):
            if data_blob == 0:
                continue
            step += 1
            steps_run += 1

            events, images, poses, disps, K, supervision_mask = [x.to("cuda") for x in data_blob]

            optimizer.zero_grad()
            fix_repr_pose = step < 1000 and args.ckpt is None

            traj = net(
                STEPS=18,
                disps=disps,
                intrinsics=K,
                poses=SE3(poses).inv(),
                input_=(events, images, supervision_mask),
                structure_only=fix_repr_pose,
            )

            loss, e, ro, tr = compute_losses(
                traj=traj,
                so=fix_repr_pose,
                train_config=train_cfg,
                patch_size=patch_size,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg["clip"])
            optimizer.step()
            scheduler.step()
            metrics = {
                "loss": loss.item(),
                "px1": (e < 0.25).float().mean().item(),
                "ro": ro.float().mean().item(),
                "tr": tr.float().mean().item(),
            }

            if log_results:
                wandb.log(data=metrics, step=step)

            if step % train_cfg["steps_to_save_ckpt"] == 0:
                torch.cuda.empty_cache()
                directory = os.path.join("checkpoints", run_name)
                os.makedirs(directory, exist_ok=True)
                PATH = directory + "/%s_%06d.pth" % (run_name, step)
                torch.save(
                    {
                        "batch_idx": batch_idx,
                        "total_idx": step,
                        "epoch": curr_epoch,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    PATH,
                )

                steps_to_do_validation = 0
                if train_cfg.get("steps_to_do_validation") is not None:
                    steps_to_do_validation = train_cfg["steps_to_do_validation"]
                
                validation_results = None
                if step > steps_to_do_validation:
                    try:
                        valid_start = time.time()
                        validation_results = validate(
                            dataset_path=args.data_path,
                            eval_cfg=config,
                            net=net,
                            validation_subset=args.validation_subset,
                        )
                        for k in validation_results:
                            print(k, validation_results[k])
                        print("\n Validation time: ", time.time() - valid_start, "s")
                    except Exception:
                        traceback.print_exc()
                        print("\n VALIDATION HASN'T WORKED")

                if log_results and validation_results is not None:
                    wandb.log(data=validation_results, step=step)

                torch.cuda.empty_cache()
                net.train()

            if args.max_steps is not None and steps_run >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Dataset path")
    parser.add_argument("--name", type=str, default=None, help="name your experiment")
    parser.add_argument("--ckpt", type=str, help="checkpoint to restore")
    parser.add_argument("--config_path", type=str, required=True, help="config file path")
    parser.add_argument("--log_results", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="maximum optimizer steps in this invocation",
    )
    parser.add_argument(
        "--validation_subset",
        nargs="*",
        default=None,
        help="optional relative scene paths to evaluate",
    )

    train(args=parser.parse_args())
