# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import os
import sys
import uuid
from pathlib import Path
from argparse import Namespace

import sacred
import submitit

import train
from trackformer.util.misc import nested_dict_to_namespace

WORK_DIR = str(Path(__file__).parent.absolute())


ex = sacred.Experiment('submit', ingredients=[train.ex])
ex.add_config('cfgs/submit.yaml')


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/storage/slurm").is_dir():
        path = Path(f"/storage/slurm/{user}/runs")
        path.mkdir(exist_ok=True)
        return path
    raise RuntimeError("No shared folder available")


def get_init_file() -> Path:
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer:
    def __init__(self, args: Namespace) -> None:
        self.args = args

    def __call__(self) -> None:
        sys.path.append(WORK_DIR)

        import train
        self._setup_gpu_args()
        train.train(self.args)

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        import os

        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
            self.args.resume_optim = True
            self.args.resume_vis = True
            self.args.load_mask_head_from_model = None
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self) -> None:
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        print(self.args.output_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main(args: Namespace):
    # Note that the folder will depend on the job_id, to easily track experiments
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    executor = submitit.AutoExecutor(
        folder=args.job_dir, cluster=args.cluster, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.num_gpus
    nodes = args.nodes
    timeout_min = args.timeout

    if args.slurm_gres:
        slurm_gres = args.slurm_gres
    else:
        slurm_gres = f'gpu:{num_gpus_per_node},VRAM:{args.vram}'
        # slurm_gres = f'gpu:rtx_8000:{num_gpus_per_node}'

    executor.update_parameters(
        mem_gb=args.mem_per_gpu * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72,
        slurm_partition=args.slurm_partition,
        slurm_constraint=args.slurm_constraint,
        slurm_comment=args.slurm_comment,
        slurm_exclude=args.slurm_exclude,
        slurm_gres=slurm_gres
    )

    executor.update_parameters(name="fair_track")

    args.train.dist_url = get_init_file().as_uri()
    # args.output_dir = args.job_dir

    trainer = Trainer(args.train)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

    if args.cluster == 'debug':
        job.wait()


@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    # TODO: hierachical Namespacing for nested dict
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    # args.train = Namespace(**config['train'])
    main(args)
