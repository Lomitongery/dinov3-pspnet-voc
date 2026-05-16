# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import lightning
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything


class EoMTCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.num_classes", "model.init_args.num_classes")
        parser.link_arguments("data.init_args.num_classes", "model.init_args.network.init_args.num_classes")
        parser.link_arguments("data.init_args.stuff_classes", "model.init_args.stuff_classes")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.encoder.init_args.img_size")
        parser.link_arguments("model.init_args.ckpt_path", "model.init_args.network.init_args.encoder.init_args.ckpt_path")


def cli_main():
    seed_everything(0)

    parser = EoMTCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
    )


if __name__ == "__main__":
    cli_main()
