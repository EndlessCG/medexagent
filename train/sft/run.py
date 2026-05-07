"""CLI entry point for Stage 1: Supervised Fine-Tuning."""

import argparse

import yaml

from .train import train


def main():
    parser = argparse.ArgumentParser(description="Medical Agent SFT Training")
    parser.add_argument(
        "--config", type=str, default="config/sft_config.yaml",
        help="Path to YAML config file",
    )
    # Allow CLI overrides for any config key
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_lora", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for wandb")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume training from, or 'latest' to auto-detect")
    parser.add_argument("--eval_before_train", type=lambda x: x.lower() == "true", default=None,
                        help="Run evaluation before training starts (default: true)")
    parser.add_argument("--stream_training", type=lambda x: x.lower() == "true", default=None,
                        help="Enable stream training mode (monitors data file for updates)")
    parser.add_argument("--max_pending_time", type=float, default=None,
                        help="Max time (in seconds) to wait for new data before stopping (default: 1800 = 30 min)")
    parser.add_argument("--epochs_per_data", type=int, default=None,
                        help="Number of epochs to train on each data batch before waiting for more (default: 1)")
    parser.add_argument("--stream_check_interval", type=float, default=None,
                        help="Seconds between file checks while waiting for new data (default: 60)")

    args = parser.parse_args()

    # Load YAML config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    print(f"Config: {config}")
    train(config)


if __name__ == "__main__":
    main()
