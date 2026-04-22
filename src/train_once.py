import torch
from config.presets import get_preset
from train.registry import get_trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preset_name = "ls_lstm_word_semantic_hier"
    seed = 0

    cfg = get_preset(preset_name, seed=seed)
    cfg.ensure_dirs()
    cfg.validate()

    trainer = get_trainer(cfg)
    trainer(cfg, device)


if __name__ == "__main__":
    main()