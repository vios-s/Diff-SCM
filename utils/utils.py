import torch
import logging
from pathlib import Path


def restore_checkpoint(ckpt_dir: Path, state, device):
    ckpt_file = ckpt_dir / "checkpoint.pth"

    if not ckpt_file.exists():
        logging.warning(f"No checkpoint found at {ckpt_file}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_file, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir: Path, state):
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_file = ckpt_dir / "checkpoint.pth"
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_file)


def get_logger(logger_name, create_file=False):
    # create logger for prd_ci
    log = logging.getLogger(logger_name)
    log.setLevel(level=logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if create_file:
        # create file handler for logger.
        fh = logging.FileHandler('my_last_exp.log')
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
    # reate console handler for logger.
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if create_file:
        log.addHandler(fh)

    log.addHandler(ch)
    return log


def print_memory_alloc():
    torch.cuda.empty_cache()
    mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
    print("memory allocated:", mem, "MiB")
