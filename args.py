import toml
from tap import Tap


class Args(Tap):
    # Model params
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset: str = "cnndm"
    split: str = "validation"
    subsample: float = 1.0

    # Generation params
    max_input_len: int = -1  # -1 for no truncation
    beam_width: int = 1

    # BLooP params
    alpha: float = 0.0
    use_fw: bool = False

    # Other params
    model_seed: int = 1
    data_seed: int = 1
    profile: bool = False


DATASET_CONFIGS = toml.load("dataset_info.toml")


class DatasetInfo(Tap):
    name: str = DATASET_CONFIGS["cnndm"]["name"]
    doc_key: str = DATASET_CONFIGS["cnndm"]["doc_key"]
    ref_key: str = DATASET_CONFIGS["cnndm"]["ref_key"]
    subset: str = DATASET_CONFIGS["cnndm"]["subset"]


def create_namespaces() -> tuple[Args, DatasetInfo]:
    args = Args().parse_args()
    dataset_info = DatasetInfo()

    assert args.dataset in DATASET_CONFIGS, f"Dataset {args.dataset} not found"

    # Update dataset info from config
    for key, value in DATASET_CONFIGS[args.dataset].items():
        setattr(dataset_info, key, value)

    return args, dataset_info
