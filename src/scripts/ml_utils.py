import optuna
from fastai.vision.all import *
from optuna.integration import FastAIPruningCallback
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import gc
from enum import Enum, auto
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import Dataset

FIXED_SEED = 42


class ModelType(Enum):
    RESNET18 = auto()
    RESNET34 = auto()
    RESNET50 = auto()
    MOBILENET_V3_LARGE = auto()


class GenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Extract gender from filename (1 is the index for gender in your filename)
        gender = int(img_name.split("_")[1])

        if self.transform:
            image = self.transform(image)

        return image, gender


class GenderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Create the full dataset
        full_dataset = GenderDataset(self.data_dir, transform=self.transform)

        # Split into train and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


def _get_base_model(model_type: ModelType):
    if model_type == ModelType.RESNET18:
        return models.resnet18
    elif model_type == ModelType.RESNET34:
        return models.resnet34
    elif model_type == ModelType.RESNET50:
        return models.resnet50
    elif model_type == ModelType.MOBILENET_V3_LARGE:
        return models.mobilenet_v3_large
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def utils_set_seed(seed=FIXED_SEED):
    """Sets the seed for reproducibility."""
    set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


utils_set_seed(FIXED_SEED)


def create_dataloaders_OLD(batch_size, aug_mode, path="Mushrooms"):
    utils_set_seed(FIXED_SEED)
    if aug_mode == "mult_2":
        batch_tfms = [
            *aug_transforms(mult=2.0, do_flip=True, flip_vert=True, max_rotate=25.0),
            Normalize.from_stats(*imagenet_stats),
        ]
    elif aug_mode == "mult_2_no_trans":
        print("mult_2_no_trans")
        batch_tfms = [*aug_transforms(mult=2.0), Normalize.from_stats(*imagenet_stats)]
    elif aug_mode == "mult_1.25_more_trans":
        print("mult_1.25_more_trans")
        batch_tfms = [
            *aug_transforms(
                mult=1.25,
                do_flip=True,
                flip_vert=True,
                max_rotate=25.0,
                min_zoom=0.75,
                max_zoom=1.8,
                max_warp=0.15,
                p_affine=0.05,
                xtra_tfms=[RandomErasing(p=0.2, sl=0.0, sh=0.3)],
            ),
            Normalize.from_stats(*imagenet_stats),
        ]
    elif aug_mode == "mult_1.0_more_trans":
        print("mult_1.0_more_trans")
        batch_tfms = [
            *aug_transforms(
                mult=1.00,
                do_flip=True,
                flip_vert=True,
                max_rotate=25.0,
                min_zoom=0.75,
                max_zoom=1.8,
                max_warp=0.15,
                p_affine=0.05,
                xtra_tfms=[RandomErasing(p=0.2, sl=0.0, sh=0.3)],
            ),
            Normalize.from_stats(*imagenet_stats),
        ]
    elif aug_mode == "mult_0.75_more_trans":
        print("mult_0.75_more_trans")
        batch_tfms = [
            *aug_transforms(
                mult=0.75,
                do_flip=True,
                flip_vert=True,
                max_rotate=25.0,
                min_zoom=0.75,
                max_zoom=1.8,
                max_warp=0.15,
                p_affine=0.05,
                xtra_tfms=[RandomErasing(p=0.2, sl=0.0, sh=0.3)],
            ),
            Normalize.from_stats(*imagenet_stats),
        ]

    elif aug_mode == "mult_2.0_more_trans":
        print("asd 222 NO")
        batch_tfms = [
            *aug_transforms(
                mult=2.0,
                do_flip=True,
                flip_vert=True,
                max_rotate=25.0,
                min_zoom=0.75,
                max_zoom=1.8,
                max_warp=0.15,
                p_affine=0.05,
                xtra_tfms=[RandomErasing(p=0.2, sl=0.0, sh=0.3)],
            ),
            Normalize.from_stats(*imagenet_stats),
        ]
    elif aug_mode == "min_aug":
        batch_tfms = [
            *aug_transforms(size=224, min_scale=0.75),
            Normalize.from_stats(*imagenet_stats),
        ]
    else:
        print("no augmentations")
        batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    data = ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        seed=FIXED_SEED,
        item_tfms=Resize(224),
        # batch_tfms=[*aug_transforms(size=224, min_scale=0.75), Normalize.from_stats(*imagenet_stats)],
        # batch_tfms=[*aug_transforms(mult=2.0, do_flip=True, flip_vert=True, max_rotate=20.0), Normalize.from_stats(*imagenet_stats)],
        batch_tfms=batch_tfms,
        num_workers=4,
        bs=batch_size,
    )
    return data


def create_dataloaders(
    batch_size: int, aug_mode: str, path: str, is_test=False, valid_pct=0.2
):
    utils_set_seed(FIXED_SEED)
    normalization = Normalize.from_stats(*imagenet_stats)
    print(
        f"Augmentation Mode: {aug_mode} | Data Type: {'Test' if is_test else 'Train'}"
    )

    # path = f"{path}/test" if is_test else f"{path}/train"

    if is_test:
        valid_pct = 0.001
        aug_mode = None

    if aug_mode == "mult_2":
        batch_tfms = aug_transforms(
            mult=2.0, do_flip=True, flip_vert=True, max_rotate=25.0
        )
    elif aug_mode == "mult_2_no_trans":
        batch_tfms = aug_transforms(mult=2.0)
    elif aug_mode in (
        "mult_1.25_more_trans",
        "mult_1.0_more_trans",
        "mult_0.75_more_trans",
        "mult_2.0_more_trans",
    ):
        mult = float(aug_mode.split("_")[1])
        batch_tfms = aug_transforms(
            mult=mult,
            do_flip=True,
            flip_vert=True,
            max_rotate=25.0,
            min_zoom=0.75,
            max_zoom=1.8,
            max_warp=0.15,
            p_affine=0.05,
            xtra_tfms=[RandomErasing(p=0.2, sl=0.0, sh=0.3)],
        )
    elif aug_mode == "min_aug":
        batch_tfms = aug_transforms(size=224, min_scale=0.75)
    else:
        print("No augmentations applied.")
        batch_tfms = []

    batch_tfms = [*batch_tfms, normalization] if batch_tfms else [normalization]

    data = ImageDataLoaders.from_folder(
        path,
        valid_pct=valid_pct,
        seed=FIXED_SEED,
        item_tfms=Resize(224),
        shuffle=False,
        batch_tfms=batch_tfms,
        num_workers=20,
        bs=batch_size,
    )
    return data


def compute_weights(data, path):
    label_dict = {v: k for k, v in enumerate(data.vocab)}  # get class to index mapping
    counts = np.zeros(len(label_dict))

    for label in label_dict:
        # Count files in each directory/class
        dir_path = os.path.join(path, label)
        counts[label_dict[label]] = len(os.listdir(dir_path))

    total_counts = np.sum(counts)
    class_weights = total_counts / (len(label_dict) * counts)
    class_weights = torch.tensor(
        class_weights, dtype=torch.float, device="cuda"
    )  # Adjust device as needed

    return class_weights


def create_learner(
    data: DataLoader,
    weight_decay: float,
    balancing: str,
    model_type: ModelType = ModelType.RESNET34,
):
    base_model = _get_base_model(model_type)
    print(f"using {model_type.name}")
    print(f"{base_model}")

    utils_set_seed(FIXED_SEED)
    if balancing == "class_weight":
        class_weights = compute_weights(data, path)
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_func = LabelSmoothingCrossEntropy()

    learn = vision_learner(
        data,
        base_model,
        # models.resnet50,
        metrics=[
            accuracy,
            F1Score(average="weighted"),
            F1Score(average="micro"),
            F1Score(average="macro"),
        ],
        wd=weight_decay,
        loss_func=loss_func,
        # loss_func=LabelSmoothingCrossEntropy(),
    )
    return learn


def train_model(
    learn,
    base_lr,
    lr_mult,
    lr_scheduler_type,
    freeze_epochs,
    num_epochs,
    pct_start,
    callbacks=None,
):
    utils_set_seed(FIXED_SEED)

    if torch.cuda.is_available():
        learn.model.to("cuda")  # Explicitly move the model to GPU
    else:
        raise RuntimeError("CUDA is not available. Training on GPU is not possible.")

    if next(learn.model.parameters()).is_cuda:
        print("Model is on CUDA")
    else:
        raise Exception("Model is not on GPU")

    if lr_scheduler_type == "one_cycle":
        learn.fit_one_cycle(
            freeze_epochs, slice(base_lr), pct_start=0.99, cbs=callbacks
        )
        learn.unfreeze()
        learn.fit_one_cycle(
            num_epochs,
            slice(base_lr / lr_mult, base_lr),
            pct_start=pct_start,
            cbs=callbacks,
        )
    elif lr_scheduler_type == "flat_cos":
        learn.fit_flat_cos(freeze_epochs, slice(base_lr), pct_start=0.99, cbs=callbacks)
        learn.unfreeze()
        learn.fit_flat_cos(
            num_epochs,
            slice(base_lr / lr_mult, base_lr),
            pct_start=pct_start,
            cbs=callbacks,
        )

    return learn


def extract_metrics(learn):
    metrics = {
        "train_loss": learn.recorder.final_record[0],
        "valid_loss": learn.recorder.final_record[1],
        "accuracy": learn.recorder.final_record[2],
        "f1_weighted": learn.recorder.final_record[3],
        "f1_micro": learn.recorder.final_record[4],
        "f1_macro": learn.recorder.final_record[5],
    }
    return metrics


def objective(trial):
    utils_set_seed(FIXED_SEED)

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 224, 256])
    # weight_decay = trial.suggest_float('weight_decay', 0.00001, 0.1, log=True)
    # base_lr = trial.suggest_float('base_lr', 0.00001, 0.1, log=True)
    base_lr = trial.suggest_float("base_lr", 0.001, 0.01, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.00005, 0.1, log=True)

    lr_mult = trial.suggest_categorical("lr_mult", [10, 50, 100])
    lr_scheduler_type = trial.suggest_categorical(
        "lr_scheduler", ["one_cycle", "flat_cos"]
    )
    freeze_epochs = trial.suggest_int("freeze_epochs", 1, 6)
    pct_start = trial.suggest_float("pct_start", 0.01, 0.99)
    balancing = None

    aug_mode = trial.suggest_categorical(
        "aug_mode", [None, "mult_2", "mult_2.0_more_trans", "mult_1.25_more_trans"]
    )

    num_epochs = 25  # trial.suggest_int('num_epochs', 2, 10)

    data = create_dataloaders(batch_size, aug_mode=aug_mode)

    learn = create_learner(data, weight_decay, balancing)

    callbacks = [
        FastAIPruningCallback(trial, monitor="valid_loss"),
        EarlyStoppingCallback(monitor="valid_loss", patience=5),
    ]
    learn = train_model(
        learn,
        base_lr,
        lr_mult,
        lr_scheduler_type,
        freeze_epochs,
        num_epochs,
        pct_start,
        callbacks=callbacks,
    )
    metrics = extract_metrics(learn)

    print(
        f'Trial {trial.number}: train_loss: {metrics["train_loss"]}, valid_loss: {metrics["valid_loss"]}, Accuracy {metrics["accuracy"]}, F1 Weighted {metrics["f1_weighted"]}, F1 Micro {metrics["f1_micro"]}, F1 Macro {metrics["f1_macro"]}'
    )

    del data
    del learn
    torch.cuda.empty_cache()
    gc.collect()

    return metrics["f1_weighted"]


def get_study_params(row):
    row_params = {}
    for col, value in row.items():
        if col.startswith("params"):
            row_params[col.replace("params_", "")] = value
            if isinstance(value, float):
                if math.isnan(value):
                    # print(f"{col}: {value}")
                    row_params[col.replace("params_", "")] = None
    return row_params
