"""Train the linear probe and write data/probe.npz.

    .venv/bin/python training/train_probe.py train data/train data/probe.npz

Expects ``data/train`` to hold one subdirectory per source class. The mapping
onto the six visual groups is SOURCE_TO_GROUP below; source classes absent from
it are skipped, which is how sandstorm is dropped.

CLEAR and CLOUDY have no representative class in the Weather Image Recognition
dataset (jehanbhathena/weather-dataset, unzipped into data/train). They come
instead from the Multi-class Weather Dataset (pratik2901/multiclass-weather-dataset),
whose four classes are Cloudy, Rain, Shine, Sunrise.

That second dataset is also the only place a genuinely held-out test set can
come from for CLEAR/CLOUDY. The original task plan said "train CLEAR/CLOUDY
from data/test, then measure accuracy on data/test" -- that trains and tests
on the same images and would report a fake number. Instead, split_dataset()
below performs a deterministic, file-level 50/50 split of the Multi-class
Weather Dataset: sort each class's filenames, then alternate -- even index
goes to the train half, odd index to the test half. No file ever appears on
both sides, and the split is reproducible from source data alone.

    .venv/bin/python training/train_probe.py split data/_raw_multiclass/"Multi-class Weather Dataset" data/train data/test

Run `split` first (copies half of each of the 4 classes into data/train and
the other half into data/test), then run `train` against the combined
data/train (Weather Image Recognition classes + the Multi-class train half).
"""

import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from app.model import VisionModel
from app.prompts import GROUP_ORDER

SOURCE_TO_GROUP = {
    "dew": "FOG",
    "frost": "FOG",
    "fogsmog": "FOG",
    "glaze": "SNOW",
    "rime": "SNOW",
    "snow": "SNOW",
    "hail": "STORM",
    "lightning": "STORM",
    "rain": "RAIN",
    "rainbow": "CLOUDY",
    "cloudy": "CLOUDY",
    "shine": "CLEAR",
    "sunrise": "CLEAR",
    # "sandstorm" is deliberately absent: it maps to no SkyDex phenomenon, so
    # training on it would only teach a class that can never be acted on.
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def split_dataset(source: Path, train_root: Path, test_root: Path) -> None:
    """Deterministically split a class-labelled image folder by file.

    ``source`` holds one subdirectory per class. For each class, sort the
    filenames and alternate: even index -> ``train_root/<class>``, odd index
    -> ``test_root/<class>``. Copies rather than moves, so re-running is
    idempotent and the original archive extraction is left untouched.
    """
    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

        train_dest = train_root / class_dir.name.lower()
        test_dest = test_root / class_dir.name.lower()
        train_dest.mkdir(parents=True, exist_ok=True)
        test_dest.mkdir(parents=True, exist_ok=True)

        train_count = test_count = 0
        for index, path in enumerate(images):
            if index % 2 == 0:
                shutil.copy2(path, train_dest / path.name)
                train_count += 1
            else:
                shutil.copy2(path, test_dest / path.name)
                test_count += 1
        print(f"{class_dir.name}: {len(images)} images -> {train_count} train / {test_count} test")


def embed_folder(model: VisionModel, root: Path) -> tuple[np.ndarray, list[str]]:
    vectors, labels = [], []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        group = SOURCE_TO_GROUP.get(class_dir.name.lower())
        if group is None:
            print(f"skipping unmapped class {class_dir.name}")
            continue
        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        print(f"{class_dir.name} -> {group}: {len(images)} images")
        for path in images:
            try:
                vectors.append(model.embed(path.read_bytes())[0].numpy())
                labels.append(group)
            except Exception as error:  # a corrupt file must not stop a 6,900-image run
                print(f"  skipping {path.name}: {error}")
    return np.stack(vectors), labels


def train(source: Path, destination: Path) -> None:
    model = VisionModel()
    features, labels = embed_folder(model, source)
    print(f"\n{len(labels)} embeddings, distribution: {Counter(labels)}")

    classifier = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    classifier.fit(features, labels)

    predictions = classifier.predict(features)
    print("\n--- on the training set (optimistic by construction) ---")
    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions, labels=sorted(set(labels))))

    groups = list(classifier.classes_)
    unknown = [group for group in groups if group not in GROUP_ORDER]
    if unknown:
        raise SystemExit(f"refusing to save a probe with unknown groups: {unknown}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        groups=np.array(groups),
        weights=classifier.coef_,
        bias=classifier.intercept_,
    )
    print(f"\nwrote {destination} ({destination.stat().st_size} bytes)")


def main() -> None:
    command = sys.argv[1]
    if command == "split":
        split_dataset(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif command == "train":
        train(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(
            "usage: train_probe.py split <multiclass_source> <train_dir> <test_dir>\n"
            "       train_probe.py train <train_dir> <probe_path>"
        )


if __name__ == "__main__":
    main()
