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

A note on why splitting alone is not enough: the Multi-class Weather Dataset
ships the same photograph under multiple filenames -- some byte-identical,
some a recompressed near-copy. Alternating a sorted file list without removing
these first places both copies of such a pair on opposite sides of the split
with near-certainty, which manufactures exactly the leakage the split exists
to prevent. _deduplicate() below removes exact (md5) and near (dHash)
duplicates within each class *before* alternating, so every surviving file is
content-distinct from every other file in that class on both sides.
"""

import hashlib
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
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

# A dHash Hamming distance at or below this is treated as "the same photograph,
# recompressed or resized". Matches the tier a code review confirmed against
# this exact dataset (32x32 RGB mean-abs-diff < 6 against a same-class random-pair
# baseline of ~59.8), rather than an arbitrary guess.
NEAR_DUPLICATE_DHASH_DISTANCE = 3

# The slow part of this whole pipeline is embedding images on CPU (minutes);
# fitting -- or cross-validating -- the classifier over already-computed
# features is seconds. Caching here lets a retrain or a cross-validation pass
# skip straight to the fast part. Gitignored by the existing `*.npz` rule.
TRAIN_CACHE_PATH = Path("data/train_embeddings.npz")


def _content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _dhash(path: Path, hash_size: int = 8) -> int:
    """An 8x8 difference hash: robust to the mild recompression or resizing
    that turns one duplicate photograph into a byte-different but
    visually-identical second file, unlike the exact md5 check above."""
    with Image.open(path) as image:
        pixels = list(
            image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS).getdata()
        )
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _deduplicate(images: list[Path]) -> list[Path]:
    """Keep one file per distinct photograph, in stable sorted order.

    First pass removes exact byte-for-byte duplicates (md5). Second pass
    removes near-duplicates (dHash within NEAR_DUPLICATE_DHASH_DISTANCE of a
    file already kept) -- this is what catches the same photo re-saved at a
    different quality or size under a different filename.
    """
    seen_md5: set[str] = set()
    kept_hashes: list[int] = []
    unique: list[Path] = []

    for path in images:
        digest = _content_hash(path)
        if digest in seen_md5:
            continue
        seen_md5.add(digest)

        fingerprint = _dhash(path)
        if any(_hamming(fingerprint, other) <= NEAR_DUPLICATE_DHASH_DISTANCE for other in kept_hashes):
            continue

        kept_hashes.append(fingerprint)
        unique.append(path)

    return unique


def split_dataset(source: Path, train_root: Path, test_root: Path) -> None:
    """Deterministically split a class-labelled image folder by file.

    ``source`` holds one subdirectory per class. For each class, images are
    deduplicated (see _deduplicate) and the survivors sorted and alternated:
    even index -> ``train_root/<class>``, odd index -> ``test_root/<class>``.
    Copies rather than moves, so the original archive extraction is left
    untouched.

    Re-running is idempotent even if the source directory has gained or lost
    files since the last run (which would otherwise shift every later index
    and leave a stale copy from the old split sitting on the wrong side, quietly
    recreating the exact contamination this function exists to prevent):
    before writing, any existing destination file whose name appears anywhere
    in the *current* source listing is removed. This is scoped to this
    source's own filenames rather than a blanket ``rmtree`` because a
    destination class directory can be shared with an unrelated dataset under
    the same class name -- ``rain`` here is also fed directly from the Weather
    Image Recognition dataset, which must be left untouched.
    """
    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        source_names = {p.name for p in images}

        unique = _deduplicate(images)
        removed = len(images) - len(unique)

        train_dest = train_root / class_dir.name.lower()
        test_dest = test_root / class_dir.name.lower()
        train_dest.mkdir(parents=True, exist_ok=True)
        test_dest.mkdir(parents=True, exist_ok=True)

        for dest in (train_dest, test_dest):
            for existing in dest.iterdir():
                if existing.name in source_names:
                    existing.unlink()

        train_count = test_count = 0
        for index, path in enumerate(unique):
            if index % 2 == 0:
                shutil.copy2(path, train_dest / path.name)
                train_count += 1
            else:
                shutil.copy2(path, test_dest / path.name)
                test_count += 1
        print(
            f"{class_dir.name}: {len(images)} images, {removed} duplicate(s) removed, "
            f"{len(unique)} unique -> {train_count} train / {test_count} test"
        )


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


def load_or_embed(model_factory, source: Path, cache_path: Path) -> tuple[np.ndarray, list[str]]:
    """Reuse a cached embedding of ``source`` if one exists, else compute and cache it.

    ``model_factory`` is only called on a cache miss, so a cached run does not
    pay even the cost of loading CLIP.
    """
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        labels = [str(label) for label in payload["labels"]]
        print(f"loaded {len(labels)} cached embeddings from {cache_path}")
        return payload["features"], labels

    features, labels = embed_folder(model_factory(), source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, labels=np.array(labels))
    print(f"cached {len(labels)} embeddings to {cache_path}")
    return features, labels


def train(source: Path, destination: Path) -> None:
    features, labels = load_or_embed(VisionModel, source, TRAIN_CACHE_PATH)
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
