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

# Running this file as a script puts `training/` on sys.path[0], not the repo
# root, so the `from app...` imports below fail with ModuleNotFoundError before
# argument parsing has even started. Every command in this module's docstring,
# in README.md's retrain section and in load_cache_or_raise's error message
# invokes it exactly that way -- including the one that fires when a reader is
# already stuck on a stale cache -- so the bootstrap belongs here rather than
# in a PYTHONPATH the four call sites would each have to remember. The notebook
# does the same thing in its first cell.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import MODEL_ARCHITECTURE, MODEL_WEIGHTS, VisionModel  # noqa: E402
from app.prompts import GROUP_ORDER  # noqa: E402

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
# recompressed or resized", and `_deduplicate` drops on that distance alone --
# it performs no pixel comparison of any kind.
#
# The tier is 3 rather than an arbitrary guess because an offline review of
# this exact dataset measured this threshold's false-positive rate at 43%: of
# the pairs dHash<=3 calls duplicates, pixel comparison showed nearly half to
# be distinct photographs, not a recompression or resize of the same one. That
# measurement is not a step the code repeats -- a removal here is a dHash match
# and nothing stronger -- and the 43% is accepted rather than tightened because
# the error is one-directional: this tier only ever removes, it never lets a
# true duplicate leak across the split. A false positive costs one discarded
# photograph; over-removal on this pool cost a few hundred images out of 6,656
# and moved 5-fold CV by 0.0000, while under-removal would leak a photograph
# across the split and inflate every number after it.
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


def _fingerprint_tree(root: Path) -> str:
    """A cheap fingerprint of everything the cached arrays are computed from.

    The cache stores ``(features, labels)``, and three separate things decide
    what those arrays contain. All three are hashed here, because a cache hit
    is a promise that recomputing would produce the same arrays:

    1. **The image tree.** Sorted ``(relative_path, size, mtime_ns)`` triples,
       which change whenever a file is added, removed or replaced (a replace
       changes size and/or mtime; a rename changes the path) -- exactly the set
       of edits `split_dataset` can make to `data/train`. Deliberately not the
       file *contents*: this runs over thousands of images on every `train`
       invocation and re-reading all of them would defeat the point of caching.
    2. **SOURCE_TO_GROUP**, which decides the labels. `embed_folder` consults
       it per class, so remapping one source class -- `rainbow` from CLOUDY to
       CLEAR, say, or un-dropping `sandstorm`; both are judgement calls sitting
       right there in the table -- changes what the cached labels mean while
       leaving the tree byte-identical.
    3. **The CLIP checkpoint**, which decides the features. Bumping
       MODEL_ARCHITECTURE or MODEL_WEIGHTS makes every cached embedding a
       vector from a different model in a different space.
    4. **IMAGE_SUFFIXES**, which decides which files in the tree are even
       candidates for embedding. `embed_folder` filters on it per class, so
       widening or narrowing the set -- adding `.webp`, say -- changes which
       files a cache hit is silently standing in for, while the tree entries
       above (which enumerate *every* file, filtered or not) stay identical.

    Without 2, 3 and 4 the fingerprint matched, `_load_cache` printed
    `cache hit`, and the retrain silently refitted on stale labels or stale
    embeddings -- three times caught in review, which is why this docstring
    enumerates rather than summarises. Anything new that feeds `embed_folder`
    belongs in this list.
    """
    entries = sorted(
        (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    )
    payload = repr(
        (
            entries,
            sorted(SOURCE_TO_GROUP.items()),
            MODEL_ARCHITECTURE,
            MODEL_WEIGHTS,
            sorted(IMAGE_SUFFIXES),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(source: Path, cache_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """Return cached ``(features, labels)`` if ``cache_path`` exists and its
    stored fingerprint matches ``source``'s current tree, else None.

    Never raises for a missing or stale cache -- what to do about that (embed,
    or refuse) is the caller's decision. Always prints why it returned what it
    returned, so a stale cache is never reused silently.
    """
    fingerprint = _fingerprint_tree(source)

    if not cache_path.exists():
        print(f"cache miss: {cache_path} does not exist")
        return None

    payload = np.load(cache_path, allow_pickle=False)
    cached_fingerprint = str(payload["fingerprint"]) if "fingerprint" in payload.files else None

    if cached_fingerprint != fingerprint:
        reason = (
            "cache has no fingerprint (written before cache invalidation existed)"
            if cached_fingerprint is None
            else f"{source} has changed since the cache was written"
        )
        print(f"cache miss: {cache_path} is stale -- {reason}; recomputing")
        return None

    labels = [str(label) for label in payload["labels"]]
    print(f"cache hit: {cache_path} fingerprint matches {source} ({len(labels)} embeddings)")
    return payload["features"], labels


def load_cache_or_raise(source: Path, cache_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load ``cache_path`` for ``source``, refusing a missing or stale cache.

    For consumers -- like the notebook's cross-validation cell -- that must
    not silently train or cross-validate on stale embeddings, but also must
    not pay the cost of re-embedding thousands of images inline just to
    validate a cache. If the cache is missing or stale, the fix is to re-run
    `train_probe.py train`, not to embed here.
    """
    cached = _load_cache(source, cache_path)
    if cached is None:
        raise SystemExit(
            f"{cache_path} is missing or stale for {source}; run "
            "`.venv/bin/python training/train_probe.py train data/train data/probe.npz` "
            "to refresh it before trusting this result"
        )
    return cached


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

    Re-running is idempotent with respect to files gained or replaced in the
    source since the last run (either of which would otherwise shift what an
    index-based split lands on the wrong side, quietly recreating the exact
    contamination this function exists to prevent): before writing, every
    existing destination file whose name is present anywhere in the *current*
    source listing is removed, then re-copied fresh from this run's split.
    This is scoped to the current source listing's filenames rather than a
    blanket ``rmtree`` because a destination class directory can be shared
    with an unrelated dataset under the same class name -- ``rain`` here is
    also fed directly from the Weather Image Recognition dataset, which must
    be left untouched.

    This does NOT cover a file being *removed* from the source between runs:
    a destination copy whose name is no longer present anywhere in the source
    listing is not cleared, and survives as a stale leftover from the earlier
    run. Removing that guarantee, too, would require distinguishing "this name
    used to be in this source and is gone" from "this name has always
    belonged to the other dataset sharing this class directory" -- the two
    look identical from inside this function, since nothing records what a
    previous run wrote. The Kaggle archives this function reads are static
    downloads that do not lose files between runs, so this gap is not
    currently reachable in practice; if that ever changes, the safe fix is an
    explicit manifest of what this function wrote last time, not a broader
    filename-based clear.
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
            # Materialise the listing before deleting: Path.iterdir() is a
            # lazy os.scandir generator, and unlinking entries while it is
            # still being consumed can skip entries in the same readdir pass.
            for existing in list(dest.iterdir()):
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
    """Reuse a cached embedding of ``source`` if it exists and still matches
    ``source``'s current tree (see ``_fingerprint_tree``), else (re)compute
    and cache it.

    ``model_factory`` is only called on a cache miss, so a cache hit does not
    pay even the cost of loading CLIP.
    """
    cached = _load_cache(source, cache_path)
    if cached is not None:
        return cached

    features, labels = embed_folder(model_factory(), source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, labels=np.array(labels), fingerprint=_fingerprint_tree(source))
    print(f"cached {len(labels)} embeddings to {cache_path}")
    return features, labels


def make_classifier() -> LogisticRegression:
    """The estimator the shipped probe is fitted with.

    A factory rather than a literal because `training/train.ipynb` cross-validates
    this same configuration. When the two were spelled out separately, the
    notebook's 5-fold number silently described an estimator nobody shipped.

    ``class_weight="balanced"`` is the load-bearing argument. The training pool
    is SNOW 2,420 / FOG 2,024 / STORM 968 / RAIN 621 / CLOUDY 362 / CLEAR 261:
    FOG and SNOW are 67% of it, CLEAR is 3.9%. A multinomial logistic regression
    fitted without weights inherits those priors, and inheriting them is correct
    only if the deployment sees the same distribution — which it does not. Users
    photograph the sky in front of them, not a Kaggle sample. Unweighted, the
    probe pushed ambiguous real photographs toward FOG and SNOW hard enough that
    the backend's contradiction matrix (which admits only FOG for FOG and only
    SNOW for SNOW) refused an honest photograph of a clear blue sky. Weighting
    each class inversely to its frequency states the prior we actually mean:
    none. See tests/test_phenomenon_golden.py for the measurement.

    ``multi_class`` is deliberately not passed. It was previously set to
    "multinomial", which is what lbfgs already does for more than two classes
    and what the default selects here; passing it explicitly only earns a
    deprecation warning from scikit-learn 1.5. Verified identical coefficients
    on this pool with and without it.
    """
    return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")


def train(source: Path, destination: Path) -> None:
    features, labels = load_or_embed(VisionModel, source, TRAIN_CACHE_PATH)
    print(f"\n{len(labels)} embeddings, distribution: {Counter(labels)}")

    classifier = make_classifier()
    classifier.fit(features, labels)

    predictions = classifier.predict(features)
    print("\n--- on the training set (optimistic by construction) ---")
    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions, labels=sorted(set(labels))))

    groups = list(classifier.classes_)
    unknown = [group for group in groups if group not in GROUP_ORDER]
    if unknown:
        raise SystemExit(f"refusing to save a probe with unknown groups: {unknown}")

    # `classes_` is whatever the pool happened to contain, so a `data/train`
    # with no `hail` and no `lightning` fits five classes and would otherwise
    # write a five-group probe that answers the documented six-key
    # `phenomenon_scores` contract with five keys. `load_probe` refuses such a
    # file; refuse to create it here too, where the operator can still see why.
    missing = [group for group in GROUP_ORDER if group not in groups]
    if missing:
        raise SystemExit(
            f"refusing to save a probe missing {missing}: {source} has no images "
            f"for the source classes that map to those groups (see SOURCE_TO_GROUP)"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        groups=np.array(groups),
        weights=classifier.coef_,
        bias=classifier.intercept_,
    )
    print(f"\nwrote {destination} ({destination.stat().st_size} bytes)")


USAGE = (
    "usage: train_probe.py split <multiclass_source> <train_dir> <test_dir>\n"
    "       train_probe.py train <train_dir> <probe_path>"
)


def main() -> None:
    # Length-checked before indexing, and each branch checked before indexing
    # its own arguments. Running the script bare is the commonest mistake and
    # used to answer `IndexError: list index out of range` instead of the usage
    # string written directly below for exactly that moment.
    if len(sys.argv) < 2:
        raise SystemExit(USAGE)

    command = sys.argv[1]
    if command == "split" and len(sys.argv) == 5:
        split_dataset(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif command == "train" and len(sys.argv) == 4:
        train(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main()
