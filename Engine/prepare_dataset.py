import os
import shutil
import zipfile

ARCHIVE_EXTENSIONS = (".zip", ".7z")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
MOVE_EXTENSIONS = IMAGE_EXTENSIONS + (".txt",)
JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
JUNK_DIRS = {"__macosx"}


class DatasetPrepareError(Exception):
    """Raised when dataset archives cannot be prepared."""


def _project_root():
    return os.getcwd()


def _dataset_path():
    return os.path.join(_project_root(), "dataset")


def _is_archive(name):
    return name.lower().endswith(ARCHIVE_EXTENSIONS)


def _is_dataset_file(name):
    return name.lower().endswith(MOVE_EXTENSIONS)


def _is_junk_name(name):
    return name.lower() in JUNK_NAMES


def _has_junk_dir(path, dataset_path):
    rel = os.path.relpath(path, dataset_path)
    return any(part.lower() in JUNK_DIRS for part in rel.split(os.sep))


def find_archives(dataset_path):
    archives = []
    for name in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, name)
        if os.path.isfile(full_path) and _is_archive(name):
            archives.append(full_path)
    return sorted(archives)


def wipe_non_archives(dataset_path):
    deleted = 0
    for name in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, name)
        if os.path.isfile(full_path) and _is_archive(name):
            continue
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            deleted += 1
        elif os.path.isfile(full_path):
            os.remove(full_path)
            deleted += 1
    return deleted


def extract_archives(archives, dataset_path):
    extracted = 0
    for archive_path in archives:
        name = os.path.basename(archive_path)
        print(f"Extracting {name}...")
        lower_name = name.lower()
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(dataset_path)
        elif lower_name.endswith(".7z"):
            import py7zr

            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                archive.extractall(path=dataset_path)
        else:
            continue
        extracted += 1
    return extracted


def flatten_dataset_files(dataset_path):
    moved = 0
    for root, dirs, files in os.walk(dataset_path):
        dirs[:] = [d for d in dirs if d.lower() not in JUNK_DIRS]
        if os.path.abspath(root) == os.path.abspath(dataset_path):
            continue
        if _has_junk_dir(root, dataset_path):
            continue

        for file_name in files:
            if _is_junk_name(file_name) or _is_archive(file_name):
                continue
            if not _is_dataset_file(file_name):
                continue

            src = os.path.join(root, file_name)
            dest = os.path.join(dataset_path, file_name)
            if os.path.abspath(src) == os.path.abspath(dest):
                continue
            if os.path.exists(dest):
                print(f"  Warning: overwriting existing file: {file_name}")
                os.remove(dest)
            shutil.move(src, dest)
            moved += 1
    return moved


def delete_empty_folders(dataset_path):
    removed = 0
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        if os.path.abspath(root) == os.path.abspath(dataset_path):
            continue

        for file_name in files:
            if _is_junk_name(file_name):
                os.remove(os.path.join(root, file_name))

        try:
            if not os.listdir(root):
                os.rmdir(root)
                removed += 1
        except OSError:
            continue
    return removed


def prepare_dataset():
    """Extract zip/7z archives in dataset/ and flatten files to the folder root."""
    dataset_path = _dataset_path()
    if not os.path.isdir(dataset_path):
        raise DatasetPrepareError("Error: dataset/ folder not found")

    archives = find_archives(dataset_path)
    if not archives:
        raise DatasetPrepareError("Error: zip/7z files not found in dataset/")

    print(f"Found {len(archives)} archive(s):")
    for archive_path in archives:
        print(f"  - {os.path.basename(archive_path)}")

    deleted = wipe_non_archives(dataset_path)
    print(f"Deleted {deleted} leftover file(s)/folder(s) (archives kept)")

    extracted = extract_archives(archives, dataset_path)
    print(f"Extracted {extracted} archive(s)")

    moved = flatten_dataset_files(dataset_path)
    print(f"Moved {moved} file(s) to dataset/")

    removed = delete_empty_folders(dataset_path)
    print(f"Removed {removed} empty folder(s)")

    print("Dataset preparation completed.")
    return True
