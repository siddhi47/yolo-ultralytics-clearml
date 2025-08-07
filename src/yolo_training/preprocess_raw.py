import os
import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

with open("ALLOWED_CLASS.txt", "r") as f:
    classes = f.read().strip().split("\n")
CLASS_DICT = {c: idx for idx, c in enumerate(classes)}


def extract_zip_file(zip_path: str, extract_to: str):
    """Extracts a single ZIP file to the target directory."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=extract_to)
    except zipfile.BadZipFile:
        print(f"Warning: Failed to unzip {zip_path} (Bad zip file)")


def export_zip(path: str, num_threads: int = 8):
    """
    Unzips all .zip files in the given directory using multithreading.
    Extracted files go into: <path>/extracted/<zip_file_name_without_ext>
    """
    # List all .zip files in the directory
    list_zip_files = [f for f in os.listdir(path) if f.endswith(".zip")]

    # Target root extraction path
    unzip_root_path = os.path.join(path, "extracted")
    os.makedirs(unzip_root_path, exist_ok=True)

    # Create tasks for each zip file
    tasks = []
    for zip_file in list_zip_files:
        zip_path = os.path.join(path, zip_file)
        extract_to = os.path.join(unzip_root_path, os.path.splitext(zip_file)[0])
        os.makedirs(extract_to, exist_ok=True)
        tasks.append((zip_path, extract_to))

    # Run multithreaded extraction
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(lambda args: extract_zip_file(*args), tasks),
                total=len(tasks),
                desc="Unzipping files",
            )
        )

    return unzip_root_path


def copy_annotations(
    annot_file, destination_folder, annot_type, folder, tmp_class_map, split=True
):
    """
    Copies annotation files to the destination folder.
    """
    try:
        if not os.path.exists(os.path.join(destination_folder, annot_type)):
            os.makedirs(os.path.join(destination_folder, annot_type))
        annot_df = pd.read_csv(annot_file, sep=" ", header=None)
        annot_df[0] = annot_df[0].map(
            lambda x: CLASS_DICT.get(tmp_class_map[folder][x]), None
        )

        # drop rows with NaN values in the first column
        annot_df = annot_df.dropna(subset=[0])

        if split:
            save_path = os.path.join(
                destination_folder,
                annot_type,
                str(folder) + "_" + annot_file.split(os.sep)[-1],
            )
        else:
            save_path = os.path.join(
                destination_folder,
                str(folder) + "_" + annot_file.split(os.sep)[-1],
            )
        annot_df.to_csv(
            save_path,
            sep=" ",
            header=None,
            index=False,
        )
    except Exception as e:
        print(f"Error processing {annot_file}: {e}")


def merge_annotations(annotation_path):
    if not os.path.exists("data/labels"):
        os.makedirs("data/labels")

    annotation_folders = os.listdir(annotation_path)
    print(annotation_folders)
    tmp_class_map = defaultdict()
    for folder in annotation_folders:
        obj_name_file = os.path.join(annotation_path, folder, "obj.names")
        with open(obj_name_file, "r") as f:
            classes = f.read().strip().split("\n")
        tmp_class_map[folder] = {idx: c for idx, c in enumerate(classes)}

        train_valid_files = defaultdict(list)

        if os.path.exists(os.path.join(annotation_path, folder, "obj_Validation_data")):
            validation_annotation_files = [
                os.path.join("obj_Validation_data", f)
                for f in os.listdir(
                    os.path.join(annotation_path, folder, "obj_Validation_data")
                )
                if f.endswith(".txt")
            ]
            train_valid_files["valid"].extend(validation_annotation_files)
            with open("split.txt", "w") as f:
                f.write(f"{folder}, valid\n")

        if os.path.exists(os.path.join(annotation_path, folder, "obj_train_data")):
            training_annotation_files = [
                os.path.join("obj_train_data", f)
                for f in os.listdir(
                    os.path.join(annotation_path, folder, "obj_train_data")
                )
                if f.endswith(".txt")
            ]
            train_valid_files["train"].extend(training_annotation_files)
            with open("split.txt", "a") as f:
                f.write(f"{folder}, train\n")

        for annot_type, file_list in train_valid_files.items():
            file_list = [os.path.join(annotation_path, folder, f) for f in file_list]
            for fi in file_list:
                if os.path.exists(fi):
                    copy_annotations(
                        fi, "data/labels", annot_type, folder, tmp_class_map
                    )

    return os.path.join("data/labels")


def video_to_frames(video_path, output_path):
    """
    converts a video file to image frames and saves them in the specified output path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        output_frame_path = os.path.join(
            output_path, f"{video_name}_frame_{frame_number:06d}.jpg"
        )

        cv2.imwrite(output_frame_path, frame)

    video.release()


def video_to_image_frames(video_files_path, output_path, num_threads=8, split=True):
    """
    Converts video files to image frames using multithreading.
    """

    split_type = {}
    if split:
        with open("split.txt", "r") as f:
            for line in f:
                split_type[line.split(",")[0]] = line.split(",")[1].strip()

    video_files = [f for f in os.listdir(video_files_path) if f.endswith(".mp4")]

    if split:
        tasks = [
            (
                os.path.join(video_files_path, vf),
                os.path.join(output_path, split_type[vf.split(".")[0]]),
            )
            for vf in video_files
        ]
    else:
        tasks = [
            (os.path.join(video_files_path, vf), output_path) for vf in video_files
        ]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(lambda args: video_to_frames(*args), tasks),
                total=len(tasks),
                desc="Converting videos to frames",
            )
        )
