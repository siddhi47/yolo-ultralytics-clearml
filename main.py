from yolo_training.preprocess_raw import merge_annotations, video_to_image_frames


merge_annotations("raw_data/raw_data/annotations/extracted")
video_to_image_frames("raw_data/raw_data/videos", "data/images")
