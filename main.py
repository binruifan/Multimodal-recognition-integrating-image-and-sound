import os
import cv2
import numpy as np
import csv
import shutil
import librosa
from skimage.feature import local_binary_pattern
from moviepy.editor import VideoFileClip, concatenate_videoclips


def resize_clip(clip, target_resolution, target_fps, target_audio_fps):
    """
    Resize video clips to specified resolution, frame rate, and audio sample rate.

    parameter：
    - clip: VideoFileClip。
    - target_resolution: Target resolution (width, height) tuple.
    - target_fps: Target frame rate (e.g. 30 frames/s).
    - target_audio_fps: Target audio sample rate (e.g. 44100Hz)。


    """
    resized_clip = clip.resize(newsize=target_resolution)  # 调整视频分辨率
    resized_clip = resized_clip.set_fps(target_fps)  # 设置视频帧率
    resized_clip = resized_clip.set_audio(clip.audio.set_fps(target_audio_fps))  # 设置音频采样率
    return resized_clip


def merge_videos(video_files, output_file, target_resolution=(1280, 720), target_fps=30, target_audio_fps=44100):
    """
   Merge multiple video files into one video file and adjust to the same resolution, frame rate and audio sample rate.

    parameter：
    - video_files: List of video file paths.
    - output_file: The output merged video file path.
    - target_resolution: Target resolution, default is (1280, 720).
    - target_fps: Target frame rate, default is 30 frames per second.
    - target_audio_fps: Target audio sampling rate, default is 44100Hz.
    """
    clips = []  # Save adjusted video clips
    for file in video_files:
        clip = VideoFileClip(file)  # Load video files
        resized_clip = resize_clip(clip, target_resolution, target_fps, target_audio_fps)  # Adjust video parameters
        clips.append(resized_clip)

    # Merge all video clips into one video
    final_clip = concatenate_videoclips(clips, method="compose")
    # Write merged video to output file
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # Release resources and close the video file
    final_clip.close()
    for clip in clips:
        clip.close()


# Data preprocessing and cleaning video frames
def preprocess_video(video_path):
    """
    Preprocess the video by removing noise and normalizing it.

    - video_path: Video file path.

   return:
    - Cleaned frame list.
    """
    cap = cv2.VideoCapture(video_path)  # Open video file
    clean_frames = []  # Store cleaned frames
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Gaussian blur denoising
        normalized_frame = cv2.normalize(blurred_frame, None, 0, 255, cv2.NORM_MINMAX)  # normalization
        clean_frames.append(normalized_frame)  # Add processed frames to list
        frame_count += 1

    cap.release()  # Release the video capture object
    return clean_frames


def preprocess_audio(audio_path):
    """
    Preprocess audio files, remove silent parts and normalize.

    parameter:
    - audio_path: audio file path.

    return:
    - Preprocessed audio data and sample rate.
    """
    audio, sr = librosa.load(audio_path, sr=22050)  # Load audio files
    normalized_audio = librosa.util.normalize(clean_audio)  # Normalize audio data
    return normalized_audio, sr


# LBPFeature extraction
def extract_lbp(frame, radius=3, n_points=8 * 3):
    """
    Extract LBP features of video frames.

    parameter:
    - frame: a single video frame (grayscale image).
    - radius: radius in LBP algorithm.
    - n_points: the number of points in the LBP algorithm.

    return:
    - LBP feature histogram.
    """
    lbp = local_binary_pattern(frame, n_points, radius, method='uniform')  # Calculate LBP features
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))  # Calculate histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum()  # normalized histogram
    return lbp_hist


# Process video frames and extract LBP features every 10 frames
def process_video(video_path, output_folder, frames_per_file=10):
    """
    Extract LBP features for every 10 frames from the video and save them as CSV files.

    参数：
    - video_path: 视频文件路径。
    - output_folder: 保存特征文件的目录。
    - frames_per_file: 每个文件包含的帧数。

    返回：
    - 样本文件数。
    """
    clean_frames = preprocess_video(video_path)
    frame_count = 0
    lbp_features = []
    os.makedirs(output_folder, exist_ok=True)

    for frame in clean_frames:
        feature = extract_lbp(frame)
        lbp_features.append(feature)
        frame_count += 1

        if frame_count % frames_per_file == 0:
            output_file = f'{output_folder}/sam{frame_count // frames_per_file}_aa.csv'
            np.savetxt(output_file, lbp_features, delimiter=",")
            lbp_features = []

    if lbp_features:
        output_file = f'{output_folder}/sam_{(frame_count // frames_per_file) + 1}.csv'
        np.savetxt(output_file, lbp_features, delimiter=",")

    return frame_count // frames_per_file + (1 if lbp_features else 0)


# 提取音频特征
def extract_audio_features(video_path, audio_output_path, output_dir, num_samples, frames_per_sample=10, sr=22050):
    """
    从视频中提取音频特征（MFCC），每个样本包含10帧的音频。

    参数：
    - video_path: 视频文件路径。
    - audio_output_path: 输出的音频文件路径。
    - output_dir: 保存音频特征的目录。
    - num_samples: 样本数量。
    - frames_per_sample: 每个样本包含的帧数。
    - sr: 音频采样率。
    """
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)

    clean_audio, sr = preprocess_audio(audio_output_path)
    frame_duration = 1 / 30  # 每帧对应的时间长度
    sample_duration = frames_per_sample * frame_duration
    num_frames = int(sample_duration * sr)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        start_sample = int(i * num_frames)
        end_sample = int((i + 1) * num_frames)
        mfcc_list = []

        for j in range(frames_per_sample):
            frame_start = start_sample + int(j * frame_duration * sr)
            frame_end = start_sample + int((j + 1) * frame_duration * sr)
            mfcc = librosa.feature.mfcc(y=clean_audio[frame_start:frame_end], sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_list.append(mfcc_mean)

        sample_file = os.path.join(output_dir, f'sam{i + 1}.csv')
        with open(sample_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(mfcc_list)

    shutil.make_archive('audio_features', 'zip', output_dir)


# 特征融合
def combine_features(image_dir, audio_dir, output_dir, num_samples):
    """
    将图像特征和音频特征融合，并保存为CSV文件。

    参数：
    - image_dir: 图像特征目录。
    - audio_dir: 音频特征目录。
    - output_dir: 输出目录。
    - num_samples: 样本数量。
    """
    os.makedirs(output_dir, exist_ok=True)

    for sample_number in range(1, num_samples + 1):
        image_file = f'{image_dir}/sam{sample_number}_aa.csv'
        audio_file = f'{audio_dir}/sam{sample_number}.csv'
        image_features = np.loadtxt(image_file, delimiter=',')
        audio_features = np.loadtxt(audio_file, delimiter=',')

        combined_features = []
        for image_row, audio_row in zip(image_features, audio_features):
            combined_row = np.hstack([image_row, audio_row])
            combined_features.append(combined_row)

        combined_file = f'{output_dir}/sam{sample_number}.csv'
        with open(combined_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(combined_features)


# 运行整个流程
def main():
    video_files = [
        'Kiwi.mp4',
        '1515015853-1-16.mp4',
        'Kiwi birds. Male and female calls at Russell New Zealand.mp4',
        'Male Kiwi Calling.mp4'
    ]
    merged_video_path = 'merged_video.mp4'

    # 合并视频并保存到文件
    merge_videos(video_files, merged_video_path)

    image_output_folder = 'NeuCube/database/s1'
    audio_output_path = 'output_audio.wav'
    audio_output_folder = 'NeuCube/database/v1'
    combined_output_folder = 'NeuCube/database/c1'

    # 提取图像特征并获取样本数
    num_samples = process_video(merged_video_path, image_output_folder)

    # 提取音频特征
    extract_audio_features(merged_video_path, audio_output_path, audio_output_folder, num_samples)

    # 合并图像和音频特征
    combine_features(image_output_folder, audio_output_folder, combined_output_folder, num_samples)

    print("Complete video merging, feature extraction and fusion. The video output is: ", merged_video_path)


# 执行主程序
if __name__ == '__main__':
    main()
