# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:40:49 2025

@author: stephania.rodriguez
"""
# Needed libraries
import os
import re
import cv2
import numpy as np
import pandas as pd

# Extract video number from filename (the definition structure will depend on your video file name pattern).
def extract_video_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

# Parse potentiostat filename to extract experimental parameters (the definition structure will depend on your file name pattern).
def parse_potentiostat_filename(filename):
    patterns = [
        r'(\d+)E(\d+)_(\d+)_(\d+)_(\d+)uM_(\d+)',
        r'(\d+)E(\d+)_(\d+)_(\d+)_(\d+)_(\d+)uM_(\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            concentration = float(f"{groups[4]}.{groups[5]}") if len(groups) > 6 else float(groups[4])
            return {
                'experiment': int(groups[0]),
                'electrode': int(groups[1]),
                'potential': float(f"{groups[2]}.{groups[3]}"),
                'concentration': concentration,
                'video': int(groups[-1])
            }
    return None #return None if pattern doesn´t match

# Analyze a single video frame: apply circular mask withing the region of interest (ROI), calculate mean and standard deviation (std) values of RGB channels.
def analyze_frame(frame):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)
    mean_r = cv2.mean(frame[:, :, 2], mask=mask)[0]
    std_r = np.std(frame[:, :, 2][mask > 0])
    mean_g = cv2.mean(frame[:, :, 1], mask=mask)[0]
    std_g = np.std(frame[:, :, 1][mask > 0])
    mean_b = cv2.mean(frame[:, :, 0], mask=mask)[0]
    std_b = np.std(frame[:, :, 0][mask > 0])
    return {'R': mean_r, 'std R': std_r, 'G': mean_g, 'std G': std_g, 'B': mean_b, 'std B': std_b}

# Process video: iterate over video frames, keeping the frame with max R channel intensity.
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    max_r = -1
    result = None
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_data = analyze_frame(frame)
        if frame_data['R'] > max_r:
            max_r = frame_data['R']
            result = frame_data
    cap.release()
    return result

# Main function to generate Excel report from potentiostat data and corresponding video analysis. It generates an excel file with RGB mean and std data of the maximum signal frame. 
def generate_report(potentiostat_dir, video_dir, output_file):
    video_map = {}
    for f in os.listdir(video_dir):
        vid_num = extract_video_number(f)
        if vid_num is not None:
            video_map[vid_num] = os.path.join(video_dir, f)
    # Use a set to track unique (video, concentration) pairs
    seen = set()
    results = []
    for f in os.listdir(potentiostat_dir):
        params = parse_potentiostat_filename(f)
        if not params or params['video'] not in video_map:
            continue
        key = (params['video'], params['concentration'])
        if key in seen:
            continue  # Skip duplicates
        seen.add(key)
        video_data = process_video(video_map[params['video']])
        if video_data:
            row = {
               'Video': params['video'],
               'Concentration(uM)': params['concentration']
           }
            row.update(video_data)
            results.append(row)
    # Sort by concentration
    df = pd.DataFrame(results)
    df['Concentration(uM)'] = pd.to_numeric(df['Concentration(uM)'], errors='coerce')
    df = df.sort_values(by='Concentration(uM)')
    df.to_excel(output_file, index=False)
    print(f"Report generated: {output_file}")

# Example usage: customize with your actual folder paths
if __name__ == "__main__":
    generate_report(
        potentiostat_dir="Path to your potentiostat data", # Replace with your actual path
        video_dir="Path to your video data",               # Replace with your actual path
        output_file="Output path to save RGBresults.xlsx"  # Replace with your output file path
    )
