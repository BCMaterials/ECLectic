# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 18:05:24 2025

@author: stephania.rodriguez
"""
# Needed libraries

import os
import re
import cv2  # OpenCV for video processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Functions for Potentiostat Files === #
# These functions handle the extraction and processing of potentiostat data files (they depend in your filename pattern)

def extract_video_number_from_filename(filename):
    """
    Extracts the video number from a filename using a regular expression.
    Returns the number as an integer, or None if not found.
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def process_potentiostat_file(filepath):
    """
    Reads a potentiostat CSV file, cleans the data, and returns it as a pandas DataFrame.
    Skips header rows, drops NaNs, and ensures numeric types.
    """
    try:
        # Read data from the file, skipping irrelevant rows
        data = pd.read_csv(filepath, skiprows=6, header=None, encoding='utf-16', names=["Time_s", "Current_uA"])

        # Drop rows with NaN values
        data = data.dropna(subset=["Time_s", "Current_uA"])
        
        # Convert columns to numeric
        data["Time_s"] = pd.to_numeric(data["Time_s"], errors='coerce')
        data["Current_uA"] = pd.to_numeric(data["Current_uA"], errors='coerce')

        print(f"Potentiostat data from {filepath}:")
        print(data.head())

        return data
    except Exception as e:
        print(f"Error processing potentiostat file {filepath}: {e}")
        return None

# === Functions for Video Processing === #
# These functions process video files to extract red intensity information.

def calculate_mean_red_intensity(image, mask=None):
    """
    Calculates the mean red channel intensity of an image.
    If a mask is provided, calculates the mean within the masked area.
    """
    if mask is None:
        mean_r = np.mean(image[:, :, 2])  # Red channel mean for the entire frame
    else:
        mean_r = cv2.mean(image[:, :, 2], mask=mask)[0]  # Red channel mean within the mask
    return mean_r

def process_video(video_path, exposure_rate, output_folder):
    """
    Processes a video file to extract time-resolved red intensity data.
    For each frame, calculates the mean red intensity (optionally in a masked region).
    Returns a DataFrame with time, intensity, and detection flag.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open the video file {video_path}.")
        return None

    frame_data = []
    frame_number = 0
    red_dot_detected = False  # Flag to indicate if red dot is detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        time_seconds = frame_number * exposure_rate-1 #time_seconds = frame_number * exposure_rate  # Calculate time

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        radius = min(height, width) // 4  # Adjust radius

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate overall frame intensity
        intensity = np.mean(gray_frame)

        # Check if the frame is dark
        if intensity < 5:  # Adjust this threshold to suit your videos
            # If it's dark, measure the intensity of the full frame
            mean_r = calculate_mean_red_intensity(frame)
            red_dot_detected = False  # Reset the flag

        else:
            # Create a circular mask for the red dot area
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # Create the mask
            mean_r = calculate_mean_red_intensity(frame, mask)
            red_dot_detected = True  # Indicate red dot is detected
            
            if frame_number == 50:  # Save only for the first non-dark frame
                # Display the mask using Matplotlib
                plt.figure(figsize=(6, 6))
                plt.imshow(mask, cmap="gray")
                plt.title("Circular Mask")
                plt.axis("off")
                plt.show()

        # Store the data
        frame_data.append([time_seconds, mean_r, red_dot_detected])

    cap.release() 

    # Convert to DataFrame
    df = pd.DataFrame(frame_data, columns=["Time_s", "Red_Intensity", "Red_Dot_Detected"])
    df["Normalized_Red"] = (df["Red_Intensity"] / 255) * 100  # Normalize red intensity
    print(f"Processed video data from {video_path}:")
    print(df.head())
    return df

# === Combined Processing and Plotting === #
# This function coordinates the matching, processing, and plotting of potentiostat and video data.

def process_and_plot(potentiostat_folder, video_folder, output_folder, exposure_rate=1/5):
    """
    Processes all potentiostat and video files in the specified folders.
    Matches files by video number, merges data, saves combined results, and generates plots.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get potentiostat and video files
    potentiostat_files = [f for f in os.listdir(potentiostat_folder) if f.endswith('.csv')]
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    for video_file in video_files:
        video_number = extract_video_number_from_filename(video_file)
        if video_number is None:
            continue

        # Match potentiostat file by video number
        matching_potentiostat_file = next(
            (f for f in potentiostat_files if str(video_number) in f), None
        )
        if not matching_potentiostat_file:
            print(f"No matching potentiostat file found for video {video_file}.")
            continue

        # Process potentiostat file
        potentiostat_path = os.path.join(potentiostat_folder, matching_potentiostat_file)
        potentiostat_data = process_potentiostat_file(potentiostat_path)

        # Process video
        video_path = os.path.join(video_folder, video_file)
        video_data = process_video(video_path, exposure_rate, output_folder)

        if potentiostat_data is None or video_data is None:
            print(f"Skipping processing for video {video_file}.")
            continue

        # Merge and save combined data
        combined_df = pd.merge_asof(
            potentiostat_data.sort_values("Time_s"), 
            video_data.sort_values("Time_s"), 
            on="Time_s", 
            direction="nearest"
        )
        
        combined_df_path = os.path.join(output_folder, f"combined_data_{video_number}.xlsx")
        combined_df.to_excel(combined_df_path, index=False)
        
        print(f"Saved combined data to {combined_df_path}.")

        # Plot data
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot potentiostat data (current)
        ax1.set_xlabel("Time (s)", fontsize=30)
        ax1.set_ylabel("Current (µA)", color="tab:blue", fontsize=30)
        ax1.plot(combined_df["Time_s"], combined_df["Current_uA"], color="tab:blue", label="Current (µA)", linewidth=3)
        ax1.tick_params(axis= "y", labelcolor="tab:blue", labelsize=30)
        ax1.tick_params(axis='both', which='major', labelsize=30)

        # Plot video red intensity
        ax2 = ax1.twinx()
        ax2.set_ylabel("Normalized Red Intensity (%)", color="tab:red", fontsize=30)
        ax2.plot(combined_df["Time_s"], combined_df["Normalized_Red"], color="tab:red", label="Red Intensity (%)", linewidth=3)
        ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=30)
        
        plot_path = os.path.join(output_folder, f"plot_{video_number}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}.")

# === Main Execution === #
# Set up input/output folders and run the processing pipeline.

potentiostat_folder = "Replace with path to your folder"
video_folder = potentiostat_folder 
output_folder = potentiostat_folder 

process_and_plot(potentiostat_folder, video_folder, output_folder, exposure_rate=1/5)  # Modify to your actual exposure rate






