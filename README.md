# ECLectic. WP3 Task 3.1. Gamma Correction and Color Space Transformations for Quantitative Analysis of Electrochemiluminescence Images Using Smartphone Cameras

**Dataset Authors:**  
- Stephania Rodríguez Muiña (BCMaterials and University of the Basque Country-UPV/EHU), ORCID: 0009-0006-3196-8964  
- Rajendra Kumar Reddy Gajjala (BCMaterials), ORCID: 0000-0002-9193-5295  
- Francisco Javier del Campo García (BCMaterials), ORCID: 0000-0002-3637-5782

**Contact Person:**  
Francisco Javier del Campo García (BCMaterials), ORCID: 0000-0002-3637-5782

**Publication Year:** 2025 

**Project:** ECLectic, funded by European Union, MSCA Doctoral Network Horizon Europe programme. Grant Agreement num. 101119951

## Abstract

This dataset includes Python scripts designed to analyze electrochemiluminescence (ECL) data, perform color space transformations, apply gamma corrections, and generate both linear and non-linear calibration curves via data fitting. These tools enable synchronized processing of electrochemical signals and ECL video recordings for quantitative analysis.

## Folder Structure

**Scripts/**
- `analyze_video_frames_rgb_intensity.py`  
  Extracts ECL signal intensity from videos using a user-defined mask (ROI). Calculates and stores the mean and standard deviation of RGB pixel values for the selected region.

- `color_code_transformations.py`  
  Transforms BT.709 RGB data into other color spaces (XYZ, CIELAB, sRGB). Applies gamma correction (user-defined gamma) and identifies perceptual thresholds to segment and plot piecewise linear and non-linear calibration models. Includes visualization of fitted curves across all color spaces for quantitative comparison.

- `combine_by_time_video_potentiostat.py`  
  Synchronizes video-derived ECL intensity data with chronoamperometry time series. Aligns red-channel intensity from the ROI with corresponding electrochemical measurements for integrated analysis.

## File Specifics

- All `.py` files are Python 3 scripts developed and tested using Spyder 6.0.5 within the Anaconda distribution.
- Scripts are structured to process and analyze ECL video and electrochemical data for quantitative evaluation.
- File type: `.py` (UTF-8 encoded)
- Language: Python 3.12.3

