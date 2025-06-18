# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:32:27 2025

@author: stephania.rodriguez
"""
# Needed libraries
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score

    #--------Colour space transformations-----------

# Linearization from RGB BT709 to linear rgb: Inverse BT.709 OETF

def bt709_to_linear(RGB, gamma=2.35): # Use your gamma digital value

    return np.where(RGB < 0.081, RGB / 4.5, ((RGB + 0.099) / 1.099) ** gamma)

# Transformation from rgb to XYZ (D65): Transformation matrix
BT709_TO_XYZ = np.array([
     [0.4124, 0.3576, 0.1805],
     [0.2126, 0.7152, 0.0722],
     [0.0193, 0.1192, 0.9505]])

def rgb_to_xyz(row):
    rgb = np.array([row['r'], row['g'], row['b']])
    return pd.Series(BT709_TO_XYZ @ rgb, index=['X', 'Y', 'Z'])

# Transformation from XYZ to lab: CIELAB OETF
def xyz_to_lab(row):
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883  # D65 reference white
    threshold = (6/29)**3

    def f(i):
        return i ** (1/3) if i > threshold else (((1/3)*(29/6)**2) * i) + (16 / 116)

    x_xn, y_yn, z_zn = row['X'] / Xn, row['Y'] / Yn, row['Z'] / Zn

    fx, fy, fz = f(x_xn), f(y_yn), f(z_zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return pd.Series([L, a, b, x_xn, y_yn, z_zn], index=['L*', 'a*', 'b*', 'x_xn', 'y_yn', 'z_zn'])

# Transformation from rgb to sRGB: sRGB OETF
def linear_to_sRGB(RGB, gamma=2.4):
    return np.where(RGB < 0.0031308, RGB * 12.92, (1.055 * (RGB ** (1 / gamma)) - 0.055))

    #----------- Color Channel vs Concentration Curve Fitting-------------

# RGB Transfer Function with continuity enforced by computing c 

def RGB_transfer(value, gamma, b, d, T):
    c = (b * T + d) / (T ** (1 / gamma))
    return np.where(value < T, b * value, c * value ** (1 / gamma) - d)

def RGB_transfer_T(T_fixed):
    def RGB_fit(concentration, gamma, b, d):
        return RGB_transfer(concentration, gamma, b, d, T_fixed)
    return RGB_fit

def compute_c_and_error_3params(popt, pcov, T):
    """Compute c and error for 3-parameter fit: gamma, b, d"""
    gamma, b, d = popt
    c = (b * T + d) / (T ** (1 / gamma))
    partial_c_gamma = (-np.log(T) * (b * T + d)) / (gamma ** 2 * T ** (1 / gamma))
    partial_c_b = T / (T ** (1 / gamma))
    partial_c_d = 1 / (T ** (1 / gamma))
    c_error = np.sqrt(
        (partial_c_gamma ** 2) * pcov[0, 0] +
        (partial_c_b ** 2) * pcov[1, 1] +
        (partial_c_d ** 2) * pcov[2, 2] +
        2 * partial_c_b * partial_c_d * pcov[1, 2] +
        2 * partial_c_gamma * partial_c_b * pcov[0, 1] +
        2 * partial_c_gamma * partial_c_d * pcov[0, 2]
    )
    return c, c_error

def fit_rgb_channel(x, y, T, label=None, verbose=True):
    """Fit RGB transfer function and return gamma, b, d, their errors, c, c_error, R2, T."""
    fitRGB = RGB_transfer_T(T)
    popt, pcov = curve_fit(fitRGB, x, y, p0=[2.2, 0.01, 0.0], bounds=([1.0, -1, -1], [3, 3, 3]), maxfev=10000)
    gamma, b, d = popt
    gamma_err, b_err, d_err = np.sqrt(np.diag(pcov))
    y_pred = fitRGB(x, gamma, b, d)
    r2 = r2_score(y, y_pred)
    c, c_err = compute_c_and_error_3params(popt, pcov, T)
    return gamma, gamma_err, b, b_err, d, d_err, c, c_err, r2, T

# Linear fit (Used for r and CIEXYZ)
def fit_linear(x, b):
    return b * x
def fit_linear_channel(x, y, label=None, verbose=True):
    """Fit y = a*x and return a, a_err, R2."""
    popt, pcov = curve_fit(fit_linear, x, y, p0=[1.0])
    a = popt[0]
    a_err = np.sqrt(pcov[0, 0])
    r2 = r2_score(y, fit_linear(x, a))
    return a, a_err, r2

# Lab Transfer Function approximation with continuity enforced by computing c 
def lab_transfer(value, b, d, T):
    c = (b * T + d) / (T ** (1 / 3))
    return np.where(value < T, b * value, c * value ** (1/3)-d)

def lab_transfer_T(T_fixed):
    def lab_fit(value, b, d):
        return lab_transfer(value, b, d, T_fixed)
    return lab_fit

def compute_c_and_error_2params(popt, pcov, T):
    """Compute c and error for 2-parameter fit: b, d (Lab fits)"""
    b, d = popt
    c = (b * T + d) / (T ** (1 / 3))
    partial_c_b = T / (T ** (1 / 3))
    partial_c_d = 1 / (T ** (1 / 3))
    c_error = np.sqrt(
        (partial_c_b ** 2) * pcov[0, 0] +
        (partial_c_d ** 2) * pcov[1, 1] +
        2 * partial_c_b * partial_c_d * pcov[0, 1]
    )
    return c, c_error

def fit_lab_channel(x, y, T, label=None, verbose=True):
    """Fit Lab transfer function and return b, b_err, d, d_err, c, c_err, R2, T."""
    fitLab = lab_transfer_T(T)
    popt, pcov = curve_fit(fitLab, x, y, p0=[0.01, 0.0], maxfev=10000)
    b, d = popt
    b_err, d_err = np.sqrt(np.diag(pcov))
    y_pred = fitLab(x, b, d)
    r2 = r2_score(y, y_pred)
    c, c_err = compute_c_and_error_2params(popt, pcov, T)
    return b, b_err, d, d_err, c, c_err, r2, T


    #--------Support Functions----------

# Background substraction
def subtract_background(grouped, variables, background_concentration=0):
    return {
        var: grouped[(var, 'mean')].values - grouped.loc[background_concentration, (var, 'mean')]
        for var in variables
    }

# Threshold calculations
def find_threshold_concentration(conc, values, threshold, group_first=False):

    # Convert to numpy arrays
    conc = np.asarray(conc)
    values = np.asarray(values)
    
    # Group data if requested
    if group_first:
        # Create temporary DataFrame for grouping
        df = pd.DataFrame({'conc': conc, 'values': values})
        grouped = df.groupby('conc', as_index=False).agg({'values': ['mean', 'std']})
        
        # Flatten multi-index columns
        grouped.columns = ['conc', 'values_mean', 'values_std']
        
        conc = grouped['conc'].values
        values = grouped['values_mean'].values
    
    # Sort by concentration (critical for interpolation)
    sort_idx = np.argsort(conc)
    conc = conc[sort_idx]
    values = values[sort_idx]
    
    # Find threshold crossing
    for i in range(len(values) - 1):
        if (values[i] < threshold and values[i+1] >= threshold) or \
           (values[i] > threshold and values[i+1] <= threshold):
            
            c1, c2 = conc[i], conc[i+1]
            v1, v2 = values[i], values[i+1]
            
            if v2 != v1:
                c_thresh = c1 + (threshold - v1) * (c2 - c1) / (v2 - v1)
                return c_thresh
                
    return None

# -------Color transformations: Processing Excel data containing Concentration and RGB values from selected max signal frame in videos-----

def process_rgb_excel(file_path):
    # Read Excel file
    df = pd.read_excel(file_path)
    
    #Color spaces transformations using previous definitions

    # Normalization of RGB values (0-255 to 0-1)
    df[['R_BT709','G_BT709','B_BT709']] = df[['R','G','B']] / 255.0
    RGB_BT709 = df[['R_BT709', 'G_BT709', 'B_BT709']].values
    
    # Application of BT.709 linearization
    linear_rgb = bt709_to_linear(RGB_BT709)
    df[['r', 'g', 'b']] = linear_rgb
    
    # Convertion of rgb to XYZ
    df[['X', 'Y', 'Z']] = df.apply(rgb_to_xyz, axis=1)
    
    # Convertion of XYZ to Lab
    df[['L', 'a*', 'b*', 'x_xn', 'y_yn', 'z_zn']] = df.apply(xyz_to_lab, axis=1)
    
    # Convertion of rgb to sRGB
    df[['R_sRGB', 'G_sRGB', 'B_sRGB']] = df[['r', 'g', 'b']].apply(linear_to_sRGB)
        
    return df

#------Fitting color channels against concentrations------

def fitting_plotting(df):
    
    # Group by concentration and calculate mean & std
    grouped = df.groupby('Concentration(uM)').agg(['mean', 'std'])
    concentrations = grouped.index.values
    summary_rows = []

# RGB fit with background (_b)

    # Extract mean and std values
    mean_R_BT709_b, mean_G_BT709_b, mean_B_BT709_b = grouped['R_BT709', 'mean'].values, grouped['G_BT709', 'mean'].values, grouped['B_BT709', 'mean'].values
    std_R_BT709_b, std_G_BT709_b, std_B_BT709_b = grouped['R_BT709', 'std'].values, grouped['G_BT709', 'std'].values, grouped['B_BT709', 'std'].values
    
    # Threshold calculation
    T_R_BT709_b = find_threshold_concentration(df['Concentration(uM)'].values, df['R_BT709'].values,threshold=0.081, group_first=True)
 
    # Application of fitting function
    gamma_R_BT709_b, gamma_R_BT709_b_error, b_R_BT709_b, b_R_BT709_b_error, d_R_BT709_b, d_R_BT709_b_error, c_R_BT709_b, c_R_BT709_b_error, r_sq_R_BT709_b, T_R_BT709_b = fit_rgb_channel(concentrations, mean_R_BT709_b, T_R_BT709_b, label="R_BT709_b")

    # Data summary
    summary_rows.append({
    'variable': 'R_BT709_b',
    'b': b_R_BT709_b, 'b_error': b_R_BT709_b_error,
    'c': c_R_BT709_b, 'c_error': c_R_BT709_b_error,
    'd': d_R_BT709_b, 'd_error': d_R_BT709_b_error,
    'gamma': gamma_R_BT709_b, 'gammb_error': gamma_R_BT709_b_error,
    'R2': r_sq_R_BT709_b, 'T': T_R_BT709_b
    })
    
    # Plot 
    plt.figure(figsize=(10, 6))
    plt.errorbar(concentrations, mean_R_BT709_b, yerr=std_R_BT709_b, fmt='o',markersize=15,capsize=10, capthick=3, elinewidth=3, label='Red Data', color='red')
    plt.errorbar(concentrations, mean_G_BT709_b, yerr=std_G_BT709_b, fmt='<',markersize=15,capsize=10, capthick=3, elinewidth=3, label='Green Data', color='green')
    plt.errorbar(concentrations, mean_B_BT709_b, yerr=std_B_BT709_b, fmt='>',markersize=15,capsize=10, capthick=3, elinewidth=3, label='Blue Data', color='blue')
    fitted_concentrations = np.linspace(min(concentrations), max(concentrations), 100)
    fitRGB = RGB_transfer_T(T_R_BT709_b)
    plt.plot(fitted_concentrations, fitRGB(fitted_concentrations, gamma_R_BT709_b, b_R_BT709_b, d_R_BT709_b), '--', linewidth=3, color='red', label='Red Fit')
    
    # Linear segment extension
    linear_x = concentrations[concentrations <= 100] # Extend up to concentration 100
    linear_y = b_R_BT709_b * concentrations[concentrations <= 100] # Linear segment of BT.709
    plt.plot(linear_x, linear_y, '--', color='gray', linewidth=3, label='Red Linear Extension')
   

    # Define intersection coordinates
    x_intersect_BT709 = T_R_BT709_b
    y_intersect_BT709 = 0.081  # approximately (Linear segment slope is roughly 4.5 as labeled)
    
    # Add the intersection point
    plt.scatter(x_intersect_BT709, y_intersect_BT709,  s=60,facecolors='red', edgecolors='black', linewidths=1.5, label='Intersection Point', zorder=10)
    

    # Horizontal line going slightly past the intersection
    plt.plot([-10, x_intersect_BT709 * 1.2], [y_intersect_BT709, y_intersect_BT709], 
             color='red', linestyle='--', linewidth=1)
    
    # Vertical line going up to the intersection point
    plt.plot([x_intersect_BT709, x_intersect_BT709], [-0.03, y_intersect_BT709* 1.2],
             color='red', linestyle='--', linewidth=1)
 
   
    legend_handles = [
    Line2D([0], [0], color='red', linestyle='--',linewidth=3, marker='o', markersize=15, label="R"),
    Line2D([0], [0], color='green', marker='<',linestyle='None', markersize=15, label="G"),
    Line2D([0], [0], color='blue', marker='>',linestyle='None', markersize=15, label="B")
    ]
    
    plt.xlabel('Concentration (µM)', fontsize=30)
    plt.ylabel('Normalized RGB Data', fontsize=30)
    plt.legend(handles=legend_handles, fontsize=30, frameon=False)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()
    
    
# Analysis for background substracted data
    # --- Background subtraction of all variables---
    variables = ['R_BT709', 'r','X','Y','Z','x_xn','y_yn','z_zn', 'L', 'a*', 'b*', 'R_sRGB']
    mean_nobackground = subtract_background(grouped, variables)
    df_nobg = pd.DataFrame(mean_nobackground, index=grouped.index)
    df_nobg.index.name = 'Concentration(uM)'
    
#R_BT709 fit
    # Extract mean and std values
    mean_R_BT709 = mean_nobackground['R_BT709']
    std_R_BT709 = grouped[('R_BT709', 'std')].values
    
    # Threshold calculation
    T_R_BT709 = find_threshold_concentration(concentrations, mean_R_BT709, threshold=0.081, group_first=True)
    
    # Application of fitting function
    gamma_R_BT709, gamma_R_BT709_error, b_R_BT709, b_R_BT709_error, d_R_BT709, d_R_BT709_error, c_R_BT709, c_R_BT709_error, r_sq_R_BT709, T_R_BT709 = fit_rgb_channel(concentrations, mean_R_BT709, T_R_BT709, label="R_BT709")

    # Data summary 
    summary_rows.append({
    'variable': 'R_BT709',
    'b': b_R_BT709, 'b_error': b_R_BT709_error,
    'c': c_R_BT709, 'c_error': c_R_BT709_error,
    'd': d_R_BT709, 'd_error': d_R_BT709_error,
    'gamma': gamma_R_BT709, 'gammb_error': gamma_R_BT709_error,
    'R2': r_sq_R_BT709, 'T': T_R_BT709
    })
    
#r fit
    # Extract mean and std values
    mean_r = mean_nobackground['r']
    std_r = grouped[('r', 'std')].values
    
    # Application of fitting function
    b_rBT709, b_rBT709_err, r_sq_rBT709 = fit_linear_channel(concentrations, mean_r, label="r")
    
    # Data summary
    summary_rows.append({
    'variable': 'r',
    'b': b_rBT709, 'b_error': b_rBT709_err,
    'R2': r_sq_rBT709, 
    })
    
    # r vs R_BT709 plot
    plt.figure(figsize=(7, 6))
    
    plt.errorbar(concentrations, mean_R_BT709, yerr=std_R_BT709, fmt='o', markersize=15, capsize=10,
                 capthick=3, elinewidth=3, color='red', label='BT709 Data')
    fitRGB = RGB_transfer_T(T_R_BT709)
    plt.plot(concentrations, fitRGB(concentrations, gamma_R_BT709, b_R_BT709, d_R_BT709), '--', linewidth=3, color='red', label='BT.709 Fit')
    
    plt.errorbar(concentrations, mean_r, yerr=std_r, fmt='^', markersize=15, capsize=10,
                 capthick=3, elinewidth=3, color='darkred', label='rBT709 Data')
    plt.plot(concentrations, fit_linear(concentrations, b_rBT709), '--', linewidth=3,
             color='darkred', label='rBT709 Linear Fit')
    

    # Define intersection coordinates
    x_intersect_BT709 = T_R_BT709
    y_intersect_BT709 = 0.081  # approximately (Linear segment slope is roughly 4.5 as labeled)
    
    # Add the intersection point
    plt.scatter(x_intersect_BT709, y_intersect_BT709,  s=60,facecolors='red', edgecolors='black', linewidths=1.5, label='Intersection Point', zorder=10)
    

    # Horizontal line going slightly past the intersection
    plt.plot([-10, x_intersect_BT709 * 1.2], [y_intersect_BT709, y_intersect_BT709], 
             color='red', linestyle='--', linewidth=1)
    
    # Vertical line going up to the intersection point
    plt.plot([x_intersect_BT709, x_intersect_BT709], [-0.03, y_intersect_BT709* 1.2],
             color='red', linestyle='--', linewidth=1)

    
    
    legend_handles = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=3, marker='o', markersize=15, label="R_BT709"),
        Line2D([0], [0], color='darkred', linestyle='--', linewidth=3, marker='^', markersize=15, label="r")
    ]
    
    plt.xlabel('Concentration (μM)', fontsize=30)
    plt.ylabel('R_BT709 and r', fontsize=30)
    plt.legend(handles=legend_handles, fontsize=30, frameon=False)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.show()
    
# XYZ fit
    # Extract mean and std values
    mean_X, mean_Y, mean_Z= mean_nobackground['X'], mean_nobackground['Y'],mean_nobackground['Z']
    std_X, std_Y, std_Z = grouped['X', 'std'].values, grouped['Y', 'std'].values, grouped['Z', 'std'].values
    b_X, b_X_err, r_sq_X = fit_linear_channel(concentrations, mean_X, label="X")
    b_Y, b_Y_err, r_sq_Y = fit_linear_channel(concentrations, mean_Y, label="Y")
    b_Z, b_Z_err, r_sq_Z = fit_linear_channel(concentrations, mean_Z, label="Z")

    # Data summary    
    # X
    summary_rows.append({
    'variable': 'X',
    'b': b_X, 'b_error': b_X_err,
    'R2': r_sq_X, 
    })
    
    # Y
    summary_rows.append({
    'variable': 'Y',
    'b': b_Y, 'b_error': b_Y_err,
    'R2': r_sq_Y, 
    })
    
    # Z
    summary_rows.append({
    'variable': 'Z',
    'b': b_Z, 'b_error': b_Z_err,
    'R2': r_sq_Z, 
    })
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(concentrations, mean_X, yerr=std_X, fmt='v',markersize=15,capsize=10, capthick=3, elinewidth=3, color='purple', label='_nolegend_')
    plt.errorbar(concentrations, mean_Y, yerr=std_Y, fmt='<',markersize=15,capsize=10, capthick=3, elinewidth=3, color='green', label='_nolegend_')
    plt.errorbar(concentrations, mean_Z, yerr=std_Z, fmt='>',markersize=15,capsize=10, capthick=3, elinewidth=3, color='orange', label='_nolegend_')

   
    plt.plot(concentrations, fit_linear(concentrations, b_X), '--',linewidth=3, color='purple')
    plt.plot(concentrations, fit_linear(concentrations, b_Y), '--',linewidth=3, color='green')
    plt.plot(concentrations, fit_linear(concentrations, b_Z), '--',linewidth=3, color='orange')
    
    legend_handles = [
    Line2D([0], [0], color='purple', linestyle='--',linewidth=3, marker='v', markersize=15, label="X"),
    Line2D([0], [0], color='green', linestyle='--',linewidth=3, marker='<', markersize=15, label="Y"),
    Line2D([0], [0], color='orange', linestyle='--',linewidth=3, marker='>', markersize=15, label="Z"),
    ]

    plt.xlabel('Concentration (μM)', fontsize=30)

    plt.ylabel('CIE XYZ Data', fontsize=30)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.legend(handles=legend_handles, fontsize=30, frameon=False)
    plt.show()

#sRGB fit
    # Extract mean and std values
    mean_R_sRGB = mean_nobackground['R_sRGB']
    std_sRGB = grouped['R_sRGB', 'std'].values
    
    # Threshold calculation
    T_R_sRGB = find_threshold_concentration(concentrations, mean_R_sRGB, threshold=0.04045, group_first=True)
    
    # Application of fitting function
    gamma_R_sRGB, gamma_R_sRGB_error, b_R_sRGB, b_R_sRGB_error, d_R_sRGB, d_R_sRGB_error, c_R_sRGB, c_R_sRGB_error, r_sq_R_sRGB, T_R_sRGB = fit_rgb_channel(concentrations, mean_R_sRGB, T_R_sRGB, label="R_sRGB")
   
    # Data summary
    summary_rows.append({
    'variable': 'R_sRGB',
    'b': b_R_sRGB, 'b_error': b_R_sRGB_error,
    'c': c_R_sRGB, 'c_error': c_R_sRGB_error,
    'd': d_R_sRGB, 'd_error': d_R_sRGB_error,
    'gamma': gamma_R_sRGB, 'gammb_error': gamma_R_sRGB_error,
    'R2': r_sq_R_sRGB, 'T': T_R_sRGB
    })
    
    # R_sRGB vs R_BT709 Plot 
    plt.figure(figsize=(10, 6))
    plt.errorbar(concentrations, mean_R_BT709, yerr=std_R_BT709, fmt='o', markersize=15,
                 capsize=10, capthick=3, elinewidth=3, color='red', label='BT709 Data')
    fitRGB = RGB_transfer_T(T_R_BT709)
    plt.plot(concentrations, fitRGB(concentrations, gamma_R_BT709, b_R_BT709, d_R_BT709), '--', linewidth=3, color='red', label='BT709 Fit')
    fitRGB = RGB_transfer_T(T_R_sRGB)
    fitted_concentrations = np.linspace(min(concentrations), max(concentrations), 100)
    plt.errorbar(concentrations,  mean_R_sRGB, yerr=std_sRGB, fmt='<', markersize=15,
                 capsize=10, capthick=3, elinewidth=3, color='darkred', label='sRGB Data')
    plt.plot(fitted_concentrations, fitRGB(fitted_concentrations, gamma_R_sRGB, b_R_sRGB, d_R_sRGB), 
             '--', linewidth=3, color='darkred', label='sRGB Fit')
    
    
    # Define intersection coordinates
    x_intersect_BT709 = T_R_BT709
    y_intersect_BT709 = 0.081  # approximately (Linear segment slope is roughly 4.5 as labeled)
    
    # Add the intersection point
    plt.scatter(x_intersect_BT709, y_intersect_BT709,  s=60,facecolors='red', edgecolors='black', linewidths=1.5, label='Intersection Point', zorder=10)
    

    # Horizontal line going slightly past the intersection
    plt.plot([-10, x_intersect_BT709 * 1.2], [y_intersect_BT709, y_intersect_BT709], 
             color='red', linestyle='--', linewidth=1)
    
    # Vertical line going up to the intersection point
    plt.plot([x_intersect_BT709, x_intersect_BT709], [-0.03, y_intersect_BT709* 1.2],
             color='red', linestyle='--', linewidth=1)
    
    # Define intersection coordinates
    x_intersect_sRGB = T_R_sRGB 
    y_intersect_sRGB = 0.04045
    
    
    # Add the intersection point
    plt.scatter(x_intersect_sRGB, y_intersect_sRGB, s=60, facecolors='darkred', edgecolors='black', linewidths=1.5, label='Intersection Point', zorder=10)
    
    # For sRGB
    # Horizontal line going slightly past the intersection
    plt.plot([-10, x_intersect_sRGB * 1.2], [y_intersect_sRGB, y_intersect_sRGB], 
             color='darkred', linestyle='--', linewidth=1)

    # Vertical line going up to the intersection point
    plt.plot([x_intersect_sRGB, x_intersect_sRGB], [-0.03, y_intersect_sRGB* 1.2],
             color='darkred', linestyle='--', linewidth=1)
    


    legend_handles = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=3, marker='o', markersize=15, label="R_BT709 (γ=2.35)"),
        Line2D([0], [0], color='darkred', linestyle='--', linewidth=3, marker='<', markersize=15, label="R_sRGB (γ=2.4)")
    ]
    
    plt.xlabel('Concentration (μM)', fontsize=30)
    plt.ylabel('Normalized R Data', fontsize=30)
    plt.legend(handles=legend_handles, fontsize=30, frameon=False)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show() 
    
#CIELAB 
    # Extract mean and std values
    mean_L, mean_a, mean_b  = mean_nobackground['L'], mean_nobackground['a*'], mean_nobackground['b*']
    std_L, std_a, std_b = grouped[('L', 'std')].values, grouped[('a*', 'std')].values, grouped[('b*', 'std')].values
    
    # Threshold calculation
    x_xn, y_yn, z_zn = mean_nobackground['x_xn'], mean_nobackground['y_yn'], mean_nobackground['z_zn']
    
    thresh_y_yn = find_threshold_concentration(concentrations, y_yn, threshold=(6/29)**3, group_first=True)
    thresh_x_xn = find_threshold_concentration(concentrations, x_xn, threshold=(6/29)**3, group_first=True)
    thresh_z_zn = find_threshold_concentration(concentrations, z_zn, threshold=(6/29)**3, group_first=True)
    
    T_L = thresh_y_yn 
    T_a = np.minimum(thresh_x_xn, thresh_y_yn)
    T_b = np.minimum(thresh_y_yn, thresh_z_zn)

    # Application of fitting function
    fit_L = lab_transfer_T(T_L)
    fit_a = lab_transfer_T(T_a)
    fit_b = lab_transfer_T(T_b)
    
    b_L, b_L_error, d_L, d_L_error, c_L, c_L_error, r_sq_L, T_L = fit_lab_channel(concentrations, mean_L, T_L, label="L")
    b_a, b_a_error, d_a, d_a_error, c_a, c_a_error, r_sq_a, T_a = fit_lab_channel(concentrations, mean_a, T_a, label="a*")
    b_b, b_b_error, d_b, d_b_error, c_b, c_b_error, r_sq_b, T_b = fit_lab_channel(concentrations, mean_b, T_b, label="b*")
    
    # Data summary
    summary_rows.append({
    'variable': 'L',
    'b': b_L, 'b_error': b_L_error,
    'c': c_L, 'c_error': c_L_error,
    'd': d_L, 'd_error': d_L_error,
    'R2': r_sq_L, 'T': T_L
    })
    summary_rows.append({
    'variable': 'a',
    'b': b_a, 'b_error': b_a_error,
    'c': c_a, 'c_error': c_a_error,
    'd': d_a, 'd_error': d_a_error,
    'R2': r_sq_a, 'T': T_a
    })
    summary_rows.append({
    'variable': 'b',
    'b': b_b, 'b_error': b_b_error,
    'c': c_b, 'c_error': c_b_error,
    'd': d_b, 'd_error': d_b_error,
    'R2': r_sq_b, 'T': T_b
    })
    
    # Plot
    plt.figure(figsize=(10, 6))
    fitted_concentrations = np.linspace(min(concentrations), max(concentrations), 100)
    
    # L*
    plt.errorbar(concentrations, mean_L, std_L, fmt='o', markersize=15, capsize=10, capthick=3, elinewidth=3, label='L', color='green')
    plt.plot(fitted_concentrations, fit_L(fitted_concentrations, b_L, d_L), '--', linewidth=3, label='L* fit', color='green')
    
    # a*
    plt.errorbar(concentrations, mean_a, std_a, fmt='o', markersize=15, capsize=10, capthick=3, elinewidth=3, label='a*', color='purple')
    plt.plot(fitted_concentrations, fit_a(fitted_concentrations, b_a, d_a), '--', linewidth=3, label='a* fit', color='purple')
    
    # b*
    plt.errorbar(concentrations, mean_b, std_b, fmt='o', markersize=15, capsize=10, capthick=3, elinewidth=3, label='b*', color='orange')
    plt.plot(fitted_concentrations, fit_b(fitted_concentrations, b_b, d_b), '--', linewidth=3, label='b* fit', color='orange')
    plt.xlabel('Concentration (μM)', fontsize=30)
    plt.ylabel('Lab* Data', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    legend_handles = [
        Line2D([0], [0], color='green', linestyle='--', linewidth=3, marker='o', markersize=15, label="L"),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=3, marker='o', markersize=15, label="a*"),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=3, marker='o', markersize=15, label="b*")
    ]
    plt.legend(handles=legend_handles, fontsize=30, frameon=False)
    plt.show()
    
#------Data summary to Excel file-------

    summary_table = pd.DataFrame(summary_rows, columns=['variable', 'b', 'b_error', 'c', 'c_error', 'd', 'd_error', 'gamma', 'gammb_error', 'R2', 'T'])
    with pd.ExcelWriter("Output path of your colorprocesseddata.xlsx") as writer:  # Replace with your output path
        df_processed.to_excel(writer, sheet_name="Processed Data", index=False)
        grouped.to_excel(writer, sheet_name="Grouped Stats With BG")
        df_nobg.to_excel(writer, sheet_name="Grouped Stats No BG")
        summary_table.to_excel(writer, sheet_name="Summary Table", index=False)
        
# Usage:
if __name__ == "__main__":
    input_file = "Path to your RGBresults.xlsx.xlsx"  # Replace with your actual path
    df_processed = process_rgb_excel(input_file)
    fitting= fitting_plotting(df_processed)
