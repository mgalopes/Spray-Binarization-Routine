# -*- coding: utf-8 -*-
"""
Enhanced Visualization Pipeline with Centered Title
- Centered plot title
- Automatic legend names from filenames
- Individual contour images
- Combined plot with formatted legend
"""

import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ================= Configuration =================
FOLDER_PATH = 'C:/Users/garci/Desktop/Test1/cropped_img/etanol_conv_25C'
OUTPUT_FOLDER = 'C:/Users/garci/Desktop/Test1/output_images/binarized and contour'
DPI = 600
FONT_SIZE = 10
GRID_SPACING = 50
CROP_Y = 650
CUSTOM_COLORS = ['red', 'blue', 'green']
PLOT_TITLE = "Spray Pattern Analysis - Ethanol at 25°C"  # Add your title here
# ==================================================

# Configure Matplotlib
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.linewidth'] = 1.2

def process_images():
    """Main processing pipeline"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    contour_data = []

    # Process images
    for filename in sorted(os.listdir(FOLDER_PATH)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(FOLDER_PATH, filename)
            img = cv2.imread(file_path)
            if img is not None:
                # Process image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Calculate area
                total_area = sum(cv2.contourArea(cnt) for cnt in contours)
                print(f"{filename} - Total contour area: {total_area} pixels")

                # Store data
                contour_data.append({
                    'filename': filename,
                    'contours': contours,
                    'shape': img.shape,
                    'area': total_area
                })

                # Save individual outputs
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"binarized_{filename}"), thresh)
                
                # Save individual contour plot
                contour_img = np.ones(img.shape, dtype=np.uint8) * 255
                cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 1)
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"contour_{filename}"), contour_img)

    # Create combined plot
    if contour_data:
        fig, ax = plt.subplots(figsize=(10, CROP_Y/100), dpi=100)
        base_img = np.ones((CROP_Y, contour_data[0]['shape'][1], 3), dtype=np.uint8) * 255
        
        legend_elements = []
        
        for idx, data in enumerate(contour_data):
            color = CUSTOM_COLORS[idx % len(CUSTOM_COLORS)]
            
            # Plot contours
            for contour in data['contours']:
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], 
                        color=color, linewidth=0.8)
            
            # Generate legend name from filename
            match = re.search(r'(\d+)bar', data['filename'])
            if match:
                legend_name = f"{match.group(1)} bar"
            else:
                legend_name = os.path.splitext(data['filename'])[0]
            
            legend_text = f"{legend_name}\n({data['area']:.0f} px)"
            legend_elements.append(Patch(facecolor=color, label=legend_text))

        # Configure axis and grid
        ax.imshow(base_img)
        ax.set_xlim(0, base_img.shape[1])
        ax.set_ylim(CROP_Y, 0)
        
        # Add centered title
        ax.set_title(PLOT_TITLE, fontsize=FONT_SIZE+4, pad=20, loc='center')
        
        ax.set_xlabel("X Position (pixels)", fontsize=FONT_SIZE+2, fontweight='bold')
        ax.set_ylabel("Y Position (pixels)", fontsize=FONT_SIZE+2, fontweight='bold')
        ax.set_xticks(np.arange(0, base_img.shape[1]+1, GRID_SPACING))
        ax.set_yticks(np.arange(0, CROP_Y+1, 100))
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Add legend
        legend = ax.legend(handles=legend_elements, 
                          bbox_to_anchor=(1.05, 1), 
                          loc='upper left',
                          title='Pressure Conditions:',
                          title_fontsize=FONT_SIZE+1,
                          frameon=True,
                          framealpha=1,
                          edgecolor='black')
        legend.get_frame().set_linewidth(1.2)

        # Save output
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "contour_combined.png"), 
                   dpi=DPI, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    process_images()