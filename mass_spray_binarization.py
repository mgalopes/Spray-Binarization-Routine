# -*- coding: utf-8 -*-
"""
Scientific Visualization Pipeline with Matplotlib
- Publication-quality formatting with Times New Roman
- Enhanced axis labeling and grid
- Professional legend placement
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ================= Configuration =================
FOLDER_PATH = 'C:/Users/garci/Desktop/Test1/cropped_img/etanol_conv_25C'
OUTPUT_FOLDER = 'C:/Users/garci/Desktop/Test1/output_images/binarized and contour'
LEGEND_WIDTH_RATIO = 0.2  # 20% of figure width for legend
DPI = 600  # Publication quality resolution
FONT_SIZE = 10
GRID_SPACING = 50  # Pixels between grid lines
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
            img = cv2.imread(os.path.join(FOLDER_PATH, filename))
            if img is not None:
                # Process image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Store data
                contour_data.append({
                    'filename': filename,
                    'contours': contours,
                    'shape': img.shape
                })

                # Save individual outputs
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"binarized_{filename}"), thresh)

    # Create combined plot
    if contour_data:
        # Create figure with proper dimensions
        fig_width = contour_data[0]['shape'][1] * (1 + LEGEND_WIDTH_RATIO) / 100
        fig_height = contour_data[0]['shape'][0] / 100
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        
        # Create base image
        base_img = np.ones(contour_data[0]['shape'], dtype=np.uint8) * 255
        
        # Plot contours with unique colors
        legend_elements = []
        cmap = plt.get_cmap('tab20')
        for idx, data in enumerate(contour_data):
            color = cmap(idx/len(contour_data))
            for contour in data['contours']:
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], 
                        color=color, linewidth=0.8)
            
            # Add legend entry
            legend_elements.append(Patch(facecolor=color,
                                        label=os.path.splitext(data['filename'])[0]))

        # Configure axis
        ax.imshow(base_img)
        ax.set_xlim(0, base_img.shape[1])
        ax.set_ylim(base_img.shape[0], 0)  # Matplotlib origin is top-left
        ax.set_xlabel("X Position (pixels)", fontsize=FONT_SIZE+2, fontweight='bold')
        ax.set_ylabel("Y Position (pixels)", fontsize=FONT_SIZE+2, fontweight='bold')
        
        # Configure grid
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xticks(np.arange(0, base_img.shape[1], GRID_SPACING))
        ax.set_yticks(np.arange(0, base_img.shape[0], GRID_SPACING))
        
        # Add legend
        legend = ax.legend(handles=legend_elements, 
                          bbox_to_anchor=(1.05, 1), 
                          loc='upper left',
                          title='Experimental Conditions:',
                          title_fontsize=FONT_SIZE+1,
                          frameon=True,
                          framealpha=1,
                          edgecolor='black')
        legend.get_frame().set_linewidth(1.2)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "contour_combined.png"), 
                   dpi=DPI, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    process_images()