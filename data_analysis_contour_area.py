import os
import pandas as pd
import matplotlib.pyplot as plt

# Search for Excel files in a directory
directory = "C:/Users/garci/Desktop/Test1/output_images/binarized and contour"
files = [f for f in os.listdir(directory) if f.endswith(".xlsx")]
print("Found Excel files:", files)

# Load the first Excel file (update the selection as needed)
if files:
    file_path = os.path.join(directory, files[0])
    df = pd.read_excel(file_path)
    
    # Display the first few rows to understand the structure
    print(df.head())
    
    # Manual filtering options
    fuel_filter = input("Enter Fuel type to filter (or press Enter to skip): ")
    nozzle_filter = input("Enter Nozzle type to filter (or press Enter to skip): ")
    temp_filter = input("Enter Gas Phase Temperature [C] to filter (or press Enter to skip): ")
    
    if fuel_filter:
        df = df[df["Fuel"] == fuel_filter]
    if nozzle_filter:
        df = df[df["Nozzle"] == nozzle_filter]
    if temp_filter:
        try:
            temp_filter = float(temp_filter)
            df = df[df["Gas Phase Temperature [C]"] == temp_filter]
        except ValueError:
            print("Invalid temperature value, skipping filter.")
    
    # Plot Relative Growth [%]
    plt.figure(figsize=(8, 6))
    for name, group in df.groupby("Concatenate name"):
        plt.plot(group["Pressure [bar]"], group["Relative Growth [%]"], marker='o', linestyle='--', label=name)
        for x, y in zip(group["Pressure [bar]"], group["Relative Growth [%]"]):
            plt.text(x, y, f'{y:.3f}', fontsize=9, ha='right')
    plt.xlabel("Pressure [bar]")
    plt.ylabel("Relative Growth [%]")
    plt.title("Relative Growth vs. Pressure")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot Absolute Growth [%]
    plt.figure(figsize=(8, 6))
    for name, group in df.groupby("Concatenate name"):
        plt.plot(group["Pressure [bar]"], group["Absolute Growth [%]"], marker='s', linestyle='--', label=name)
        for x, y in zip(group["Pressure [bar]"], group["Absolute Growth [%]"]):
            plt.text(x, y, f'{y:.3f}', fontsize=9, ha='right')
    plt.xlabel("Pressure [bar]")
    plt.ylabel("Absolute Growth [%]")
    plt.title("Absolute Growth vs. Pressure")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("No Excel files found in the specified directory.")
