import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    'Model': ['UNet++', 'Swin-PANet', 'MedT', 'TransUNet', 'UNetX'],
    'Total Images': [9392, 1488, 1838, 4654, 16000]
}

df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(8, 6))
bars = plt.barh(df['Model'], df['Total Images'], color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'violet'])

plt.title('Total Number of Images/Scans per Model')
plt.xlabel('Number of Images/Scans')
plt.ylabel('Model')

# Add numbers on bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 200, bar.get_y() + bar.get_height()/2,
             f'{int(width):,}', va='center', fontsize=9)

plt.tight_layout()
plt.show()
