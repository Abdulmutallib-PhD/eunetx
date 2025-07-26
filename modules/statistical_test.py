import pandas as pd
from scipy.stats import wilcoxon

# Load UNetX detailed evaluation data
unetx_path = 'results/unetx_final_performance.csv'
df_unetx = pd.read_csv(unetx_path)

# Validate the presence of required columns
if 'DSC' not in df_unetx.columns or 'HD95' not in df_unetx.columns:
    raise ValueError("The CSV file must contain 'DSC' and 'HD95' columns.")

# Summary statistics
mean_dsc = df_unetx['DSC'].mean()
mean_hd95 = df_unetx['HD95'].mean()
median_dsc = df_unetx['DSC'].median()
median_hd95 = df_unetx['HD95'].median()

summary = {
    'Mean DSC': round(mean_dsc, 4),
    'Median DSC': round(median_dsc, 4),
    'Mean HD95 (mm)': round(mean_hd95, 4),
    'Median HD95 (mm)': round(median_hd95, 4)
}

summary_df = pd.DataFrame([summary])
output_path = 'results/unetx_statistical_summary.csv'
summary_df.to_csv(output_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="UNetX Statistical Summary", dataframe=summary_df)

print(f"UNetX statistical summary saved to: {output_path}")
