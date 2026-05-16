import pandas as pd
import numpy as np
import eurostat
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

print("Fetching deep historical data (1998-2026) from Eurostat...")
df_list = []

# =====================================================================
# 1. Fetch Supply & Demand Data
# =====================================================================
# Supply Side: Chemicals (C20)
pars_base = {'geo': 'EU27_2020', 'nace_r2': 'C20', 'unit': 'I21', 's_adj': 'SCA'}
try:
    df_raw_base = eurostat.get_data_df('sts_inpr_m', filter_pars=pars_base)
    date_cols = [col for col in df_raw_base.columns if col[0].isdigit()]
    df_list.append(df_raw_base.melt(id_vars=['nace_r2'], value_vars=date_cols, var_name='Month', value_name='Index'))
except Exception as e: 
    print(f"❌ Failed to fetch C20 (Chemicals): {e}")

# Demand Side: Plastics/Rubber (C22) and Automotive (C29)
for nace in ['C22', 'C29']:
    pars_d = {'geo': 'EU27_2020', 'nace_r2': nace, 'unit': 'I21', 's_adj': 'SCA'}
    try:
        df_raw_d = eurostat.get_data_df('sts_inpr_m', filter_pars=pars_d)
        date_cols = [col for col in df_raw_d.columns if col[0].isdigit()]
        df_list.append(df_raw_d.melt(id_vars=['nace_r2'], value_vars=date_cols, var_name='Month', value_name='Index'))
    except Exception as e: 
        print(f"❌ Failed to fetch {nace}: {e}")

# Demand Side: Construction (F)
for nace in ['F']:
    pars_d = {'geo': 'EU27_2020', 'nace_r2': nace, 'unit': 'I21', 's_adj': 'SCA'}
    try:
        df_raw_d = eurostat.get_data_df('sts_copr_m', filter_pars=pars_d)
        date_cols = [col for col in df_raw_d.columns if col[0].isdigit()]
        df_list.append(df_raw_d.melt(id_vars=['nace_r2'], value_vars=date_cols, var_name='Month', value_name='Index'))
    except Exception as e: 
        print(f"❌ Failed to fetch {nace}: {e}")

# Combine, pivot, and clean the dataset
df_all = pd.concat(df_list)
df_all['Month'] = pd.to_datetime(df_all['Month'])
df_pivot = df_all.pivot(index='Month', columns='nace_r2', values='Index').sort_index()

# Keep data from 1998 onwards, forward fill maximum 2 months for quarterly reporting gaps, then drop remaining NaNs
df_pivot = df_pivot[df_pivot.index >= '1998-01-01'].apply(pd.to_numeric, errors='coerce').ffill(limit=2).dropna()

# =====================================================================
# 2. ACADEMIC PRE-FLIGHT: ADF Stationarity Test I(1)
# =====================================================================
print("\n" + "="*50)
print("🔬 ACADEMIC PRE-FLIGHT: STATIONARITY CHECK")
print("="*50)
print("Cointegration requires all series to be non-stationary I(1).")

all_i1 = True
for col in df_pivot.columns:
    # Test levels
    adf_level = adfuller(df_pivot[col].dropna(), autolag='AIC')
    p_level = adf_level[1]
    
    # Test first differences
    adf_diff = adfuller(df_pivot[col].diff().dropna(), autolag='AIC')
    p_diff = adf_diff[1]
    
    status = "Pass" if (p_level > 0.05 and p_diff < 0.05) else "Fail"
    if status == "Fail": 
        all_i1 = False
    print(f"[{status}] {col}: Level p-value = {p_level:.3f} | First Diff p-value = {p_diff:.3f}")

if all_i1:
    print("✅ All series are I(1). Cointegration mathematics are valid.\n")
else:
    print("⚠️ WARNING: Not all series are perfectly I(1). Results should be interpreted with caution.\n")

# =====================================================================
# 3. ROBUST Rolling Cointegration Tests
# =====================================================================
window = 84        # 7-year rolling window
min_periods = 60   # Minimum 5 years of overlap required
target_sectors = ['C22', 'C29', 'F']

df_eg_pvalues = pd.DataFrame(index=df_pivot.index)
df_johansen_system = pd.DataFrame(index=df_pivot.index, columns=['System_Cointegrated'])

print("Calculating dynamic pairwise Engle-Granger (AIC Lags) and systemic Johansen Cointegration...")

for i in range(len(df_pivot)):
    if i < min_periods:
        for target in target_sectors:
            df_eg_pvalues.loc[df_pivot.index[i], target] = np.nan
        df_johansen_system.loc[df_pivot.index[i], 'System_Cointegrated'] = np.nan
    else:
        # Start index and data slicing (properly indented)
        start_idx = max(0, i - window)
        slice_data = df_pivot.iloc[start_idx:i]
        
        # A. Engle-Granger (Pairwise with Dynamic AIC lags)
        for target in target_sectors:
            try:
                _, pval, _ = coint(slice_data['C20'], slice_data[target], trend='c', autolag='AIC')
                df_eg_pvalues.loc[df_pivot.index[i], target] = pval
            except Exception:
                df_eg_pvalues.loc[df_pivot.index[i], target] = np.nan
                
        # B. Johansen Test (Multivariate System Check)
        try:
            j_res = coint_johansen(slice_data[['C20', 'C22', 'C29', 'F']], det_order=0, k_ar_diff=1)
            
            trace_stat = j_res.lr1[0]
            trace_crit_95 = j_res.cvt[0, 1] 
            
            is_coint = 1 if trace_stat > trace_crit_95 else 0
            df_johansen_system.loc[df_pivot.index[i], 'System_Cointegrated'] = is_coint
        except Exception:
            df_johansen_system.loc[df_pivot.index[i], 'System_Cointegrated'] = np.nan

# Filter DataFrames to only plot the period with valid test outputs
df_plot_eg = df_eg_pvalues[df_eg_pvalues.index >= '2005-01-01']
df_plot_johansen = df_johansen_system[df_johansen_system.index >= '2005-01-01']

# =====================================================================
# 4. Multi-Panel Visualisation
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})

# Panel 1: The Engle-Granger Heatmap (Pairwise)
cmap_colors = ['green', 'yellow', 'red']
custom_cmap = LinearSegmentedColormap.from_list("CustomMap", list(zip([0.0, 0.05, 1.0], cmap_colors)))

heatmap_data = df_plot_eg[['F', 'C29', 'C22']].T.values.astype(float)
heatmap_data_masked = np.ma.masked_invalid(heatmap_data)
x_edges = list(df_plot_eg.index)
x_edges.append(x_edges[-1] + pd.DateOffset(months=1)) 

heatmap = ax1.pcolormesh(x_edges, [-0.5, 0.5, 1.5, 2.5], heatmap_data_masked, cmap=custom_cmap, vmin=0, vmax=1.0)
ax1.set_ylim(-0.5, 2.5)
ax1.set_title('Part 1: Pairwise Decoupling (Engle-Granger, Dynamic AIC Lags)', fontsize=14, fontweight='bold')
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['F: Construction', 'C29: Auto', 'C22: Plastics'], fontsize=11, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.axvline(pd.to_datetime('2008-09-01'), color='black', linestyle=':', linewidth=2)
ax1.axvline(pd.to_datetime('2022-02-01'), color='black', linestyle='--', linewidth=2)

cbar = fig.colorbar(heatmap, ax=ax1, location='right', pad=0.02, aspect=10)
cbar.set_ticks([0.0, 0.05, 0.3, 1.0])
cbar.set_ticklabels(['Secure', 'Caution', 'Fraying', 'Snapped'])

# Panel 2: The Johansen Systemic Health Indicator (with step='post' fix)
johansen_vals = df_plot_johansen['System_Cointegrated'].values.astype(float)

ax2.fill_between(df_plot_johansen.index, 0, johansen_vals, color='green', alpha=0.5, label="System Intact", step='post')
ax2.fill_between(df_plot_johansen.index, johansen_vals, 1, color='red', alpha=0.5, label="System Decoupled", step='post')

ax2.set_title('Part 2: Multivariate Systemic Health (Johansen Trace Statistic)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.set_yticks([0.1, 0.9])
ax2.set_yticklabels(['DECOUPLED\n(No Cointegrating Vector)', 'INTACT\n(Multivariate Equilibrium)'], fontsize=10, fontweight='bold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.axvline(pd.to_datetime('2008-09-01'), color='black', linestyle=':', linewidth=2)
ax2.axvline(pd.to_datetime('2022-02-01'), color='black', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('EU_Chemicals_Academic_Audit.png', dpi=300, bbox_inches='tight')
print("✅ Visualisation successfully saved as 'EU_Chemicals_Academic_Audit.png'.")

# =====================================================================
# 5. EXPORT DATA FOR ACADEMIC REVIEW
# =====================================================================
print("\n" + "="*50)
print("💾 EXPORTING DATA TO CSV")
print("="*50)

# Create copies with clearer column headers for the export
df_export = df_pivot.copy()
df_export.columns = [f"Raw_{col}" for col in df_export.columns]

df_eg_export = df_eg_pvalues.copy()
df_eg_export.columns = [f"EngleGranger_P_{col}" for col in df_eg_export.columns]

# Concatenate all data into a master DataFrame
df_final = pd.concat([df_export, df_eg_export, df_johansen_system], axis=1)

# Save the final file
csv_filename = "EU_Chemicals_Academic_Data.csv"
df_final.to_csv(csv_filename)
print(f"✅ Detailed econometric data successfully exported to '{csv_filename}'.")
print(f"Total evaluated months exported: {len(df_final)}")
print("\nScript Complete.")