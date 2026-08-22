import csv
import re
import math

# Load basin names from task-214 log
basin_names = {}
with open(r'C:\Users\harsh\.gemini\antigravity-ide\brain\39b0e73a-0bba-429c-97e6-5c92de07b520\.system_generated\tasks\task-214.log', 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        match = re.match(r'^(\d+):\s*(.*)$', line)
        if match:
            b_id = int(match.group(1))
            name = match.group(2).strip()
            basin_names[b_id] = name

# We will also check if the manuscript's mentioned basins match.
# Manuscript: 51=Indus, 17=Tigris-Euphrates, 40,45=Western NA
# task-214: 51=Ganges-Brahmaputra, 17=Don, 42=Indus, 39=Tigris-Euphrates
# Let's just create the table using the basin_names dictionary we extracted.

summary_file = r'c:\SILIKA\Thesis\data2\outputs\tables\basin_summary_table.csv'
bootstrap_file = r'c:\SILIKA\Thesis\data2\outputs\tables\bootstrap_attribution_uncertainty.csv'

# Read basin summary
summary_data = {}
with open(summary_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        b_id = int(row['Basin_ID'])
        summary_data[b_id] = row

# Read bootstrap (which might have NaNs for now, but we will structure the code so it works when updated)
boot_data = {}
with open(bootstrap_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        b_id = int(row['Basin_ID'])
        boot_data[b_id] = row

# Output Table S1
output_csv = r'c:\SILIKA\Thesis\data2\outputs\tables\Table_S1_Supplementary.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [
        'Basin_ID', 'Basin_Name', 'Trend_cm_yr', 'p_value', 'Significant',
        'NSE_Natural', 'NSE_Anthro', 'Delta_NSE', 'Delta_R2_Mean', 'Delta_R2_95_CI',
        'Dominant_Driver'
    ]
    writer.writerow(header)
    
    # We will determine dominant driver based on max of Top_Driver_Prob from boot_data,
    # or if NaN, from Imp_X_Mean. Or if not available, we leave it blank for now.
    
    for b_id in range(1, 104):
        name = basin_names.get(b_id, 'Unknown')
        s_row = summary_data.get(b_id, {})
        b_row = boot_data.get(b_id, {})
        
        trend = s_row.get('Trend_cm_yr', 'NaN')
        pval = s_row.get('MK_p_value', 'NaN')
        sig = s_row.get('Is_Significant', '0')
        
        nse_nat = s_row.get('NSE_Natural', 'NaN')
        nse_ant = s_row.get('NSE_Anthro', 'NaN')
        
        try:
            d_nse = f"{(float(nse_ant) - float(nse_nat)):.4f}"
        except:
            d_nse = 'NaN'
            
        dr2_mean = b_row.get('Delta_R2_Mean', 'NaN')
        dr2_low = b_row.get('Delta_R2_CI_2_5', 'NaN')
        dr2_upp = b_row.get('Delta_R2_CI_97_5', 'NaN')
        
        if dr2_mean == 'NaN' or dr2_mean == '':
            dr2_ci = 'NaN'
        else:
            try:
                dr2_ci = f"[{float(dr2_low):.4f}, {float(dr2_upp):.4f}]"
                dr2_mean = f"{float(dr2_mean):.4f}"
            except:
                dr2_ci = 'NaN'
                
        # Find dominant driver from bootstrap means if available
        dom_driver = 'Unknown'
        if b_row and b_row.get('Imp_P_Mean') not in ['NaN', '']:
            feats = ['P', 'ET', 'Q', 'T', 'ONI', 'GW_abs', 'SW_abs']
            means = []
            for ft in ['Imp_P_Mean', 'Imp_ET_Mean', 'Imp_Q_Mean', 'Imp_T_Mean', 'Imp_ONI_Mean', 'Imp_GW_Mean', 'Imp_SW_Mean']:
                try:
                    means.append(float(b_row.get(ft, -999)))
                except:
                    means.append(-999)
            max_idx = means.index(max(means))
            dom_driver = feats[max_idx]
            
        # Format for nice reading
        if trend != 'NaN': trend = f"{float(trend):.4f}"
        if pval != 'NaN': pval = f"{float(pval):.4e}"
        if nse_nat != 'NaN': nse_nat = f"{float(nse_nat):.4f}"
        if nse_ant != 'NaN': nse_ant = f"{float(nse_ant):.4f}"
            
        writer.writerow([
            b_id, name, trend, pval, sig, nse_nat, nse_ant, d_nse, dr2_mean, dr2_ci, dom_driver
        ])

print(f"Successfully generated {output_csv}")
