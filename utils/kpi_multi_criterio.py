import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

with open('./configs/config.yml', 'r') as file:
    config = yaml.safe_load(file)


def normalize_dataset(df):
    df_normalized = df.copy()

    for col in df_normalized.columns:
        # Step 1: Override negativi a 0
        df_normalized[col] = df_normalized[col].clip(lower=0)

        # Step 2: Normalizzazione standard tra 0 e 1
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()

        if col_max != col_min:
            df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        else:
            # Evita divisioni per zero se tutti i valori sono uguali
            df_normalized[col] = 0.0

    return df_normalized * 100  # Moltiplica per 100 per avere valori tra 0 e 100


def calculate_kpis(folder_path, month):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    results = []
    for idx, file_csv in enumerate(csv_files):
        file_path = os.path.join(folder_path, file_csv)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        df = pd.read_csv(file_path)
        df = df[df['Month'] == month]  # Gennaio

        # Raggruppamento per giorno
        grouped = df.groupby('Day')

        # Liste per salvare i KPI giornalieri
        daily_kpi_heating = []
        daily_kpi_cooling = []
        daily_kpi_fan = []
        daily_kpi_comfort = []
        daily_kpi_iaq = []

        for day, day_df in grouped:
            # Evitiamo divisioni per zero
            day_df = day_df.copy()
            day_df[config['discharge_flow']] = day_df[config['discharge_flow']].replace(0, np.nan)

            # Cooling Power (kWh)
            p_cooling = (day_df[config['p_cooling']] / 4).sum() / 1000
            daily_kpi_cooling.append(p_cooling)

            # Heating Power (kWh)
            p_heating = (day_df[config['p_heating']] / 4).sum() / 1000
            daily_kpi_heating.append(p_heating)

            # Fan Power (Wh)
            p_fan = (day_df[config['fan_power']] / 4).sum()  # Wh
            daily_kpi_fan.append(p_fan)

            # Comfort Deviation
            day_df_comfort = day_df[(day_df['Hour'] >= 6) & (day_df['Hour'] <= 18)].copy()  # Filtriamo i valori
            dev_heating = np.where(
                day_df_comfort[config['t_heating_sp']] - 5 / 9 < day_df_comfort[config['t_room']],
                0,
                (day_df_comfort[config['t_heating_sp']] - 5 / 9) - day_df_comfort[config['t_room']])
            dev_heating = dev_heating.sum()
            # dev_heating = dev_heating.max()

            dev_cooling = np.where(
                day_df_comfort[config['t_cooling_sp']] + 5 / 9 > day_df_comfort[config['t_room']],
                0,
                day_df_comfort[config['t_room']] - (day_df_comfort[config['t_cooling_sp']] + 5 / 9))
            dev_cooling = dev_cooling.sum()
            # dev_cooling = dev_cooling.max()

            dev_tot = (dev_heating + dev_cooling) / 4
            # dev_tot = max(dev_heating, dev_cooling)
            daily_kpi_comfort.append(dev_tot)

            # IAQ
            day_df_iaq = day_df[day_df[config['discharge_flow']] > 10].copy()  # Filtriamo i valori
            oa_ratio = (day_df_iaq[config['outdoor_flow']] / day_df_iaq[config['discharge_flow']]).mean()
            daily_kpi_iaq.append(oa_ratio)

        # Media mensile dei KPI
        avg_kpi = {'Heating_kWh': np.nanmean(daily_kpi_heating), 'Cooling_kWh': np.nanmean(daily_kpi_cooling),
                   'Fan_Wh': np.nanmean(daily_kpi_fan), 'Comfort_dT': np.nanmean(daily_kpi_comfort),
                   'IAQ_Ratio': np.nanmean(daily_kpi_iaq), 'File': file_name}

        results.append(avg_kpi)

    # Costruzione DataFrame finale
    df_kpis = pd.DataFrame(results)
    df_kpis.set_index('File', inplace=True)

    ## Calcolo del FIR
    # 1. Estrai i valori normali (FaultFree)
    df_normal = df_kpis.loc['FCU_FaultFree']

    # 2. Rimuovi la riga "FaultFree"
    df_faults = df_kpis.drop('FCU_FaultFree')

    # 3. Calcolo del FIR: (KPI_fault - KPI_normal) / KPI_normal
    df_fir = pd.DataFrame(index=df_faults.index, columns=df_faults.columns)

    for col in df_faults.columns:
        fault_vals = df_faults[col]
        normal_val = df_normal[col]

        fir = np.where(
            (normal_val == 0) & (fault_vals == 0),
            0,  # entrambi zero → FIR = 0
            np.where(
                (normal_val == 0) & (fault_vals != 0),
                fault_vals,  # solo normal == 0 → FIR = fault
                (fault_vals - normal_val) / normal_val  # normal ≠ 0 → formula classica
            )
        )

        df_fir[col] = fir

    # 4. Inverti il segno solo per IAQ_Ratio
    df_fir['IAQ_Ratio'] = -df_fir['IAQ_Ratio']

    df_fir = df_fir.rename(columns={
        'Heating_kWh': 'Heating energy',
        'Cooling_kWh': 'Cooling energy',
        'Fan_Wh': 'Fan energy',
        'Comfort_dT': 'Comfort',
        'IAQ_Ratio': 'IAQ ratio'
    })
    df_fir = df_fir[['Heating energy', 'Cooling energy', 'Fan energy', 'Comfort', 'IAQ ratio']]

    df_fir_norm = normalize_dataset(df_fir)

    return df_kpis, df_fir, df_fir_norm


def kpi_carpet_plot(df, figsize, season):
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap='coolwarm', linewidths=1, linecolor='white', annot=True, fmt=".1f", annot_kws={"color": "black"})
    plt.ylabel("Fault scenario", fontsize=13)
    plt.xticks(rotation=15)
    plt.xlabel(f"KPI values during {season}", fontsize=13)
    plt.tight_layout()
    # plt.savefig(f'./figs/kpi_carpet_plot_{season}.png', dpi=300)
    plt.show()


def fir_carpet_plot(df, figsize, season):
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap='coolwarm', linewidths=1, linecolor='white', annot=True, fmt=".1f", annot_kws={"color": "black"})
    plt.ylabel("Fault scenario", fontsize=13)
    plt.xticks(rotation=15)
    plt.xlabel(f"FIR values during {season}", fontsize=13)
    plt.tight_layout()
    # plt.savefig(f'./figs/fir_carpet_plot_{season}.png', dpi=300)
    plt.show()


def fir_norm_carpet_plot(df, figsize, season):
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap='coolwarm', linewidths=1, linecolor='white', annot=True, fmt=".1f", annot_kws={"color": "black"})
    plt.ylabel("Fault scenario", fontsize=13)
    plt.xticks(rotation=15)
    plt.xlabel(f"Normalized FIR values during {season}", fontsize=13)
    plt.tight_layout()
    # plt.savefig(f'./figs/fir_norm_carpet_plot_{season}.png', dpi=300)
    plt.show()


def multi_criteria_fir_norm(df_fir, n, alpha, season):
    # Genera n campioni di pesi da una distribuzione di Dirichlet
    weight_samples = np.random.dirichlet(alpha, n)

    # Calcola i punteggi pesati
    kpi_matrix = df_fir.values
    weighted_scores = np.dot(weight_samples, kpi_matrix.T)  # shape: (n_samples, n_files)
    avg_weighted_scores = np.mean(weighted_scores, axis=0)

    # Prepara il dataframe con i risultati
    results_df = pd.DataFrame({
        'File': df_fir.index,
        'Average Weighted Score': avg_weighted_scores
    }).sort_values('Average Weighted Score', ascending=True)

    # Plot
    file_names = results_df['File']
    avg_scores = results_df['Average Weighted Score']

    norm = plt.Normalize(min(avg_scores), max(avg_scores))
    cmap = cm.RdYlBu_r
    colors = cmap(norm(avg_scores))

    plt.figure(figsize=(12, 9))
    plt.barh(file_names, avg_scores, color=colors, edgecolor='black')
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Average weighted normalized FIR', fontsize=14)
    plt.title(f'Multi-criteria analysis for {season}', fontweight='bold', fontsize=16)
    plt.grid(linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    # plt.savefig(f'./figs/ranking_multi_criteria_fir_norm_{season}.png', dpi=300)
    plt.show()

    # Exponential scaling factor
    manual_row = pd.DataFrame({
        'File': ['Normal'],
        'Average Weighted Score': [100]
    })  # Normal class is the most important (linked to the concept of reduction FAR - SEB 2024)

    results_df_new = pd.DataFrame({
        'File': df_fir.index,
        'Average Weighted Score': avg_weighted_scores
    }).sort_values('Average Weighted Score', ascending=False)

    results_df_new = pd.concat([manual_row, results_df_new], ignore_index=True)
    results_df_new['rank'] = results_df_new['Average Weighted Score'].rank(method='min', ascending=False)

    # Calcola i pesi esponenziali
    p = 2
    n_tot = len(results_df_new)
    numerators = (n_tot - results_df_new['rank'] + 1) ** p
    denominator = numerators.sum()
    results_df_new['exp_weights'] = numerators / denominator

    return results_df_new
