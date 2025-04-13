import yaml
from utils.kpi_multi_criterio import (calculate_kpis, kpi_carpet_plot, fir_carpet_plot, fir_norm_carpet_plot,
                                      multi_criteria_fir_norm)

if __name__ == '__main__':
    with open('./configs/config.yml', 'r') as file:
        config = yaml.safe_load(file)

    folder_path = './data/preprocessed'

    n = 10000  # Number of samples to be used for the analysis
    alpha = [1, 1, 1, 1, 1]  # 'Heating energy', 'Cooling energy', 'Fan energy', 'Comfort dT', 'IAQ ratio'

    ### WINTER ###
    df_winter, df_fir_winter, df_fir_norm_winter = calculate_kpis(folder_path, month=1)  # Gennaio

    df_winter.to_csv('./results/kpi_winter.csv', index=True)
    kpi_carpet_plot(df_winter, figsize=(9, 7), season='Winter')

    df_fir_winter.to_csv('./results/fir_winter.csv', index=True)
    fir_carpet_plot(df_fir_winter, figsize=(9, 7), season='Winter')

    df_fir_norm_winter.to_csv('./results/fir_norm_winter.csv', index=True)
    fir_norm_carpet_plot(df_fir_norm_winter, figsize=(9, 7), season='Winter')

    df_multi_winter = multi_criteria_fir_norm(df_fir_norm_winter, n, alpha, season='Winter')
    df_multi_winter.to_csv('./results/ranking_winter.csv', index=False)

    ### SUMMER ###
    df_summer, df_fir_summer, df_fir_norm_summer = calculate_kpis(folder_path, month=7)  # Luglio

    df_summer.to_csv('./results/kpi_summer.csv', index=True)
    kpi_carpet_plot(df_summer, figsize=(9, 7), season='Summer')

    df_fir_summer.to_csv('./results/fir_summer.csv', index=True)
    fir_carpet_plot(df_fir_summer, figsize=(9, 7), season='Summer')

    df_fir_norm_summer.to_csv('./results/fir_norm_summer.csv', index=True)
    fir_norm_carpet_plot(df_fir_norm_summer, figsize=(9, 7), season='Summer')

    df_multi_summer = multi_criteria_fir_norm(df_fir_norm_summer, n, alpha, season='Summer')
    df_multi_summer.to_csv('./results/ranking_summer.csv', index=False)
