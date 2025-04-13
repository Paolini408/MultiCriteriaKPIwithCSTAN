import pandas as pd
import yaml
import os
from utils.preprocessing import remove_transition
from utils.cost_sensitive_tan import features_selection_discrete_entropy, tan_classifier
from utils.performance_metrics import print_performance_and_compute_precision_and_recall, plot_precision_and_recall

if __name__ == "__main__":
    with open('./configs/config.yml', 'r') as file:
        config = yaml.safe_load(file)

    label_column = 'label_diagnosis'

    # 1. Unisci tutti i CSV del folder dopo un secondo preprocessing
    folder_path = './data/preprocessed'
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    preprocessed_files = []
    for f in all_files:
        df = pd.read_csv(f)
        df = remove_transition(df)  # Removing transition between different operational modes
        df = df[df['slope'] != 'transient']  # Removing transient data
        df = df[df[config['fan_power']] > 0.1]  # Removing data when FCU is off
        df = df[df['Operational_mode'] != 'Shutdown mode']  # During this mode the system is off!
        preprocessed_files.append(df)

    df_tot = pd.concat(preprocessed_files, ignore_index=True)

    # 2. Sostituzione etichette
    for col in ['label_detection', 'label_isolation', 'label_diagnosis']:
        df_tot[col] = df_tot[col].replace({'Normal new': 'Normal', 'Normal old': 'Normal'})

    # 3. Seleziona le variabili per l'analisi
    selected_columns = [config['t_room'],
                        config['t_cooling_sp'],
                        config['t_heating_sp'],
                        config['mat'],
                        config['rat'],
                        config['dat'],
                        config['oat'],
                        config['discharge_flow'],
                        config['outdoor_flow'],
                        config['p_cooling'],
                        config['p_heating'],
                        config['fan_power'],
                        config['fan_rpm'],
                        config['ccv_command'],
                        config['hcv_command'],
                        config['oad_command'],
                        config['ccv_position'],
                        config['hcv_position'],
                        config['oad_position'],
                        ]  # 19 features

    X_tot = df_tot[selected_columns]
    Y_tot = df_tot[label_column]

    # 4. Filtraggio per mese
    df_winter = df_tot[df_tot['Month'].isin([11, 12, 1, 2])].copy()
    df_summer = df_tot[df_tot['Month'].isin([6, 7, 8, 9])].copy()

    # 5. Creazione X e Y per stagioni
    X_winter = df_winter[selected_columns]
    Y_winter = df_winter[label_column]

    X_summer = df_summer[selected_columns]
    Y_summer = df_summer[label_column]

    # 6. Parametri
    n_input_features = 15  # Number of input features to be selected (based on availability and complexity)
    max_bins = 5  # Maximum number of bins for discretization

    ### WINTER ###
    # Discretization and feature selection
    X_columns_tan_winter = features_selection_discrete_entropy(X_winter, Y_winter, max_bins, n_input_features)
    lista_col_tan_winter = X_columns_tan_winter.columns.tolist()

    # Set the cost ratio for each class
    sorted_labels = sorted(df_winter[label_column].unique())
    equal_weights_winter = False  # Set to True if you want equal weights for all classes (fault and normal)
    if equal_weights_winter:
        # Equal weights for all classes
        cost_ratio_list_winter = [1] * (len(sorted_labels))  # Cost ratio for all faults (Normal is the last class)
        n_scenario = 1
    else:
        # Exponential weights for all classes
        weights_df_winter = pd.read_csv('./results/ranking_winter.csv')
        weights_dict_winter = dict(zip(weights_df_winter['File'], weights_df_winter['exp_weights']))
        cost_ratio_list_winter = [weights_dict_winter[label] for label in sorted_labels]
        n_scenario = 2

    # Cost sensitive TAN classifier
    y_test_winter, y_pred_winter = tan_classifier(X_columns_tan_winter, Y_winter, lista_col_tan_winter, sorted_labels,
                                                  cost_ratio_list_winter, label_column, 'Winter', n_scenario)

    # Performance metrics
    precision_winter, recall_winter = print_performance_and_compute_precision_and_recall(y_test_winter, y_pred_winter,
                                                                                         sorted_labels, 'TAN',
                                                                                         'Winter', n_scenario)

    plot_precision_and_recall(precision_winter, recall_winter, sorted_labels, 'Winter', n_scenario)

    ### SUMMER ###
    # Discretization and feature selection
    X_columns_tan_summer = features_selection_discrete_entropy(X_summer, Y_summer, max_bins, n_input_features)
    lista_col_tan_summer = X_columns_tan_summer.columns.tolist()

    # Set the cost ratio for each class
    sorted_labels = sorted(df_summer[label_column].unique())
    equal_summer = False  # Set to True if you want equal weights for all classes (fault and normal)
    if equal_summer:
        # Equal weights for all classes
        cost_ratio_list_summer = [1] * (len(sorted_labels))  # Cost ratio for all faults (Normal is the last class)
        n_scenario = 1
    else:
        # Exponential weights for all classes
        weights_df_summer = pd.read_csv('./results/ranking_summer.csv')
        weights_dict_summer = dict(zip(weights_df_summer['File'], weights_df_summer['exp_weights']))
        cost_ratio_list_summer = [weights_dict_summer[label] for label in sorted_labels]
        n_scenario = 2

    # Cost sensitive TAN classifier
    y_test_summer, y_pred_summer = tan_classifier(X_columns_tan_summer, Y_summer, lista_col_tan_summer, sorted_labels,
                                                  cost_ratio_list_summer, label_column, 'Summer', n_scenario)

    # Performance metrics
    precision_summer, recall_summer = print_performance_and_compute_precision_and_recall(y_test_summer, y_pred_summer,
                                                                                         sorted_labels, 'TAN',
                                                                                         'Summer', n_scenario)

    plot_precision_and_recall(precision_summer, recall_summer, sorted_labels, 'Summer', n_scenario)
