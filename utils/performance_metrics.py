import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def compute_far(conf_matrix_df):
    row_sum_without_N = conf_matrix_df.loc['Normal', :].sum() - conf_matrix_df.loc['Normal', 'Normal']
    row_sum_total = conf_matrix_df.loc['Normal', :].sum()
    rapporto = row_sum_without_N / row_sum_total
    false_alarm_rate = rapporto * 100
    return false_alarm_rate


def compute_fnr(conf_matrix_df):
    col_sum_without_WN = conf_matrix_df.loc[:, 'Normal'].sum() - conf_matrix_df.loc['Normal', 'Normal']
    col_sum_total = conf_matrix_df.loc[:, 'Normal'].sum()
    rapporto = col_sum_without_WN / col_sum_total
    false_negative_rate = rapporto * 100
    return false_negative_rate


def compute_mdr(conf_matrix_df):
    misdiagnoses = 0
    total_samples = 0
    for index, row in conf_matrix_df.iterrows():
        for column, value in row.items():
            if index != "Normal" and column != "Normal" and index != column:
                misdiagnoses += value
            if index != "Normal" and column != "Normal":
                total_samples += value
    MDR = (misdiagnoses / total_samples) * 100
    return MDR


def print_performance_and_compute_precision_and_recall(y_test, y_pred, lab_sorted, classifier, season, n_scenario):
    conf_matrix = confusion_matrix(y_test, y_pred)  # True labels in rows and predicted labels in columns
    conf_matrix_df = pd.DataFrame(conf_matrix, index=lab_sorted, columns=lab_sorted)
    conf_matrix_df.to_csv(f'./results/confusion_matrix_{season}_Scenario_{n_scenario}.csv', index=True, header=True)
    print(f"\nConfusion Matrix - {classifier}:")
    print(conf_matrix_df)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy - {classifier}:", accuracy * 100, '%')

    far = compute_far(conf_matrix_df)  # False Positive Rate (1° type error)
    print(f"False Alarm Rate - {classifier}:", far, '%')

    fnr = compute_fnr(conf_matrix_df)  # False Negative Rate (2° type error)
    print(f"False Negative Rate - {classifier}:", fnr, '%')

    mdr = compute_mdr(conf_matrix_df)
    print(f"Mis-Diagnosis Rate - {classifier}:", mdr, '%')

    precision = precision_score(y_test, y_pred, average=None, labels=lab_sorted)
    print(f"\nPrecision per class - {classifier}:")
    for i, class_name in enumerate(lab_sorted):
        print(f"{class_name}: {precision[i]}")

    recall = recall_score(y_test, y_pred, average=None, labels=lab_sorted)
    print(f"\nRecall per class - {classifier}:")
    for i, class_name in enumerate(lab_sorted):
        print(f"{class_name}: {recall[i]}")

    return precision, recall


def plot_precision_and_recall(precision, recall, lab_sorted, season, n_scenario):
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    bar_width = 0.14
    index = np.arange(len(lab_sorted))
    ax.bar(index, precision * 100, bar_width, label='Precision', color='orange', edgecolor='black')
    ax.bar(index + bar_width, recall * 100, bar_width, label='Recall', color='royalblue', edgecolor='black')
    ax.set_title(f'Precision [%] and Recall [%] during {season} (Scenario {n_scenario})', fontweight='bold', fontsize=12)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(lab_sorted, rotation=90)
    ax.set_ylim(0, 110)
    ax.legend(ncols=2, loc='upper center')
    ax.grid(linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f'./figs/precision_recall_{season}_Scenario_{n_scenario}.png', dpi=300, bbox_inches='tight')
    plt.show()
