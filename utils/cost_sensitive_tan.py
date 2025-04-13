import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
import networkx as nx


### EQUAL WIDTH DISCRETIZATION ###
def features_selection_discrete(X, Y, n_bins, n_features):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_bins = discretizer.fit_transform(X)
    X_discrete = pd.DataFrame(X_bins, columns=X.columns)
    mi_scores = mutual_info_classif(X_discrete, Y)
    mi_scores_df = pd.DataFrame({'Feature': X_discrete.columns, 'MI Score': mi_scores})
    mi_scores_sorted = mi_scores_df.sort_values(by='MI Score', ascending=False).reset_index(drop=True)
    selected_features = mi_scores_sorted.iloc[:n_features]['Feature'].tolist()
    X_selected_disc = X_discrete[selected_features]
    return X_selected_disc


### ENTROPY DISCRETIZATION ###
def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return entropy(probabilities, base=2)


def weighted_entropy_multiple(x_train, y_train, cut_points):
    bins = [-np.inf] + sorted(cut_points) + [np.inf]
    bin_indices = np.digitize(x_train, bins) - 1  # Output: bin membership list
    weighted_entropy = 0
    for bin_index in range(len(bins) - 1):
        bin_mask = bin_indices == bin_index  # Output: array booleano (True False ... False)
        bin_entropy = calculate_entropy(y_train[bin_mask])
        weighted_entropy += (np.sum(bin_mask) * bin_entropy) / len(y_train)
    return weighted_entropy


def find_best_splits(feature_data, y_train, n_bins_max):
    # Sorted column of x_train['feature'] maintaining the correspondence with y_train
    combined = sorted(zip(feature_data, y_train), key=lambda x: x[0])
    reduced_combined = combined[::10]  # Select every k-th element (k=1,10,100)
    # Unpack the reduced combined pairs back into separate lists
    reduced_feature_data, reduced_y_train = zip(*reduced_combined)
    reduced_feature_data = np.array(reduced_feature_data)
    reduced_y_train = np.array(reduced_y_train)
    unique_values = np.unique(reduced_feature_data)  # Only unique values (different from each other)
    selected_cut_points = []
    while len(selected_cut_points) < (n_bins_max - 1):  # bins = cut points + 1
        best_cut_point = None
        min_entropy = float('inf')  # Evaluate min_entropy < inf, otherwise n_bins will always be equal to n_bins_max
        for i in range(len(unique_values) - 1):
            cut_point = (unique_values[i] + unique_values[i + 1]) / 2
            if cut_point in selected_cut_points:
                continue
            current_entropy = weighted_entropy_multiple(reduced_feature_data, reduced_y_train,
                                                        selected_cut_points + [cut_point])
            if current_entropy < min_entropy:
                min_entropy = current_entropy
                best_cut_point = cut_point
        if best_cut_point is not None:
            selected_cut_points.append(best_cut_point)
        else:
            break
    return [-np.inf] + sorted(selected_cut_points) + [np.inf]


def obtain_X_entropy_disc(x_train, y_train, n_bins_max):
    best_splits = {}
    X_entropy = pd.DataFrame()
    for col in x_train.columns:
        best_splits[col] = find_best_splits(x_train[col], y_train, n_bins_max)
        X_entropy[col] = np.digitize(x_train[col], best_splits[col], right=False)
    return X_entropy


def features_selection_discrete_entropy(X, Y, n_bins_max, n_features):
    X_discrete = obtain_X_entropy_disc(X, Y, n_bins_max)
    mi_scores = mutual_info_classif(X_discrete, Y)
    mi_scores_df = pd.DataFrame({'Feature': X_discrete.columns, 'MI Score': mi_scores})
    mi_scores_sorted = mi_scores_df.sort_values(by='MI Score', ascending=False).reset_index(drop=True)
    selected_features = mi_scores_sorted.iloc[:n_features]['Feature'].tolist()
    X_selected_disc = X_discrete[selected_features]
    return X_selected_disc


### COST-SENSITIVE TREE-AUGMENTED NAIVE BAYES (TAN) ###
def obtain_weighted_cost_list(df_train, cost_ratio_list, label_col):
    """
    Obtain the weighted cost for each class label.
    """
    N = len(df_train)
    class_counts = df_train[label_col].value_counts().sort_index()
    weight = {}
    for i, class_label in enumerate(class_counts.index):
        weight[class_label] = (cost_ratio_list[i] * N) / (class_counts * cost_ratio_list).sum()
    return weight


def conditional_mutual_info_score_weighted(df_train, xi, xj, label_col, cost_ratios):
    """
    Compute the weighted conditional mutual information I(Xi, Xj | C) for each label C.
    """
    weighted_cmi = 0
    for label in df_train[label_col].unique():
        subset = df_train[df_train[label_col] == label]
        cmi = mutual_info_score(subset[xi], subset[xj])
        weighted_cmi += cmi * cost_ratios[label]
    return weighted_cmi


def construct_max_spanning_tree_weighted(df_train, features, label_col, cost_ratio_list):
    """
    Construct a maximum spanning tree using weighted conditional mutual information as edge weights.
    """
    G = nx.Graph()
    cost_ratios = obtain_weighted_cost_list(df_train, cost_ratio_list, label_col)

    # Add edges between features with weighted CMI as weights
    for xi in features:
        for xj in features:
            if xi != xj:
                weight = conditional_mutual_info_score_weighted(df_train, xi, xj, label_col, cost_ratios)
                G.add_edge(xi, xj, weight=weight)

    # Construct maximum spanning tree
    return nx.maximum_spanning_tree(G)


def construct_TAN_weighted(df_train, features, label_col, cost_ratio_list, root_node=None):
    """
    Construct a Tree-Augmented Naive Bayes model.
    """
    # Step 1: Construct the maximum spanning tree
    tree = construct_max_spanning_tree_weighted(df_train, features, label_col, cost_ratio_list)

    # Check if the tree is created successfully
    if tree is None:
        print("Failed to construct the maximum spanning tree.")
        return None

    # Step 2: Orient the edges to form a directed acyclic graph (DAG)
    root_feature = root_node if root_node else features[0]
    dag = nx.bfs_tree(tree, root_feature)

    # Step 3: Connect the class node to all feature nodes
    for node in tree.nodes:
        dag.add_edge(label_col, node)

    return dag


def prior_probs(sorted_labels, df_train, x_train, y_train, cost_ratio_list, label_col):
    """
    Compute the prior probabilities for each class label.
    """
    nc = len(sorted_labels)
    class_counts = df_train[label_col].value_counts().sort_index()
    weight = obtain_weighted_cost_list(df_train, cost_ratio_list, label_col)

    prior_cpt = pd.DataFrame(index=["Prior"], columns=sorted_labels, dtype=float).fillna(0)
    for class_label in sorted_labels:
        prior_p_weighted = (weight[class_label]*x_train[y_train == class_label].shape[0] + 1) /\
                           ((list(weight.values())*class_counts).sum() + nc)
        prior_cpt[class_label] = prior_p_weighted

    return prior_cpt


def root_node_prob(df_train, features, cost_ratio_list, sorted_labels, label_col, root_node=None):
    """
    Compute the conditional probability table (CPT) for the root node.
    """
    weight = obtain_weighted_cost_list(df_train, cost_ratio_list, label_col)
    root_node = root_node if root_node else features[0]
    root_values = sorted(df_train[root_node].unique())
    cpt_root_node = pd.DataFrame(index=root_values, columns=sorted_labels, dtype=float).fillna(0)
    for label in sorted_labels:
        num_label = len(df_train[df_train[label_col] == label])
        for value in root_values:
            num_value_and_label = len(df_train[(df_train[root_node] == value) & (df_train[label_col] == label)])
            prob = ((weight[label] * num_value_and_label) + 1) / (weight[label]*num_label + len(root_values))
            cpt_root_node.loc[value, label] = prob
    return cpt_root_node


def get_parents_for_all_nodes(dag, root_node, label_col):
    """
    Get the parent nodes for each node in the TAN structure.
    """
    parents = {}
    for node in dag.nodes():
        # Skip the root node and label/class node since they do not have the same structure of parents
        if node == root_node or node == label_col:
            continue
        # Get the predecessors in the DAG which are the parents of the node
        parents[node] = list(dag.predecessors(node))
    return parents


def create_cpt_for_node(df_train, node, parents, label_col, cost_ratio_list):
    """
    Create a Conditional Probability Table (CPT) for a given node (parent and child) in the TAN structure.
    """
    cost_ratios = obtain_weighted_cost_list(df_train, cost_ratio_list, label_col)
    node_values = sorted(df_train[node].unique())
    parent_label_values = sorted(df_train[label_col].unique())
    other_parent = parents[0] if parents[1] == label_col else parents[1]
    other_parent_values = sorted(df_train[other_parent].unique())
    # Create a MultiIndex for the columns with all combinations of parent label values and other parent values
    multi_index = pd.MultiIndex.from_product([parent_label_values, other_parent_values],
                                             names=[label_col, other_parent])
    # Initialize the CPT DataFrame with the MultiIndex and one row for each node value
    cpt = pd.DataFrame(index=node_values, columns=multi_index, dtype=float).fillna(0)
    # Calculate the probabilities for the CPT
    for node_value in node_values:
        for label_value in parent_label_values:
            for other_parent_value in other_parent_values:
                # Filter the dataframe for the current combination of parent values
                df_filtered = df_train[(df_train[label_col] == label_value) & (df_train[other_parent] == other_parent_value)]

                # Number of occurrences for the current node value given the parent values
                num_value_given_parents = len(df_filtered[df_filtered[node] == node_value])

                # Total number of occurrences for the current parent values
                total_given_parents = len(df_filtered)

                # Calculate the probability with Laplace smoothing
                prob = ((cost_ratios[label_value] * num_value_given_parents) + 1) / (cost_ratios[label_value]*total_given_parents + len(node_values))

                # Set the probability in the CPT
                cpt.loc[node_value, (label_value, other_parent_value)] = prob

    # Sort the CPT by its MultiIndex columns and index rows
    cpt.sort_index(axis=0, inplace=True)  # Sort rows
    cpt.sort_index(axis=1, inplace=True)  # Sort columns
    return cpt


def create_all_feature_cpts(df_train, dag, label_col, cost_ratio_list, root_node):
    """
    Create Conditional Probability Tables (CPTs) for all nodes in the TAN structure.
    """
    cpts = {}
    node_parents = get_parents_for_all_nodes(dag, root_node, label_col)

    # Exclude the root node and label from the nodes to process
    nodes_to_process = set(dag.nodes()) - {root_node, label_col}
    parents = {}
    for node in nodes_to_process:
        parents[node] = node_parents[node]
        cpt = create_cpt_for_node(df_train, node, node_parents[node], label_col, cost_ratio_list)
        cpts[node] = cpt
    return cpts


def lookup_CS_cpd_value(cpt, feature_value, label_value, other_parent_value=None):
    """
    Retrieve the conditional probability value from the CPT.
    """
    if other_parent_value is None:
        # If no other parent, assume it's the prior or root CPT
        return cpt.loc[feature_value, label_value]
    else:
        # For non-root nodes with two parents
        return cpt.loc[feature_value, (label_value, other_parent_value)]


def CS_TAN_prediction(df_test, dag, prior_cpt, root_cpt, cpts, sorted_labels, root_node, label_col):
    """
    Perform prediction of the class using the Cost-Sensitive Tree-Augmented Naive Bayes model.
    """
    y_pred = []
    for _, row in df_test.iterrows():
        evidence = row.to_dict()
        label_probs = {}
        for label in sorted_labels:
            product = prior_cpt.loc['Prior', label]  # Start with prior probability
            product *= lookup_CS_cpd_value(root_cpt, evidence[root_node], label)  # Multiply with root node probability
            # Multiply with probabilities from other CPTs
            for feature, cpt in cpts.items():
                if feature != root_node:  # Skip root node as it's already considered
                    other_parent = [p for p in dag.predecessors(feature) if p != label_col][0]
                    product *= lookup_CS_cpd_value(cpt, evidence[feature], label, evidence[other_parent])
            label_probs[label] = product
        # Select the label with the highest product value
        best_label = max(label_probs, key=label_probs.get)
        y_pred.append(best_label)
    return y_pred


def tan_classifier(X, Y, lista_col, sorted_labels, cost_ratio_list, label_col, season, n_scenario):
    """
    Perform classification using the Cost-Sensitive Tree-Augmented Naive Bayes model.

    Args:
    - X (DataFrame): The pandas DataFrame containing the input features.
    - Y (Series): The pandas Series containing the class labels.
    - lista_col (list): A list of feature variables (from X) to be used for classification.
    - sorted_labels (list): A sorted list of unique class labels in the dataset.
    - cost_ratio_list (list): A list of mis-classification cost for each class label. Higher is the cost for a defined class, higher is the priority of the model to classify that class.
    - label_col (str): The name of the column containing the class labels.
    - season (str): The season for which the classification is performed.
    - n_scenario (int): The scenario number for which the classification is performed.

    Returns:
    - Tree Structure (Graph): The constructed network for TAN.
    - y_test (list): True class labels for the test data.
    - y_pred (list): Predicted class labels for the test data.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    df_train = pd.concat([x_train, y_train], axis=1)

    x_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    df_test = pd.concat([x_test, y_test], axis=1)

    CS_TAN_model = construct_TAN_weighted(df_train, lista_col, label_col, cost_ratio_list)
    pos = nx.circular_layout(CS_TAN_model)
    nx.draw(CS_TAN_model, pos, with_labels=True)
    plt.savefig(f'./figs/struttura_{season}_Scenario_{n_scenario}.png', dpi=300)  # Salva come PNG
    plt.show()  # Display the TAN structure to the user

    prior_p = prior_probs(sorted_labels, df_train, x_train, y_train, cost_ratio_list, label_col)
    # print(prior_p)
    root_p = root_node_prob(df_train, lista_col, cost_ratio_list, sorted_labels, label_col)
    # print(root_p)
    others_p = create_all_feature_cpts(df_train, CS_TAN_model, label_col, cost_ratio_list, lista_col[0])
    # print(others_p)

    y_pred = CS_TAN_prediction(df_test, CS_TAN_model, prior_p, root_p, others_p, sorted_labels, lista_col[0], label_col)

    return y_test, y_pred
