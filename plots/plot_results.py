import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from consts.paths import Paths


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hash", help="Hash corresponding to an explainability experiment", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    sns.set(style="whitegrid")

    hash_df = pd.read_csv(Paths.explainability_csv)
    explainability_experiment = hash_df[hash_df['hash'] == args.hash]
    dataset_name = explainability_experiment['dataset'].to_list()[0].split('_')[0]
    xai_method = explainability_experiment['xmethod'].to_list()[0]

    hash_df = pd.read_csv(os.path.join(Paths.explainability_path, f'{args.hash}.csv'))

    purple_palette = ['#f5bc76', '#edeccf', '#c4e6ea', '#f09881']

    hash_df = hash_df.drop(['filename'], axis=1)
    plt.figure(figsize=(10, 6))
    g = sns.boxplot(data=hash_df, palette=purple_palette)
    g.set_xticklabels(['Attribute-Precision', 'Attribute-Accuracy', 'Attribute-F1', 'Attribute-Recall'])
    plt.title(f'Dataset: {dataset_name} -- XAI method: {xai_method}')
    plt.show()