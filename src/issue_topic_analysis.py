import pandas as pd
import matplotlib.pyplot as plt
import os
from config import DEFAULT_DATA_PATH
import seaborn as sns

def read_issue_details():
    issue_details_path = os.path.join(DEFAULT_DATA_PATH, 'issue_details_per_topic.csv')
    return pd.read_csv(issue_details_path)

def read_issues_file():
    issues_path = os.path.join(DEFAULT_DATA_PATH, 'Issues.xlsx')
    return pd.read_excel(issues_path)

def generate_manual_vs_automatic_graph(show_plot=False):
    issue_details_df = read_issue_details()
    issues_df = read_issues_file()

    # Flatten the Issues column from issue_details_df
    issue_details_df['Issues'] = issue_details_df['Issues'].apply(lambda x: x.strip('[]').replace("'", "").split(', '))

    # Explode the DataFrame so each issue is on its own row
    exploded_issue_details_df = issue_details_df.explode('Issues')

    # Merge with the issues_df to get "Manual or automatic" information
    merged_df = pd.merge(exploded_issue_details_df, issues_df, left_on='Issues', right_on='Issue key')

    # Group by Topic and Manual/Automatic, and count the number of issues
    grouped = merged_df.groupby(['Topic', 'Manual or automatic']).size().unstack(fill_value=0)

    # Plotting with better colors
    sns.set(style="whitegrid")
    ax = grouped.plot(kind='bar', stacked=True, figsize=(12, 6), color=sns.color_palette("Paired", 2), edgecolor='black')
    plt.xlabel("Topics", fontsize=14)
    plt.ylabel("Number of Issues", fontsize=14)
    plt.title("Number of Manual and Automatic Issues per Topic", fontsize=16)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Issue Type')

    if show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(DEFAULT_DATA_PATH, 'manual_vs_automatic_issues_per_topic.pdf'))
    print("Graph generated successfully.")

# Example usage:
if __name__ == "__main__":
    generate_manual_vs_automatic_graph(show_plot=False)
