import json
import pandas as pd
from config import DEFAULT_DATA_PATH, EXCEL_FILENAME, EXCEL_SHEET_NAME
from excel_utils import read_excel
from data_manipulation import transform_issues
from issue_reader import fetch_issue_details
from preprocess import preprocess_issues, create_vocabulary
import os
from topic_analysis import analyze_topics


def download_issue_details_if_not_exist(filepath: str, df: pd.DataFrame):
    print("Started: fetching data...")
    issue_details_list = []

    for index, row in df.iterrows():
        print("Searching for issue: ", row['Issue key'])
        data = fetch_issue_details(row['Issue key'])
        # print(len(data["fields"]))
        fields = data['fields']
        mapped_data = {
            'id': data['id'],
            'key': data['key'],
            'parent': fields.get('parent', {}),
            'priority': fields.get('priority', {}),
            'versions': fields.get('versions', {}),
            'issueLinks': fields.get('issuelinks', {}),
            'status': fields.get('status', {}),
            'issueType': fields.get('issuetype', {}),
            'comment': fields.get('comment', {}),
            'summary': fields.get('summary', ""),
            'description': fields.get('description', ""),
            'labels': fields.get('labels', []),
            'self': data['self'],
        }
        issue_details_list.append(mapped_data)
        print()

    issue_details_df = pd.DataFrame(issue_details_list)
    print("Finished: fetching data...")

    issue_details_df.to_excel(filepath, index=False)
    print("Saved details in excel file...")


def main():
    print("Started: reading excel file...")
    issues_excel_path = os.path.join(DEFAULT_DATA_PATH, EXCEL_FILENAME)
    issues_excel = read_excel(issues_excel_path, EXCEL_SHEET_NAME)

    print("Finished: reading excel file...")

    print("Started: transforming issues...")
    df = transform_issues(issues_excel)
    print("Finished: transforming issues...")

    excel_output_path = os.path.join(DEFAULT_DATA_PATH, "issue_details.xlsx")

    if os.path.exists(excel_output_path):
        print("Issue details already saved in excel file...")
    else:
        download_issue_details_if_not_exist(excel_output_path, df)

    issue_details_df = pd.read_excel(excel_output_path)

    for index, row in issue_details_df.iterrows():
        print(index)
        print("id: ", row["id"])
        print("self: ", row["self"])
        print("Key: ", row["key"])
        print()

    print("Started: preprocessing issues...")
    preprocessed_df = preprocess_issues(issue_details_df)
    print("Finished: preprocessing issues...")

    preprocessed_output_path = os.path.join(
        DEFAULT_DATA_PATH, 'preprocessed_issue_details.xlsx')
    preprocessed_df.to_excel(preprocessed_output_path, index=False)
    print(f"Saved preprocessed details in {preprocessed_output_path}")

    print("Started: creating vocabulary...")
    vocab_df = create_vocabulary(preprocessed_df)
    vocab_output_path = os.path.join(DEFAULT_DATA_PATH, "vocabulary.xlsx")
    vocab_df.to_excel(vocab_output_path, index=False)
    print(f"Saved vocabulary in {vocab_output_path}")

    # Topic modelling
    analyze_topics()


if __name__ == '__main__':
    main()
