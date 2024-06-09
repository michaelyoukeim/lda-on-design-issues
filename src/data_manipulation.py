import pandas as pd


def transform_issues(df: pd.DataFrame) -> pd.DataFrame:
    split_classes = df['Types of decision'].str.split()

    # Assign each part to the respective column
    df['existence'] = split_classes.str[0]
    df['executive'] = split_classes.str[1]
    df['property'] = split_classes.str[2]

    # Drop the original 'Classes' column
    df.drop(columns=['Types of decision'], inplace=True)
    df['Manual or automatic'] = df['Manual or automatic'].map({'Manual': 0, 'Automatic': 1})
    df['IsAutomatic'] = df['Manual or automatic'].apply(lambda x: True if x == 1 else False)
    df.drop('Manual or automatic', axis=1, inplace=True)

    return df
