import pandas as pd


def read_excel(filepath: str, sheet_name: str) -> pd.DataFrame:
    """
        Read data from an Excel file and return it as a pandas DataFrame.

        Parameters:
        filepath (str): The path to the Excel file.
        sheet_name (str): The name of the Excel sheet to read.

        Returns:
        pd.DataFrame: A DataFrame containing the data read from the Excel file.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df
