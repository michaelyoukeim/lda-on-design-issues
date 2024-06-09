import requests

BASE_URL = "https://issues.apache.org/jira/rest/api/2/issue/"


def fetch_issue_details(issue_id: str):
    url = f"{BASE_URL}{issue_id}"
    response = requests.get(url)
    if response.status_code == 200:
        print("Found")
        return response.json()
    else:
        print(f"Failed to fetch details for issue {issue_id}. Status code: {response.status_code}")
        return None
