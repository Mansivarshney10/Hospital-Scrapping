import pandas as pd
import json

def clean_data(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)

    # Convert data to DataFrame for easier cleaning
    df = pd.DataFrame(data)

    # Example cleaning steps
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_data("../data/raw/scraped_data.json")
    cleaned_data.to_csv("../data/cleaned/cleaned_data.csv", index=False)
