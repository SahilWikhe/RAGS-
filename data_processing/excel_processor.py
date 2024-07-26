import pandas as pd
from typing import List, Tuple, Dict

class ExcelProcessor:
    def __init__(self):
        pass

    def process_excel(self, file_path: str) -> Dict[str, List[Tuple[str, str]]]:
        xls = pd.ExcelFile(file_path)
        all_data = {}
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            print(f"Processing sheet: {sheet_name}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(df.head())
            sheet_data = self._process_dataframe(df, sheet_name)
            all_data[sheet_name] = sheet_data
            print(f"Processed {len(sheet_data)} rows from {sheet_name}")
        
        return all_data

    def _process_dataframe(self, df: pd.DataFrame, sheet_name: str) -> List[Tuple[str, str]]:
        texts = []
        for idx, row in df.iterrows():
            row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            texts.append((f"{sheet_name}_{idx}", row_text))
            if idx < 5:  # Print the first 5 processed rows for debugging
                print(f"Processed row {idx}: {row_text[:100]}...")  # Print first 100 characters
        return texts