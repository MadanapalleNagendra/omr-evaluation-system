import pandas as pd
import os

def convert_answer_key(input_path, output_path, set_name='Set A'):
    # Read the original Excel file
    df = pd.read_excel(input_path, header=0)
    # Flatten all columns into a list of (question, answer) tuples
    qa_pairs = []
    for col in df.columns:
        for cell in df[col].dropna():
            # Expect format like '1 - a' or '16 - a,b,c,d'
            parts = str(cell).split('-')
            if len(parts) == 2:
                q = parts[0].strip()
                a = parts[1].strip().replace(' ', '')
                try:
                    q_num = int(q)
                except Exception:
                    continue
                qa_pairs.append((q_num, a))
    # Sort by question number
    qa_pairs.sort(key=lambda x: x[0])
    # Create DataFrame
    out_df = pd.DataFrame(qa_pairs, columns=['Question', 'Answer'])
    # Write to new Excel file with required sheet name
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        out_df.to_excel(writer, sheet_name=set_name, index=False)
    print(f"Converted answer key saved to: {output_path} (sheet: {set_name})")

if __name__ == '__main__':
    input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Key (Set A and B).xlsx')
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'answer_keys.xlsx')
    convert_answer_key(input_path, output_path, set_name='Set A')