import pandas as pd
import os

def csv_to_latex(csv_path, output_path="table_a1.tex"):
    df = pd.read_csv(csv_path, index_col=0)
    latex = """
    \\begin{table}[h]
    \\caption{Preliminary Results on MSFT (2020-07-01 to 2020-08-15, M1 Pro)}
    \\label{tab:prelim}
    \\begin{tabular}{lcccc}
    \\toprule
    Agent & CR (\\%) & SR & AV (\\%) & MDD (\\%) \\\\
    \\midrule
    """
    for idx, row in df.iterrows():
        latex += f"{idx} & {row['cr']} & {row['sr']} & {row['av']} & {row['mdd']} \\\\\n"
    latex += """
    \\bottomrule
    \\end{tabular}
    \\footnotesize{Note: Preliminary, Seed=42, to be expanded in future work.}
    \\end{table}
    """
    os.makedirs("latex", exist_ok=True)
    with open(os.path.join("latex", output_path), "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to latex/{output_path}")

if __name__ == "__main__":
    csv_to_latex("logs/demo_msft.csv")
