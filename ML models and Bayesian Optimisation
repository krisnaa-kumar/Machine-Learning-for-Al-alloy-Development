
import pandas as pd
import numpy as np

FILE = "Individual Project Dataset-1.xlsx"

df = pd.read_excel(
    FILE,
    sheet_name="My data",
    header=0,         
    engine="openpyxl"
)
df.columns = df.columns.str.strip()


# Dropping rows missing your targets, if any
df = df.dropna(subset=["UTS (MPa)", "Elongation"]).reset_index(drop=True)
row0 = df.iloc[0]

# Reading Materials Project sheet:
desc_df = pd.read_excel(
    FILE,
    sheet_name="Materials Project",
    header=0,
    index_col=0,
    engine="openpyxl"
)
desc_df.index   = desc_df.index.str.strip()
desc_df.columns = desc_df.columns.str.strip()

df = df.drop(columns=["C","O"], errors="ignore")

composition_cols = [
    "Si","Fe","Cu","Be","Ag","Bi","Pb","Zn",
    "Mn","Mg","Sn","Ti","V","Mo","Ni","Ce","Co","Cr",
    "Li","Sc","Sr","Zr","Al"
]

# Sanity‐check that all 25 elements exist in both dataframes
missing_in_df    = [c for c in composition_cols if c not in df.columns]
missing_in_desc  = [c for c in composition_cols if c not in desc_df.columns]

if missing_in_df:
    raise KeyError(f"The following elements are in your list but missing from My data sheet: {missing_in_df}")
if missing_in_desc:
    raise KeyError(f"The following elements are in your list but missing from Materials Project sheet: {missing_in_desc}")

print("All composition columns found. Proceeding with:", composition_cols)



# Building weight‐fraction matrix, filling blanks with zero wt%
W = (
    df[composition_cols]
      .fillna(0.0)           
      .astype(float)
      .div(100, axis=0)      
)

# Computing p_mean and p_std for each descriptor
n = len(composition_cols)
df_feats = pd.DataFrame(index=df.index)
for prop in desc_df.index:      # each of 42 descriptor names
    p_i    = desc_df.loc[prop, composition_cols]  
    mean_s = W.dot(p_i)                            
    tmp    = W.mul(p_i, axis=1)                   
    var_s  = ((tmp.sub(mean_s, axis=0))**2).sum(axis=1) / n
    std_s  = var_s**0.5

    df_feats[f"{prop}_mean"] = mean_s
    df_feats[f"{prop}_std"]  = std_s

with pd.ExcelWriter(FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_feats.to_excel(writer, sheet_name="element descriptors", index=False)

print(f"✅ Wrote {df_feats.shape[1]} descriptor columns to “element descriptors”")







