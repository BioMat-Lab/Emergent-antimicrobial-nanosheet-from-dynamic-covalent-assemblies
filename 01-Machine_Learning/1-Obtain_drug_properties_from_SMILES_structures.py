import os
import subprocess
import venv

import pandas as pd
import pubchempy as pcp
from padelpy import from_smiles
from rdkit import Chem
from tqdm import tqdm


class Folder:
    def __init__(self, root=None, data=None, res=None, code=None):
        self.root = root
        self.data = data
        self.res = res
        self.code = code


folder = Folder(
    root="D:/2025_04.Drug_pred_sys",
    data="07.data",
    res="08.result",
    code="09.pp.code",
)

folders = ["data", "src", "results", "docs", "env"]

files = {
    "README.md": "# Project Title\n\nProject description and usage instructions.\n",
    "requirements.txt": "padelpy\npandas\ntqdm\nopenpyxl\npubchempy\nrdkit\n",
    "src/main.py": (
        "def main():\n"
        "    pass\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    ),
    "src/utils.py": "",
}


def create_folders(folder_list):
    for path in folder_list:
        os.makedirs(path, exist_ok=True)


def create_files(file_dict):
    for file_path, content in file_dict.items():
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def create_project_structure():
    create_folders(folders)
    create_files(files)


def create_virtualenv(env_dir):
    os.makedirs(env_dir, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(env_dir)


def install_packages(venv_path, packages, mirror="https://mirrors.aliyun.com/pypi/simple/"):
    pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
    if not os.path.exists(pip_exe):
        raise FileNotFoundError(f"pip not found: {pip_exe}")

    for package in packages:
        subprocess.run([pip_exe, "install", package, "-i", mirror], check=True)


def update_smiles_in_excel(excel_file, sheet_name, data_dir):
    os.chdir(data_dir)
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    cid_list = []
    formula_list = []
    weight_list = []
    pc_smiles_list = []
    canonical_smiles_list = []

    for _, row in df.iterrows():
        compound_name = row["Name"]
        try:
            compound = pcp.get_compounds(compound_name, "name")[0]
            cid = compound.cid
            formula = compound.molecular_formula
            weight = compound.molecular_weight
            pc_smiles = compound.canonical_smiles
        except Exception:
            cid, formula, weight, pc_smiles = None, None, None, None

        cid_list.append(cid)
        formula_list.append(formula)
        weight_list.append(weight)
        pc_smiles_list.append(pc_smiles)

        if pc_smiles is not None:
            mol = Chem.MolFromSmiles(pc_smiles)
            canonical_smiles = Chem.MolToSmiles(mol) if mol else None
        else:
            canonical_smiles = None

        canonical_smiles_list.append(canonical_smiles)

    df["CID"] = cid_list
    df["Formula"] = formula_list
    df["Weight"] = weight_list
    df["PC_SMILES"] = pc_smiles_list
    df["Canonical_SMILES"] = canonical_smiles_list

    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def calculate_descriptors(smiles):
    try:
        return from_smiles(smiles)
    except Exception:
        return {}


def extract_features_from_excel(excel_file, sheet_name, data_dir, result_dir, output_file):
    os.chdir(data_dir)
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    if "Canonical_SMILES" not in df.columns:
        raise ValueError("Column 'Canonical_SMILES' not found.")

    tqdm.pandas(desc="Calculating descriptors")
    descriptors_list = df["Canonical_SMILES"].progress_apply(calculate_descriptors)
    descriptors_df = pd.DataFrame(descriptors_list.tolist())

    df_result = pd.concat([df, descriptors_df], axis=1)

    os.chdir(result_dir)
    df_result.to_csv(output_file, index=False)


if __name__ == "__main__":
    excel_file = "compound_list.xlsx"
    sheet_name = "1.Actual_Aldyhydes"

    data_dir = os.path.join(folder.root, folder.data)
    result_dir = os.path.join(folder.root, folder.res)
    env_dir = os.path.join(folder.root, ".venv")

    packages = ["padelpy", "pandas", "tqdm", "openpyxl"]

    create_project_structure()
    create_virtualenv(env_dir)
    install_packages(env_dir, packages)
    update_smiles_in_excel(excel_file, sheet_name, data_dir)
    extract_features_from_excel(
        excel_file=excel_file,
        sheet_name=sheet_name,
        data_dir=data_dir,
        result_dir=result_dir,
        output_file="result_features-2.csv",
    )