import os
import pandas as pd

# Label mappings — verify against actual TSV values before running
LABEL_2WAY = {0: "real", 1: "fake"}
LABEL_6WAY = {
    0: "true",
    1: "satire",
    2: "misleading",
    3: "imposter",
    4: "false",
    5: "manipulated",
}


def load_split(tsv_path, image_dir, label_type="2_way"):
    """
    Load a Fakeddit TSV split and return a filtered DataFrame.

    Args:
        tsv_path:   path to train/validate/test.tsv
        image_dir:  path to public_image_set/
        label_type: "2_way" or "6_way"

    Returns:
        DataFrame with columns: id, clean_title, label, image_path
    """
    df = pd.read_csv(tsv_path, sep="\t")

    df["image_path"] = df["id"].apply(
        lambda x: os.path.join(image_dir, f"{x}.jpg")
    )

    before = len(df)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"[dataloader] {os.path.basename(tsv_path)}: {len(df)}/{before} samples have images.")

    label_col = f"{label_type}_label"
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Available: {df.columns.tolist()}")

    df = df[["id", "clean_title", label_col, "image_path"]].rename(
        columns={label_col: "label"}
    )

    return df


def get_label_names(label_type="2_way"):
    mapping = LABEL_2WAY if label_type == "2_way" else LABEL_6WAY
    return [mapping[i] for i in sorted(mapping)]
