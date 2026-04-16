import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ROUND_1_TRAIN = PROJECT_ROOT / 'data' / 'processed' / 'original' / 'test.csv'
ROUND_1_AUGMENTED = PROJECT_ROOT / 'augmentation_output_roberta' / 'round_1' / 'augmentation_candidates.csv'

ROUND_2_TRAIN = PROJECT_ROOT / 'augmentation_output_roberta' / 'round_2' / 'train.csv'
ROUND_2_AUGMENTED = PROJECT_ROOT / 'augmentation_output_roberta' / 'round_2' / 'augmentation_candidates.csv'

ROUND_3_TRAIN = PROJECT_ROOT / 'augmentation_output_roberta' / 'round_3' / 'train.csv'

def main():
    df_train = pd.read_csv(ROUND_2_TRAIN)
    df_aug = pd.read_csv(ROUND_2_AUGMENTED)

    print(f"Round 1 train size : {len(df_train)}")
    print(f"Augmented samples  : {len(df_aug)}")
    print(f"Train label split  : {df_train['is_sarcastic'].value_counts().to_dict()}")
    print(f"Augmented label split: {df_aug['is_sarcastic'].value_counts().to_dict()}")

    # Keep only columns that exist in both
    common_cols = [c for c in df_train.columns if c in df_aug.columns]
    print(f"Common columns     : {common_cols}")

    df_combined = pd.concat(
        [df_train[common_cols], df_aug[common_cols]],
        ignore_index=True
    )

    print(f"\nRound 2 train size : {len(df_combined)}")
    print(f"Round 2 label split: {df_combined['is_sarcastic'].value_counts().to_dict()}")

    # Save
    df_combined.to_csv(ROUND_3_TRAIN, index=False)
    print(f"\nSaved to: {ROUND_3_TRAIN}")

if __name__ == '__main__':
    main()