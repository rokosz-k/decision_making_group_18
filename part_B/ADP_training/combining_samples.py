import pandas as pd
df = pd.concat([
    pd.read_csv("part_B/ADP_training/samples_dummy.csv"),
    pd.read_csv("part_B/ADP_training/samples_adp_v1.csv"),
    pd.read_csv("part_B/ADP_training/samples_sp.csv"),
    pd.read_csv("part_B/ADP_training/samples_adp_new_features.csv")
])
df.to_csv("part_B/ADP_training/samples_mixed_v2.csv", index=False)