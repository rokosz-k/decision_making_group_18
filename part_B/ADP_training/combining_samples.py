import pandas as pd
df = pd.concat([
    pd.read_csv("part_B/ADP_training/samples_dummy.csv"),
    pd.read_csv("part_B/ADP_training/samples_adp_v1.csv"),
    pd.read_csv("part_B/ADP_training/samples_sp.csv"),
])
df.to_csv("part_B/ADP_training/samples_mixed.csv", index=False)