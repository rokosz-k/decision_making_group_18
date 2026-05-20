import pandas as pd
df = pd.concat([
    pd.read_csv("part_B/ADP_training/samples/samples_sp_fvi_v2.csv"),
    pd.read_csv("part_B/ADP_training/samples/samples_fvi_v5.csv")
])
df.to_csv("part_B/ADP_training/samples/samples_sp_fvi_v3.csv", index=False)