import pandas as pd
df = pd.concat([
    pd.read_csv("part_B/ADP_training/samples/samples_sp_fvi.csv"),
    pd.read_csv("part_B/ADP_training/samples/samples_fvi_v4.csv")
])
df.to_csv("part_B/ADP_training/samples/samples_sp_fvi_v2.csv", index=False)