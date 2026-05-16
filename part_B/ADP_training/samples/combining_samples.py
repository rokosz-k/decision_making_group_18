import pandas as pd
df = pd.concat([
    pd.read_csv("part_B/ADP_training/samples/samples_sp.csv"),
    pd.read_csv("part_B/ADP_training/samples/samples_fvi.csv")
])
df.to_csv("part_B/ADP_training/samples/samples_sp_fvi.csv", index=False)