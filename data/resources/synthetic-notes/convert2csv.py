
import os
from glob import glob
import pandas as pd

filenames = sorted(glob(os.path.dirname(__file__) + "/*.txt"))

df = []
for f in filenames:
    with open(f,"r") as the_file:
        f_data = the_file.read().strip()
    df.append({"document_id":os.path.basename(f).rstrip(".txt"), "text":f_data})
df = pd.DataFrame(df)

_ = df.to_csv(os.path.dirname(__file__) + "/synthetic.csv", index=False)
