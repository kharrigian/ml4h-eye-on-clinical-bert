
import os
import json
from glob import glob

filenames = sorted(glob(os.path.dirname(__file__) + "/*.txt"))

df = []
for f in filenames:
    with open(f,"r") as the_file:
        f_data = the_file.read().strip()
    df.append({"document_id":os.path.basename(f).rstrip(".txt"), "text":f_data})

with open(os.path.dirname(__file__) + "/synthetic.json", "w") as the_file:
    _ = json.dump(df, the_file, indent=1)
