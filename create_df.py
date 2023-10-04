import os
import math

import pandas as pd


# Categorize age into age_groups
def categorize_age_group(age):
    age_group = math.floor(age / 10)
    age_group = age_group if age_group < 10 else 10
    return age_group


data_path = "./data/UTKFace"
imgs = os.listdir(data_path)

datas = []
for img in imgs:
    age = int(img.split("_")[0])
    img = os.path.join(data_path, img)
    datas.append({
        "path": img,
        "age": age
    })

df = pd.DataFrame(datas)
df['age'] = df['age'].apply(categorize_age_group)
df.to_csv("./data/data.csv", index=False)
