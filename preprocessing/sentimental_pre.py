# %%
import pandas as pd
import numpy as np

# %%
senti_df = pd.read_excel("../data/sentimental_data/Training/감성대화말뭉치(최종데이터)_Training.xlsx")

# %%
senti_df2 = pd.read_excel("../data/sentimental_data/Validation/감성대화말뭉치(최종데이터)_Validation.xlsx")

# %%
senti_df.head(20)

# %%
senti_df.shape

# %%
senti_df["사람문장2"].isna().sum()

# %% [markdown]
# ## 감정_대분류 기준으로 label 분류

# %% [markdown]
# ### 중복 제거(같은 감정이지만 공백으로 인해 다르게 인식되는 항목들 병합)

# %%
set(senti_df["감정_대분류"])

# %%
set(senti_df2["감정_대분류"])

# %%
# 띄어쓰기 되어있는 label변경
senti_df = senti_df.replace({'감정_대분류' : '기쁨 '}, '기쁨')
senti_df = senti_df.replace({'감정_대분류' : '불안 '}, '불안')

# %%
set(senti_df["감정_대분류"])

# %%
def unzip_sentence(df):
    unzipped_x = []
    unzipped_y = []
    for i in range(len(df)):
        # 사람문장 1
        unzipped_x.append(df.iloc[i, 7])
        unzipped_y.append(df.iloc[i, 5])
        # 사람문장 2
        unzipped_x.append(df.iloc[i, 9])
        unzipped_y.append(df.iloc[i, 5])

    unzipped_dict = {"sentences" : unzipped_x, "sentiment" : unzipped_y}
    unzipped_df = pd.DataFrame(unzipped_dict)

    return unzipped_df

# %%
senti1_unzipped = unzip_sentence(senti_df)

# %%
senti2_unzipped = unzip_sentence(senti_df2)

# %%
senti1_unzipped.head(10)

# %%
senti2_unzipped.head(10)

# %%
senti_unzipped = pd.concat([senti1_unzipped, senti2_unzipped], ignore_index=True)

# %%
senti_unzipped.shape

# %%
senti_unzipped.to_csv("../data/sentimental.tsv", sep = "\t",index = False, header=None)


