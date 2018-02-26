
# encoding=utf8

from time import time
import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
print "Importing"

t0 = time()


test=pd.read_csv("C:/Users/Malte/OneDrive/Coding/My repositories/Mercari/test.tsv",sep="\t")

train=pd.read_csv("C:/Users/Malte/OneDrive/Coding/My repositories/Mercari/train.tsv",sep="\t")

train=train[:1000]
test=test[:1000]

print "import time:", round(time()-t0, 3), "s"

######## FEATURE ENGINEERING ###################

print "Feature Engineering"

t0 = time()


train.fillna("",inplace=True)


def cat_split(row):
    try:
        txt1, txt2, txt3 = row.split('/')
        return row.split('/')
    except:
        return ("No Label", "No Label", "No Label")

train["cat_1"], train["cat_2"], train["cat_3"] = zip(*train["category_name"].apply(lambda val: cat_split(val)))

train["desc_length"]=train["item_description"].apply(lambda x: len(x))


train["has_description"]=train["item_description"].apply(lambda x: described(x))

tfidf=TfidfVectorizer(strip_accents="ascii",min_df=5,lowercase=True,token_pattern=r'\w+',analyzer="word",ngram_range=(1,3),stop_words="english")

tfidf.fit_transform(train["item_description"].apply(str))

tfidf_dict=dict(zip(tfidf.get_feature_names(),tfidf.idf_))



train["brand_name"]=train["brand_name"].astype('category')
train["brand_name_cat"]=train["brand_name"].cat.codes



def compute_tfidf(desc):


    tfidf_score=0
    word_count=0

    for w in desc.lower().split():
        if w in tfidf_dict:
            tfidf_score+=tfidf_dict[w]
        word_count+=1

    if word_count>0:
        return tfidf_score
    else:
        return 0

train["tfidf"]=train["item_description"].apply(lambda x: compute_tfidf(x))

test.fillna("",inplace=True)



test["cat_1"], test["cat_2"], test["cat_3"] = zip(*test["category_name"].apply(lambda val: cat_split(val)))

test["desc_length"]=test["item_description"].apply(lambda x: len(x))



test["has_description"]=test["item_description"].apply(lambda x: described(x))

tfidf=TfidfVectorizer(strip_accents="ascii",min_df=5,lowercase=True,token_pattern=r'\w+',analyzer="word",ngram_range=(1,3),stop_words="english")

tfidf.fit_transform(test["item_description"].apply(str))

tfidf_dict=dict(zip(tfidf.get_feature_names(),tfidf.idf_))



test["tfidf"]=test["item_description"].apply(lambda x: compute_tfidf(x))

print "feature engineering time:", round(time()-t0, 3), "s"


################################################

print "Drawing"

t0=time()

#sns.lmplot("price","tfidf",data=train,fit_reg=False,aspect=4)



# plt.figure(figsize=(20,15))
# plt.hist(train["price"].loc[train["has_description"]==True],label="Has Description",bins=60,color="blue",alpha=0.6,range=[0,250])
# plt.hist(train["price"].loc[train["has_description"]==False],label="Does not have Description",bins=60,alpha=0.6,range=[0,250])
# plt.legend()
# plt.show


# fig,(ax) = plt.subplots(1,2,figsize=(10,5))
# sns.distplot(train["price"],ax=ax[0])
# sns.distplot(np.log1p(train["price"]),ax=ax[1])


# fig,ax=plt.subplots(2,1,figsize=(30,5))
# sns.barplot(x="cat_1",y="price",data=train.loc[train["shipping"]==1],ax=ax[0])
# sns.barplot(x="cat_1",y="price",data=train.loc[train["shipping"]==0],ax=ax[1])

# p = sns.FacetGrid(train, col="shipping",col_order=[1,0],size=5,aspect=3.5)
# p = p.map(sns.barplot,"cat_1","price",ci=None)



# most_freq_items=train["brand_name"].value_counts()
# most_freq_items=list(most_freq_items.index[:20])
# most_freq_items.remove("")
# train_top_10=train[train["brand_name"].isin(most_freq_items)]
# train_top_10=train_top_10.groupby("brand_name")
# train_top_10_by_price= train_top_10["price"].sum().reset_index()
# train_top_10_by_price.sort_values("price",ascending=False,inplace=True)
# fig,ax=plt.subplots(figsize=(20,5))
# sns.barplot(x="brand_name",y="price",data=train_top_10_by_price,ax=ax,ci=None)

def top_x_by_total_price(df,cat,x):
    most_freq_items=df[cat].value_counts()
    most_freq_items=list(most_freq_items.index[:x])
    if "" in most_freq_items:
        most_freq_items.remove("")
    df_top_10=df[df[cat].isin(most_freq_items)]
    df_top_10=df_top_10.groupby(cat)
    df_top_10_by_price= df_top_10["price"].sum().reset_index()
    df_top_10_by_price.sort_values("price",ascending=False,inplace=True)
    fig,ax=plt.subplots(figsize=(x,5))
    ax.set_title("Top {} {} by occurendce, sorted by total article value".format(str(x),cat))
    sns.barplot(x=cat,y="price",data=df_top_10_by_price,ax=ax,ci=None)

#top_x_by_total_price(train,"brand_name",10)

def top_x_by_mean_price(df,cat,x):
    most_freq_items=df[cat].value_counts()
    print most_freq_items
    most_freq_items=list(most_freq_items.index[:x])
    if "" in most_freq_items:
        most_freq_items.remove("")
    df_top_10=df[df[cat].isin(most_freq_items)]
    df_top_10=df_top_10.groupby(cat)
    df_top_10_by_price= df_top_10["price"].mean().reset_index()
    df_top_10_by_price.sort_values("price",ascending=False,inplace=True)
    fig,ax=plt.subplots(figsize=(x,5))
    ax.set_title("Top {} {} by occurendce, sorted by mean article value".format(str(x),cat))
    sns.barplot(x=cat,y="price",data=df_top_10_by_price,ax=ax,ci=None)




# cloud = WordCloud(width=1440, height=1080).generate(" ".join(train['item_description'].astype(str)))
# plt.figure(figsize=(20, 15))
# plt.imshow(cloud)
# plt.axis('off')



#top_x_by_mean_price(train,"cat_2",10)


#sns.barplot(x="cat_2",y="price",data=df_top_10_by_price,ax=ax,ci=None)


#
# sns.lmplot("price","desc_length",data=train,fit_reg=True,scatter_kws={'alpha':0.3},aspect=5,size=5)
#
#
# df = train.groupby(['cat_2'])['price'].agg(['size','sum'])
# print df
# df['mean_price']=df['sum']/df['size']
# df.sort_values(by=['mean_price'], ascending=False, inplace=True)
# df = df[:10]
# df.sort_values(by=['mean_price'], ascending=True, inplace=True)
#
# sns.barplot(data=df,x="mean_price",y=df.index)

print "print time:", round(time()-t0, 3), "s"

######################################################


print "Starting Machine Learning"

t0 = time()

train=train[["tfidf","item_condition_id","shipping","desc_length","price"]]
test=test[["tfidf","item_condition_id","shipping","desc_length"]]

train_price=train["price"].astype(int)
train.drop(["price"],axis=1,inplace=True)

train=preprocessing.scale(train)
test=preprocessing.scale(test)

clf=RandomForestRegressor()
clf.fit(train.astype(float),train_price)
pred=clf.predict(test.astype(float))

print "training time:", round(time()-t0, 3), "s"
print "DONE"


pd.DataFrame(pred).to_csv("C:/Users/Malte/OneDrive/Coding/My repositories/Mercari/submission.csv", encoding='utf-8')
