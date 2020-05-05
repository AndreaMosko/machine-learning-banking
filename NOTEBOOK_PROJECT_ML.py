#!/usr/bin/env python
# coding: utf-8

# # MACHINE LEARNING PROJECT
# MEZOUAR ADNANE / 
# MIJOT ALICE / 
# MOSKOVLJEVIC ANDREA
# 
# 
# #### Purpose:
# The purpose of this case study is to give you a very small flavor of something you may
# encounter in the banking sector, and to see how you handle a practical analysis and how you
# communicate results and analytical approaches.
# 
# #### Expected result:
# - What are the steps to take for this project?
# - Which clients are to be targeted with which offer?
# - What would be the expected revenue based on your strategy?
# 
# ## SUMMARY  
#   1. DataSets Preparation 
#   2. DataFrame : Exploration of the 'positive' clients
#   
# #### --- SALES OBJECTIVE
#   3. Models explorations for each of the sales
#   4. Exploration/Optimisation of two most promising models for each of the sales
#   5. Deployment of the best model on the 40% clients (forecast objective) dataset
#   
# #### --- REVENUE OBJECTIVE
#   6. Models explorations for each of the revenues
#   7. Exploration/Optimisation of two most promising models for each of the revenues
#   8. Deployment of the best model on the 40% clients (forecast objective) dataset
# 
# #### --- FINAL
#   9. Creation of the list of clients to target

# In[1150]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Read the xls to dataframe by tabs
excel_file = pd.ExcelFile("/Users/Alice/Documents/EML_COURS/MACHINE LEARNING/Assignments/3. Project/DMML_Project_Dataset.xlsx")
#df1  100%
df_Soc_Dem = pd.read_excel(excel_file,'Soc_Dem')
#df2  100%
df_ActBal= pd.read_excel(excel_file,'Products_ActBalance')
#df3  100%
df_Inf_Out = pd.read_excel(excel_file, 'Inflow_Outflow')
#df4  60%
df_SalesRev = pd.read_excel(excel_file, 'Sales_Revenues')


# # 1. DataSets Preparation : Merge + Sets creation

# In[1151]:


#merging datasets

#### FIRST DATA SET WITH ALL DATA COMPLETED : 1615 observations with 36 variables
df_1a = pd.merge(df_Soc_Dem, df_ActBal, on='Client', how='inner')
df_1b = pd.merge(df_1a, df_Inf_Out, on='Client', how='left') #has to be a left otherwise we loose 28 observations
df_2_100 = pd.merge(df_1b,df_SalesRev, on='Client', how='left') #base 100%

#### Definition base 60% withh all data : 969 observations with 36 variables
df_2_60 = pd.merge(df_1b,df_SalesRev, on='Client', how='inner') #base 60%
df2 = df_2_60

#### Definition base 40% with the model to later deploy it on
# 40% of the 1615 clients in the data and using for prediction
df_2_40 = pd.concat([df_2_100,df2],ignore_index=True)
df_2_40.drop_duplicates(subset='Client',keep=False,inplace=True)

df_2_100.head(5), 
df_2_60.head(5),
df_2_100.shape,df_2_60.shape, df_2_40.shape
#df_2_100.shape == df_2_60.shape[0] df_2_40.shape[0]

#df.to_pickle('df_cities_countries.dat')


# #### 60% CLIENTS DATASET - STRUCTURE AND CLEANING

# In[1152]:


df2.info()


# In[1153]:


import missingno as msno
msno.bar(df2)
#Barplot showing miss values


# ##### From categorical to numerical

# In[1154]:


#from categorical to numerical
#df2['Sex']=df2.replace(np.NaN,1)
df2['Sex']=df2.Sex.map({'F':1,'M':2}).replace(np.NaN,1)
df2['Sex']


# In[1155]:


#df2.loc(df2['Sex']==np.NaN,'Client').


# ##### Filling missing values

# In[1156]:


##Check missing variables
#print((df2==np.NaN).sum())


# In[1157]:


df2[['Count_SA','Count_CA','Count_MF','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL']] = df2[['Count_SA','Count_CA','Count_MF','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL']].replace(np.NaN,0)
print((df2[['Count_SA','Count_CA','Count_MF','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL']]).count())
df2[['Count_SA','Count_CA','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL','Tenure','Age','Client','Sale_MF','Sale_CL','Sale_CC']] = df2[['Count_SA','Count_CA','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL','Tenure','Age','Client','Sale_MF','Sale_CL','Sale_CC']].astype(np.float64)

#Dropping the rows which have one missing value
df2 = df2.dropna()#how='all',thresh=1)
print('\nShape of the dataset once the missing values have been dropped :',df2.shape)


# In[1158]:


df2.info()


# In[1159]:


import missingno as msno
msno.bar(df2)
#Barplot showing miss values


# #### 40% CLIENTS DATASET - STRUCTURE AND CLEANING

# In[1160]:


df_2_40.info()


# In[1161]:


#from categorical to numerical
df_2_40['Sex']=df_2_40.Sex.map({'F':1,'M':2}).replace(np.NaN,1)
df_2_40[['Count_SA','Count_CA','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL']] = df_2_40[['Count_SA','Count_CA','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL']].replace(np.NaN,0)
df_2_40[['Count_SA','Count_CA','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL','Tenure','Age']] = df_2_40[['Count_SA','Count_CA','Count_MF','Count_OVD','Count_CC','Count_CL','ActBal_SA','ActBal_MF','ActBal_OVD','ActBal_CC','ActBal_CL','Tenure','Age']].astype(np.float64)

# Replace it by Average
df_2_40[['VolumeCred']]=df_2_40[['VolumeCred']].replace(np.NaN,df_2_40['VolumeCred'].mean())
df_2_40[['VolumeCred_CA']]=df_2_40[['VolumeCred_CA']].replace(np.NaN,df_2_40['VolumeCred_CA'].mean())
df_2_40[['TransactionsCred']]=df_2_40[['TransactionsCred']].replace(np.NaN,df_2_40['TransactionsCred'].mean())
df_2_40[['TransactionsCred_CA']]=df_2_40[['TransactionsCred_CA']].replace(np.NaN,df_2_40['TransactionsCred_CA'].mean())
df_2_40[['VolumeDeb']]=df_2_40[['VolumeDeb']].replace(np.NaN,df_2_40['VolumeDeb'].mean())
df_2_40[['VolumeDeb_CA']]=df_2_40[['VolumeDeb_CA']].replace(np.NaN,df_2_40['VolumeDeb_CA'].mean())
df_2_40[['VolumeDebCash_Card']]=df_2_40[['VolumeDebCash_Card']].replace(np.NaN,df_2_40['VolumeDebCash_Card'].mean())
df_2_40[['VolumeDebCashless_Card']]=df_2_40[['VolumeDebCashless_Card']].replace(np.NaN,df_2_40['VolumeDebCashless_Card'].mean())
df_2_40[['VolumeDeb_PaymentOrder']]=df_2_40[['VolumeDeb_PaymentOrder']].replace(np.NaN,df_2_40['VolumeDeb_PaymentOrder'].mean())
df_2_40[['TransactionsDeb']]=df_2_40[['TransactionsDeb']].replace(np.NaN,df_2_40['TransactionsDeb'].mean())
df_2_40[['TransactionsCred']]=df_2_40[['TransactionsCred']].replace(np.NaN,df_2_40['TransactionsCred'].mean())
df_2_40[['TransactionsCred_CA']]=df_2_40[['TransactionsCred_CA']].replace(np.NaN,df_2_40['TransactionsCred_CA'].mean()) 
df_2_40[['TransactionsDeb_CA']]=df_2_40[['TransactionsDeb_CA']].replace(np.NaN,df_2_40['TransactionsDeb_CA'].mean())
df_2_40[['TransactionsDebCash_Card']]=df_2_40[['TransactionsDebCash_Card']].replace(np.NaN,df_2_40['TransactionsDebCash_Card'].mean())
df_2_40[['TransactionsDebCashless_Card']]=df_2_40[['TransactionsDebCashless_Card']].replace(np.NaN,df_2_40['TransactionsDebCashless_Card'].mean())
df_2_40[['TransactionsDeb_PaymentOrder']]=df_2_40[['TransactionsDeb_PaymentOrder']].replace(np.NaN,df_2_40['TransactionsDeb_PaymentOrder'].mean())

#df_2_40 = df_2_40.replace(np.NaN,0)
#df_2_40_2 =  df_2_40.drop(['Sale_MF', 'Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL'], axis=1)
#msno.bar(df_2_40_2)

df_2_40_3 = df_2_40.replace([np.inf, -np.inf], np.NaN, inplace=False)
#np.any(np.isnan(mat))
df_2_40_3 = df_2_40_3.replace(np.NaN,0)
#df_2_40_3 = df_2_40_3.fillna(0)
#np.all(np.isfinite(mat))


# In[1162]:


#Barplot showing miss values
import missingno as msno
msno.bar(df_2_40_3)


# ### DataFrame : Exploration of the whole dataset

# In[704]:


## Create the plot 
df_Soc_Dem.groupby('Sex').size().plot(kind='bar', color = "Orange")
plt.xlabel('Sex')
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.title('Number of Female and Male clients in the dataset')
## Number of males and females
nb_female = df_Soc_Dem.groupby('Sex').apply(lambda x: x[x=='F'].count())
nb_male = df_Soc_Dem.groupby('Sex').apply(lambda x: x[x=='M'].count())

## Addind the value labels
plt.text(-0.08, 700, '756')
plt.text(0.92, 800,'856');
## Isolating dataframes of both genders
female = df_Soc_Dem[df_Soc_Dem['Sex'] == 'F']
male = df_Soc_Dem[df_Soc_Dem['Sex'] == 'M']


# In[705]:


df_Soc_Dem.groupby('Sex').count()
#df_Soc_Dem[['Sex']].describe()
#df_Soc_Dem[['Age','Tenure']].describe()


# In[706]:


#Data Visualization
#Age histogram
bins = [0,10,20,30,40,50,60,70,80,90,100]
groups = df_Soc_Dem.groupby(pd.cut(df_Soc_Dem['Age'], bins)).count()
groups['Age'].plot.bar(width=0.6, color='Orange')
plt.xlabel('Age')
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title('Distribution of Age M & F')
nb_female = df_Soc_Dem.groupby('Age').apply(lambda x: x[x=='F'].count())
nb_male = df_Soc_Dem.groupby('Age').apply(lambda x: x[x=='M'].count())


# In[707]:


#Age histogram
df_Soc_Dem_fem = df_Soc_Dem[df_Soc_Dem['Sex'] == 'F']
bins = [0,10,20,30,40,50,60,70,80,90,100]
groups = df_Soc_Dem_fem.groupby(pd.cut(df_Soc_Dem_fem["Age"], bins)).count()
groups["Age"].plot.bar(width=0.6, color='Orange')
plt.xlabel("Age")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title('Female Age Distribution');


# In[708]:


#Age histogram
df_Soc_Dem_male = df_Soc_Dem[df_Soc_Dem['Sex'] == 'M']
bins = [0,10,20,30,40,50,60,70,80,90,100]
groups = df_Soc_Dem_male.groupby(pd.cut(df_Soc_Dem_male["Age"], bins)).count()
groups["Age"].plot.bar(width=0.6, color='Orange')
plt.xlabel("Age")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title('Male Age Distribution');


# Is this necessary?

# In[709]:


height = df_Soc_Dem.groupby('Sex').size()
bins=['F','M']
width =0.5
x = np.arange(len(bins)) # the locatuibs fir the groups
# plt.bar(height=C_Sex)
plt.bar(x, height,color = 'orange')
plt.xticks(x,bins)
plt.text(-0.08,650,756,{'color':'w','fontsize':20})
plt.text(0.92,750,856,{'color':'w','fontsize':20})
plt.title('Distribution of customer in gender')
plt.show()


# In[710]:


#Data Visualization
# bins = [0,30,60,90,120,150,180,210,240,270,300]
genders=['M','F']
colors = ['orange','blue']
# groups = df1.groupby(pd.cut(df1['Tenure'], bins)).count()
# groups['Tenure'].plot.bar(width=0.6, color='Blue')
plt.hist([df_Soc_Dem.loc[df_Soc_Dem['Sex'] == x, 'Tenure'] for x in genders], label=genders, color=colors)
plt.xlabel('Tenure')
plt.ylabel("Months")
plt.legend(['M','F'])
#plt.xticks(rotation=45)
plt.title('Distribution of Tenure by gender');


# # 2. DataFrame : Exploration of the 'positive' clients

# ### Analysis 1 : Targeting Frequency results

# In[711]:


####Analysis 1: Targeting frequency results

#Total number of offers (target variable sale) per customer (SUM)
df2['Sale_Total'] = df2['Sale_MF']+df2['Sale_CC']+df2['Sale_CL']

#Total nomber of clients in each category
nb_0_resp = df2.loc[df2['Sale_Total'] == 0,'Client'].count()
nb_1_resp = df2.loc[df2['Sale_Total'] == 1,'Client'].count()
nb_2_resp = df2.loc[df2['Sale_Total'] == 2,'Client'].count()
nb_3_resp = df2.loc[df2['Sale_Total'] == 3,'Client'].count()
NB_resp = np.array([nb_0_resp,nb_1_resp,nb_2_resp,nb_3_resp])

#checking matching
df2['Client'].count() == (nb_0_resp + nb_1_resp + nb_2_resp + nb_3_resp)

Freq_0_resp = round((nb_0_resp/(df2['Client'].count()))*100,1)
Freq_1_resp = round((nb_1_resp/(df2['Client'].count()))*100,1)
Freq_2_resp = round((nb_2_resp/(df2['Client'].count()))*100,1)
Freq_3_resp = round((nb_3_resp/(df2['Client'].count()))*100,1)
Frequency_resp = np.array([Freq_0_resp,Freq_1_resp,Freq_2_resp,Freq_3_resp])

#checking matching : error evaluated at 0.00000000000001
round((Freq_0_resp + Freq_1_resp + Freq_2_resp + Freq_3_resp),1) == 100

#creation of the dataframe
columns = ["#Client","%Partition"]
data = {'#Client': NB_resp, '%Partition': Frequency_resp}
rows = ["0 response", "1 response", "2 responses","3 responses"]
df_freq = pd.DataFrame(data=data, index=rows, columns=columns)
df_freq


# ### Analysis 2 : Targeting Value Results

# In[712]:


####Analysis 2: Targeting value results

#Total number of offers (target variable sale) per customer (SUM)
df2['Revenue_Total'] = df2['Revenue_MF']+df2['Revenue_CC']+df2['Revenue_CL']

#Total revenues for every type of clients in each category
nb_0_rev = round(df2.loc[df2['Sale_Total'] == 0,'Revenue_Total'].sum(),1)
nb_1_rev = round(df2.loc[df2['Sale_Total'] == 1,'Revenue_Total'].sum(),1)
nb_2_rev = round(df2.loc[df2['Sale_Total'] == 2,'Revenue_Total'].sum(),1)
nb_3_rev = round(df2.loc[df2['Sale_Total'] == 3,'Revenue_Total'].sum(),1)
NB_rev_ = np.array([nb_0_rev,nb_1_rev,nb_2_rev,nb_3_rev])

#check the matching
round(df2['Revenue_Total'].sum(),1) == round((nb_0_rev + nb_1_rev + nb_2_rev + nb_3_rev),1)

#Mean revenues for every type of clients in each category
mean_0_rev = round(df2.loc[df2['Sale_Total'] == 0,'Revenue_Total'].mean(),1)
mean_1_rev = round(df2.loc[df2['Sale_Total'] == 1,'Revenue_Total'].mean(),1)
mean_2_rev = round(df2.loc[df2['Sale_Total'] == 2,'Revenue_Total'].mean(),1)
mean_3_rev = round(df2.loc[df2['Sale_Total'] == 3,'Revenue_Total'].mean(),1)
MEAN_rev_ = np.array([mean_0_rev,mean_1_rev,mean_2_rev,mean_3_rev])


#Mean CL revenues for CL sales
cl_mean_0_rev = round(df2.loc[df2['Sale_CL'] == 0,'Revenue_CL'].mean(),1)
cl_mean_1_rev = round(df2.loc[df2['Sale_CL'] == 1,'Revenue_CL'].mean(),1)
cl_mean_2_rev = round(df2.loc[df2['Sale_CL'] == 2,'Revenue_CL'].mean(),1)
cl_mean_3_rev = round(df2.loc[df2['Sale_CL'] == 3,'Revenue_CL'].mean(),1)
cl_MEAN_rev_ = np.array([cl_mean_0_rev,cl_mean_1_rev,cl_mean_2_rev,cl_mean_3_rev])

#Mean MF revenues for MF sales
mf_mean_0_rev = round(df2.loc[df2['Sale_MF'] == 0,'Revenue_MF'].mean(),1)
mf_mean_1_rev = round(df2.loc[df2['Sale_MF'] == 1,'Revenue_MF'].mean(),1)
mf_mean_2_rev = round(df2.loc[df2['Sale_MF'] == 2,'Revenue_MF'].mean(),1)
mf_mean_3_rev = round(df2.loc[df2['Sale_MF'] == 3,'Revenue_MF'].mean(),1)
mf_MEAN_rev_ = np.array([mf_mean_0_rev,mf_mean_1_rev,mf_mean_2_rev,mf_mean_3_rev])

#Mean CC revenues for CC sales
cc_mean_0_rev = round(df2.loc[df2['Sale_CC'] == 0,'Revenue_CC'].mean(),1)
cc_mean_1_rev = round(df2.loc[df2['Sale_CC'] == 1,'Revenue_CC'].mean(),1)
cc_mean_2_rev = round(df2.loc[df2['Sale_CC'] == 2,'Revenue_CC'].mean(),1)
cc_mean_3_rev = round(df2.loc[df2['Sale_CC'] == 3,'Revenue_CC'].mean(),1)
cc_MEAN_rev_ = np.array([cc_mean_0_rev,cc_mean_1_rev,cc_mean_2_rev,cc_mean_3_rev])

#Repartition revenues
Freq_0_rev = round((nb_0_rev/(df2['Revenue_Total'].sum()))*100,1)
Freq_1_rev = round((nb_1_rev/(df2['Revenue_Total'].sum()))*100,1)
Freq_2_rev = round((nb_2_rev/(df2['Revenue_Total'].sum()))*100,1)
Freq_3_rev = round((nb_3_rev/(df2['Revenue_Total'].sum()))*100,1)
Revenues_freq = np.array([Freq_0_rev,Freq_1_rev,Freq_2_rev,Freq_3_rev])

#creation of the dataframe
columns = ["#Client","%Clients", "%Revenues", "TT Sum Revenues",'TT Mean Revenues','CL Mean Revenue','MF Mean Revenue','CC Mean Revenue']
data = {'#Client': NB_resp,'%Clients': Frequency_resp , "%Revenues" : Revenues_freq , 'TT Sum Revenues': NB_rev_, 'TT Mean Revenues':MEAN_rev_,'CL Mean Revenue':cl_MEAN_rev_,'MF Mean Revenue':mf_MEAN_rev_,'CC Mean Revenue':cc_MEAN_rev_}
rows = ["0 response", "1 response", "2 responses","3 responses"]
df_value = pd.DataFrame(data=data, index=rows, columns=columns)
df_value


# In[713]:


Revenues = df2[['Revenue_MF','Revenue_CC','Revenue_CL']].iloc[:,-3:].copy()
Revenue_agg = Revenues.agg(['sum',"mean","std"])
Revenue_agg


# In[714]:


# Boxplot for valueFreq
sns.boxplot(x=df_value["#Client"]);


# In[715]:


sns.boxplot(x=df_value["TT Sum Revenues"]);


# In[716]:


df_value.boxplot(column=["%Clients", "%Revenues"]);


# In[717]:


df_value_freq_1 = df_value[["#Client","TT Sum Revenues"]]
ax = df_value_freq_1.plot.hist(bins=12, alpha=0.5,color = ['orange','blue']);


# ### Explore correlations : Focus on the whole dataset

# In[718]:


df2_dem_sales = df2[['Sex','Age','Tenure','Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL']]

df3_dem_sales = df2[['Sex','Age','Tenure','Sale_Total','Revenue_Total']]


# In[719]:


# Plot correlation matrix
def plot_corr(df3_dem_sales,size=10):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df3_dem_sales.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(size, size-2))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
plot_corr(df3_dem_sales,size=4)


# In[720]:


# Plot correlation matrix
def plot_corr(df2_dem_sales,size=10):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df2_dem_sales.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(size, size-2))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
plot_corr(df2_dem_sales[1:],size=5)

#### RESULT INSIGHTS : High correlation btw
# TENURE : Sale_CL, Revenue_CL
# AGE : Sale_CC, Revenue_CC, Revenue_MF  (not Sale_MF nor CL)
# obiously : Sale_MF <> Revenue_MF   /  Sale_CC <> Revenue_CC   / Sale_CL <> Revenue_CL 


# In[721]:


#variables
CC_ = df2.loc[:, df2.columns.str.endswith("CC")]
CL_ = df2.loc[:, df2.columns.str.endswith("CL")]
MF_ = df2.loc[:, df2.columns.str.endswith("MF")]
CA_ = df2.loc[:, df2.columns.str.endswith("CA")]
SA_ = df2.loc[:, df2.columns.str.endswith("SA")]
OVD_ = df2.loc[:, df2.columns.str.endswith("OVD")]
df_variables = pd.concat([CC_,CL_,MF_,CA_,SA_,OVD_],sort=False)


# In[722]:


# Plot correlation matrix
def plot_corr(df_variables,size=10):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df_variables.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(size, size-2))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
plot_corr(df_variables,size=12)

#   >>>> HIGH CORRELATION WITH THE SALE / REVENUE OF THE SAME TYPE : obvious


# In[723]:


corrmat = df2.corr() 
cmap = sns.diverging_palette(220, 10, as_cmap=True)  
f, ax = plt.subplots(figsize =(15, 10)) 
sns.heatmap(corrmat, ax = ax, cmap =cmap, linewidths = 0.1) 


# In[724]:


corrmat2 = df2.corr() 

#cmap = sns.diverging_palette(220, 10, as_cmap=True)  
import seaborn as seaborn
cmap = seaborn.diverging_palette(210, 350, as_cmap=True)

cg = sns.clustermap(corrmat2, cmap =cmap, linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
  
cg


# ## Explore correlations : Focus on studying 'positive' clients

# #### DataFrame creation with only positive responding clients

# In[725]:


#Summarizing the responses to the 3 offers : Data from 0 to 3
df2['Sale_Total'] = df2['Sale_MF']+df2['Sale_CC']+df2['Sale_CL'] 

#Summarizing the responses in a binary way : Responds or not
#1 standing for YES already responded at least one time and 0 standing for NO
df2['Responding_Client'] = np.where(df2['Sale_Total'] != 0, '1', '0') 
#Create a dataframe with only responding clients : df_yes
df_yes = df2.loc[df2['Responding_Client'].isin(['1'])]
#Create a dataframe with only responding clients : df_no
df_no = df2.loc[df2['Responding_Client'].isin(['0'])]

#checking mapping
len(df2) == len(df_yes) + len(df_no)


# #### HEATMAP : Clients who responded at least once to a sale offer

# In[726]:


## CLIENTS WHO RESPONDED AT LEAST ONCE TO SALE

def plot_corr(df_yes,size=10):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df_yes.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(size, size-2))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
plot_corr(df_yes,size=12)


# In[727]:


def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), 
        y=y.map(y_to_num), 
        s=size * size_scale,
        marker='s') 
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    
columns = ['Tenure', 'Sale_MF', 'Sale_CC', 'Sale_CL', 'Sale_Total', 'Revenue_MF','Revenue_CC','Revenue_CL','Sex'] 
corr = df_yes[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index')
corr.columns = ['x', 'y', 'value']

heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs())


# ##### Total Sales Correlation in general

# In[728]:


# Total Sales Correlation in general
corrmat = df_yes.corr() 
corrmat.sort_values(['Sale_Total'], ascending = False, inplace = True)
print(corrmat.Sale_Total.head(15))


# In[729]:


from pandas.plotting import scatter_matrix
attributes = ["Sale_Total","Revenue_Total","TransactionsCred","Tenure","TransactionsCred_CA","Count_MF","TransactionsDeb","TransactionsDebCash_Card"]
scatter_matrix(df_yes[attributes], figsize=(10, 10), alpha=0.2, diagonal='kde');


# In[730]:


#Correlation of the positive respondants : PRODUCT CL
corrmat = df2.corr() 

corrmat.sort_values(['Revenue_CL'], ascending = False, inplace = True)
corr_cl_list = corrmat.Revenue_CL.head(15).sort_values(ascending = False)
corr_cl_list


# In[731]:


from pandas.plotting import scatter_matrix
attributes = ["Revenue_CL","Tenure", "TransactionsCred_CA","VolumeDeb","TransactionsCred","TransactionsDebCash_Card"]
scatter_matrix(df2[attributes], figsize=(10, 10));


# In[ ]:





# ##### PRODUCT CC

# In[732]:


#Correlation of the positive respondants : PRODUCT CC
corrmat = df2.corr() 
corrmat.sort_values(['Revenue_CC'], ascending = False, inplace = True)
corr_cc_list = corrmat.Revenue_CC.head(15).sort_values(ascending = False)
corr_cc_list


# In[733]:


from pandas.plotting import scatter_matrix
attributes = ["Revenue_CC","Sale_CC", "ActBal_CC","Count_CL","Sale_Total","ActBal_CL"]
scatter_matrix(df2[attributes], figsize=(10, 10));


# ##### PRODUCT MF

# In[734]:


#Correlation of the positive respondants : PRODUCT MF
corrmat = df_yes.corr() 

corrmat.sort_values(['Revenue_MF'], ascending = False , inplace = True)
corr_mf_list = corrmat.Revenue_MF.head(15).sort_values(ascending = False)
corr_mf_list


# In[735]:


from pandas.plotting import scatter_matrix
attributes = ["Revenue_MF","Sale_MF", "Count_SA","Sale_Total","Count_MF",'Age']
scatter_matrix(df_yes[attributes], figsize=(10, 10));


# #### NEW METRIC : PROFITABILITY

# In[736]:


#df_yes['%Profitability'] = round((df_yes['Revenue_Total']/df_yes['Tenure'])*100,2)
#print(df_yes['%Profitability'])
#df_yes.plot(x="Sale_Total", y="Revenue_Total", kind="scatter", label="% Profitability");


# In[737]:


from scipy import stats
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.90)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[738]:


#df_out = remove_outlier(df_yes,'%Profitability')
#df_yes['%Profitability_bis'] = df_out["%Profitability"].replace(np.NaN,0)

#ax = df_yes.plot(x="Sale_Total", y="%Profitability_bis", kind="scatter", label="% Profitability")
##df_yes.sort_values("Revenue_Total").plot(x="Sale_Total", y="Revenue_Total", ax=ax, color="green", label="Revenue_Total")


# In[739]:


### df_yes['Profitability'] = df_yes['%Profitability']
#corrmat #= df_yes.corr() 
#corrmat.sort_values(['Profitability'], ascending = False , inplace = True)
#corr_mf_list = corrmat.Profitability.head(15).sort_values(ascending = False)
#corr_mf_list


# # ------------ SALES

# # 3. MODELS EXPLORATION : Which to choose for each Sale objective

# In[740]:


df2 = df2.drop(['Responding_Client','Sale_Total','Revenue_Total'], axis=1)
print(df2.head(0))


# ### FOCUS 1 : SALE_MF 

# #### 1. CREATE THE TRAINING SET FOR THE SALE_MF TARGET

# In[741]:


MF_ = df2.loc[:, df2.columns.str.endswith("MF")]
#print(MF_.head(0))


# In[742]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_mf = df2[['Sale_MF']]


# In[743]:


import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_mf, x_test_mf, y_train_mf,y_test_mf = train_test_split(x,y_mf,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_mf.shape)
print('Training Labels Shape:', y_train_mf.shape)
print('Testing Features Shape:', x_test_mf.shape)
print('Testing Labels Shape:', y_test_mf.shape)


# #### 2. EVALUATE ACCURACY OF MODELS ON THE SALE_MF TARGET

# ##### DECISION TREE REGRESSOR

# In[744]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
regressor = DecisionTreeRegressor()

#Plug the model to the training set
mf_model_dt = regressor.fit(x_train_mf,y_train_mf)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_dt_mf = cross_val_score(mf_model_dt, x_train_mf, y_train_mf, cv=5, scoring='accuracy')
scval_dt_mf = print("CROSS VALIDATION\nAccuracy Score of the DT model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_dt_mf.mean(), scores_cross_val_dt_mf.std() * 2),'\n')
scval_dt_mf 

#Evaluate the model on the test set
mf_predictions_dt_test = mf_model_dt.predict(x_test_mf)
mf_acc_dt = round(accuracy_score(y_test_mf,mf_predictions_dt_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_dt)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:',round(metrics.mean_absolute_error(y_test_mf, mf_predictions_dt_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_dt_test),3),'\n\n')
#print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_dt_test),3)))

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_dt = confusion_matrix(y_test_mf,mf_predictions_dt_test)
print(cmx_mf_dt,'\n')
TP_mf_dt = cmx_mf_dt[1,1]
TN_mf_dt = cmx_mf_dt[0,0]
FP_mf_dt = cmx_mf_dt[0,1]
FN_mf_dt = cmx_mf_dt[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_dt[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_dt[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_dt[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_dt[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_mf_dt_micro = precision_score(y_test_mf, mf_predictions_dt_test, average='micro')
precision_mf_dt_weighted = precision_score(y_test_mf, mf_predictions_dt_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_dt_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_dt_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_dt_micro = recall_score(y_test_mf, mf_predictions_dt_test, average='micro')
recall_mf_dt_weighted = recall_score(y_test_mf, mf_predictions_dt_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_dt_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_dt_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### K-NEAREST NEIGHBOR MODEL

# In[745]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 4) #4 is the best classifier in this case

#Plug the model to the training set
mf_model_knn = knn.fit(x_train_mf,y_train_mf.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_knn_mf = cross_val_score(mf_model_knn, x_train_mf, y_train_mf, cv=5, scoring='accuracy')
scval_knn_mf = print("CROSS VALIDATION\nAccuracy Score of the KNN model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_knn_mf.mean(), scores_cross_val_knn_mf.std() * 2),'\n')
scval_knn_mf 

#Evaluate the model on the test set
mf_predictions_knn_test = mf_model_knn.predict(x_test_mf)
mf_acc_knn = round(accuracy_score(y_test_mf,mf_predictions_knn_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_knn)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_knn_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_knn_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_knn_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_knn = confusion_matrix(y_test_mf,mf_predictions_knn_test)
print(cmx_mf_knn,'\n')
TP_mf_knn = cmx_mf_knn[1,1]
TN_mf_knn = cmx_mf_knn[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_knn[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_knn[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_knn[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_knn[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_mf_knn_micro = precision_score(y_test_mf, mf_predictions_knn_test, average='micro')
precision_mf_knn_weighted = precision_score(y_test_mf, mf_predictions_knn_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_knn_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_knn_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_knn_micro = recall_score(y_test_mf, mf_predictions_knn_test, average='micro')
recall_mf_knn_weighted = recall_score(y_test_mf, mf_predictions_knn_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_knn_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_knn_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### LOGISTIC REGRESSION

# In[746]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logmodel = LogisticRegression(max_iter=1000) #solver sklearn
import warnings
warnings.filterwarnings('ignore')

#Plug the model to the training set
mf_model_lr = logmodel.fit(x_train_mf,y_train_mf.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_lr_mf = cross_val_score(mf_model_lr, x_train_mf, y_train_mf, cv=5, scoring='accuracy')
scval_lr_mf= print("CROSS VALIDATION\nAccuracy Score of the Logistic Regression model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_lr_mf.mean(), scores_cross_val_lr_mf.std() * 2),'\n')
scval_lr_mf 

#Evaluate the model on the test set
mf_predictions_lr_test = mf_model_lr.predict(x_test_mf)
mf_acc_lr = round(accuracy_score(y_test_mf,mf_predictions_lr_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_lr)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_lr_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_lr_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_lr_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lr = confusion_matrix(y_test_mf,mf_predictions_lr_test)
print(cmx_mf_lr,'\n')
TP_mf_lr = cmx_mf_lr[1,1]
TN_mf_lr = cmx_mf_lr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_lr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_lr[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_mf_lr_micro = precision_score(y_test_mf, mf_predictions_lr_test, average='micro')
precision_mf_lr_weighted = precision_score(y_test_mf, mf_predictions_lr_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_lr_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_lr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_lr_micro = recall_score(y_test_mf, mf_predictions_lr_test, average='micro')
recall_mf_lr_weighted = recall_score(y_test_mf, mf_predictions_lr_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_lr_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_lr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVM

# In[747]:


from sklearn import svm
from sklearn.metrics import accuracy_score
svmmodel = svm.SVC()

#Plug the model to the training set
mf_model_svm = svmmodel.fit(x_train_mf,y_train_mf.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_svm_mf = cross_val_score(mf_model_svm, x_train_mf, y_train_mf, cv=5, scoring='accuracy')
scval_svm_mf= print("CROSS VALIDATION\nAccuracy Score of the SVM model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_svm_mf.mean(), scores_cross_val_svm_mf.std() * 2),'\n')
scval_svm_mf 

#Evaluate the model on the test set
mf_predictions_svm_test = mf_model_svm.predict(x_test_mf)
mf_acc_svm = round(accuracy_score(y_test_mf,mf_predictions_svm_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_svm)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_svm_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_svm_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_svm_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_svm = confusion_matrix(y_test_mf,mf_predictions_svm_test)
print(cmx_mf_svm,'\n')
TP_mf_svm = cmx_mf_svm[1,1]
TN_mf_svm = cmx_mf_svm[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_svm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_svm[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_svm[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_svm[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_mf_svm_micro = precision_score(y_test_mf, mf_predictions_svm_test, average='micro')
precision_mf_svm_weighted = precision_score(y_test_mf, mf_predictions_svm_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_svm_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_svm_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_svm_micro = recall_score(y_test_mf, mf_predictions_svm_test, average='micro')
recall_mf_svm_weighted = recall_score(y_test_mf, mf_predictions_svm_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_svm_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_svm_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### LDA

# In[748]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_mf = sc.fit_transform(x_train_mf)
x_test_mf = sc.transform(x_test_mf)

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_train_mf = lda.fit_transform(x_train_mf, y_train_mf)
x_test_mf = lda.transform(x_test_mf)

# Comparison w/Random Forest
from sklearn.ensemble import RandomForestClassifier
mf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
mf_classifier_lda = mf_classifier.fit(x_train_mf, y_train_mf)#.values.ravel()

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_lda_mf = cross_val_score(mf_classifier_lda, x_train_mf, y_train_mf, cv=5, scoring='accuracy')
scval_lda_mf = print("CROSS VALIDATION\nAccuracy Score of the LDA model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_lda_mf.mean(), scores_cross_val_lda_mf.std() * 2),'\n')
scval_lda_mf 

#Evaluate the model on the test set
mf_predictions_lda_test = mf_classifier_lda.predict(x_test_mf)
mf_acc_lda = round(accuracy_score(y_test_mf,mf_predictions_lda_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_lda)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_lda_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_lda_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_lda_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lda = confusion_matrix(y_test_mf,mf_predictions_lda_test)
print(cmx_mf_lda,'\n')
TP_mf_lda = cmx_mf_lda[1,1]
TN_mf_lda = cmx_mf_lda[0,0]
FP_mf_lda = cmx_mf_lda[0,1]
FN_mf_lda = cmx_mf_lda[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lda[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lda[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_lda[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_lda[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_mf_lda_micro = precision_score(y_test_mf, mf_predictions_lda_test, average='micro')
precision_mf_lda_weighted = precision_score(y_test_mf, mf_predictions_lda_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_lda_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_lda_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_lda_micro = recall_score(y_test_mf, mf_predictions_lda_test, average='micro')
recall_mf_lda_weighted = recall_score(y_test_mf, mf_predictions_lda_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_lda_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_lda_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### RANDOM FOREST

# In[749]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_mf_rdf = sc.fit_transform(x_train_mf)
x_test_mf_rdf = sc.transform(x_test_mf)

#definition 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)

#Plug the model to the training set
mf_model_rdf = regressor.fit(x_train_mf_rdf, y_train_mf)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_rdf_mf = cross_val_score(mf_model_rdf, x_train_mf_rdf, y_train_mf,cv=5)#,scoring='accuracy') ==> Doesnt work with accuracy
print('not sure it is correct accuracy number as we cannot pass the parameter scoring')
scval_rdf_mf= print("CROSS VALIDATION\nAccuracy Score of the LDA model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_rdf_mf.mean(), scores_cross_val_rdf_mf.std() * 2),'\n')
scval_rdf_mf 

#Evaluate the model on the test set
mf_predictions_rdf_test = mf_model_rdf.predict(x_test_mf_rdf)
#mf_acc_rdf = round(accuracy_score(y_test_mf,mf_predictions_rdf_test),3)
#print("TEST SET\nAccuracy Score :", mf_acc_rdf)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_rdf_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_rdf_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_rdf_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX UNAVAILABLE")
#print("CONFUSION MATRIX (Test Set)")
#cmx_mf_rdf = confusion_matrix(y_test_mf, mf_predictions_rdf_test)
#print(cmx_mf_rdf,'\n')
#print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_rdf[1,1])
#print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_rdf[0,0])
#print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_rdf[0,1])
#print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_rdf[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
#precision_mf_rdf_micro = precision_score(y_test_mf, mf_predictions_rdf_test, average='micro')
#precision_mf_rdf_weighted = precision_score(y_test_mf, mf_predictions_rdf_test, average='weighted')
#print('Precision Rate - Micro :',round(precision_mf_rdf_micro,3))
#print('Precision Rate - Weighted :',round(precision_mf_rdf_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
#recall_mf_rdf_micro = recall_score(y_test_mf, mf_predictions_rdf_test, average='micro')
#recall_mf_rdf_weighted = recall_score(y_test_mf, mf_predictions_rdf_test, average='weighted')
#print('Recall Rate - Micro :',round(recall_mf_rdf_micro,3))
#print('Recall Rate - Weighted :',round(recall_mf_rdf_weighted,3),'\n\n')

#print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### NAIVE BAYES

# In[750]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#feature scaling
from sklearn.preprocessing import MinMaxScaler
mmsc = MinMaxScaler()
x_train_mf_nby = mmsc.fit_transform(x_train_mf)
x_test_mf_nby = mmsc.transform(x_test_mf)

#Plug the model to the training set
mnb = MultinomialNB()
mf_model_nby = mnb.fit(x_train_mf_nby, y_train_mf)#.values.ravel()

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_nby_mf = cross_val_score(mf_model_nby, x_train_mf_nby, y_train_mf, cv=5, scoring='accuracy')
scval_nby_mf = print("CROSS VALIDATION\nAccuracy Score of the NAIVE BAYERS model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_nby_mf.mean(), scores_cross_val_nby_mf.std() * 2),'\n')
scval_nby_mf 

#Evaluate the model on the test set
mf_predictions_nby_test = mf_model_nby.predict(x_test_mf_nby)
mf_acc_nby = round(accuracy_score(y_test_mf,mf_predictions_nby_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_nby)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_nby_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_nby_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_nby_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_nby = confusion_matrix(y_test_mf,mf_predictions_nby_test)
print(cmx_mf_nby,'\n')
TP_mf_nby = cmx_mf_nby[1,1]
TN_mf_nby = cmx_mf_nby[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_nby[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_nby[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_nby[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_nby[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_nby_micro = precision_score(y_test_mf, mf_predictions_nby_test, average='micro')
precision_mf_nby_weighted = precision_score(y_test_mf, mf_predictions_nby_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_nby_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_nby_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_nby_micro = recall_score(y_test_mf, mf_predictions_nby_test, average='micro')
recall_mf_nby_weighted = recall_score(y_test_mf, mf_predictions_nby_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_nby_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_nby_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### XGB Classifier

# In[751]:


#!pip install xgboost
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
mf_xgb_clf = XGBClassifier()

#Plug the model to the training set
mf_model_xgb = mf_xgb_clf.fit(x_train_mf, y_train_mf)#.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_xgb_mf = cross_val_score(mf_model_xgb, x_train_mf, y_train_mf, cv=5, scoring='accuracy')
scval_xgb_mf = print("CROSS VALIDATION\nAccuracy Score of the XGB model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_xgb_mf.mean(), scores_cross_val_xgb_mf.std() * 2),'\n')
scval_xgb_mf

#Evaluate the model on the test set
mf_predictions_xgb_test = mf_model_xgb.predict(x_test_mf)
mf_acc_xgb = round(accuracy_score(y_test_mf,mf_predictions_xgb_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_xgb)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_xgb_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_xgb_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_xgb_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_xgb = confusion_matrix(y_test_mf,mf_predictions_xgb_test)
print(cmx_mf_xgb,'\n')
TP_mf_xgb = cmx_mf_xgb[1,1]
TN_mf_xgb = cmx_mf_xgb[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_xgb[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_xgb[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_xgb[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_xgb[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_xgb_micro = precision_score(y_test_mf, mf_predictions_xgb_test, average='micro')
precision_mf_xgb_weighted = precision_score(y_test_mf, mf_predictions_xgb_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_xgb_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_xgb_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_xgb_micro = recall_score(y_test_mf, mf_predictions_xgb_test, average='micro')
recall_mf_xgb_weighted = recall_score(y_test_mf, mf_predictions_xgb_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_xgb_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_xgb_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVC 

# In[1140]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')

#Plug the model to the training set
mf_model_svc = svclassifier.fit(x_train_mf, y_train_mf)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_svc_mf = cross_val_score(mf_model_svc, x_train_mf, y_train_mf, cv=5, scoring='accuracy')
scval_svc_mf = print("CROSS VALIDATION\nAccuracy Score of the SVC model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_svc_mf.mean(), scores_cross_val_svc_mf.std() * 2),'\n')
scval_svc_mf

#Evaluate the model on the test set
mf_predictions_svc_test = mf_model_svc.predict(x_test_mf)
mf_acc_svc = round(accuracy_score(y_test_mf,mf_predictions_svc_test),3)
print("TEST SET\nAccuracy Score :", mf_acc_svc)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf, mf_predictions_svc_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf, mf_predictions_svc_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf, mf_predictions_svc_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_svc = confusion_matrix(y_test_mf,mf_predictions_svc_test)
print(cmx_mf_svc,'\n')
TP_mf_svc = cmx_mf_svc[1,1]
TN_mf_svc = cmx_mf_svc[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_svc[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_svc[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_svc[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_svc[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_svc_micro = precision_score(y_test_mf, mf_predictions_svc_test, average='micro')
precision_mf_svc_weighted = precision_score(y_test_mf, mf_predictions_svc_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_svc_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_svc_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_svc_micro = recall_score(y_test_mf, mf_predictions_svc_test, average='micro')
recall_mf_svc_weighted = recall_score(y_test_mf, mf_predictions_svc_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_svc_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_svc_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ### 3. RESULT SALE_MF :

# ###### a. POST CROSS VALIDATION - Accuracy

# In[753]:


print('CROSS VALIDATION ACCURACY (SALE_MF) :\n')
print('- DECISION TREE :            %0.2f (+/- %0.2f)' % (scores_cross_val_dt_mf.mean(), scores_cross_val_dt_mf.std() * 2))
print('- KNN :                      %0.2f (+/- %0.2f)' % (scores_cross_val_knn_mf.mean(), scores_cross_val_knn_mf.std() * 2))
print('- Logistic Regression :      %0.2f (+/- %0.2f)' % (scores_cross_val_lr_mf.mean(), scores_cross_val_lr_mf.std() * 2))
print('- SVM :                      %0.2f (+/- %0.2f)' % (scores_cross_val_svm_mf.mean(), scores_cross_val_svm_mf.std() * 2))
print('- LDA :                      %0.2f (+/- %0.2f)' % (scores_cross_val_lda_mf.mean(), scores_cross_val_lda_mf.std() * 2))
print('- Random Forest (not sure)  %0.2f (+/- %0.2f)' % (scores_cross_val_rdf_mf.mean(), scores_cross_val_rdf_mf.std() * 2))
print('- Naive Bayers :             %0.2f (+/- %0.2f)' % (scores_cross_val_nby_mf.mean(), scores_cross_val_nby_mf.std() * 2))
print('- XGB :                      %0.2f (+/- %0.2f)' % (scores_cross_val_xgb_mf.mean(), scores_cross_val_xgb_mf.std() * 2))
print('- SVC :                      %0.2f (+/- %0.2f)' % (scores_cross_val_xgb_mf.mean(), scores_cross_val_xgb_mf.std() * 2))


# ###### b.  POST TEST KPI ANALYSIS

# In[754]:


print('CONFUSION MATRIX POST TEST ANALYSIS (SALE_MF) :\n')
print('DECISION TREES')
print('True Positive :',TP_mf_dt,' ------------> TARGET')
print('True Negative :',TN_mf_dt)
print('False Positive :',FP_mf_dt)
print('False Negative :',FN_mf_dt)
print('Precision - micro : ',round(precision_mf_dt_micro,2),'\nPrecision - weighted :',round(precision_mf_dt_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_dt_micro,2),'\nRecall - weighted :',round(recall_mf_dt_weighted,2),' -----> TARGET\n')

print('KNN')
print('True Positive :',TP_mf_knn,' ------------> TARGET')
print('True Negative :',TN_mf_knn)
print('Precision - micro : ',round(precision_mf_knn_micro,2),'\nPrecision - weighted :',round(precision_mf_knn_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_knn_micro,2),'\nRecall - weighted :',round(recall_mf_knn_weighted,2),' -----> TARGET\n')

print('LOGISTIC REGRESSION')
print('True Positive :',TP_mf_lr,' ------------> TARGET')
print('True Negative :',TN_mf_lr)
print('Precision - micro : ',round(precision_mf_lr_micro,2),'\nPrecision - weighted :',round(precision_mf_lr_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_lr_micro,2),'\nRecall - weighted :',round(recall_mf_lr_weighted,2),' -----> TARGET\n')

print('SVM')
print('True Positive :',TP_mf_svm,' ------------> TARGET')
print('True Negative :',TN_mf_svm)
print('Precision - micro : ',round(precision_mf_svm_micro,2),'\nPrecision - weighted :',round(precision_mf_svm_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_svm_micro,2),'\nRecall - weighted :',round(recall_mf_svm_weighted,2),' -----> TARGET\n')

print('LDA')
print('True Positive :',TP_mf_lda,' ------------> TARGET')
print('True Negative :',TN_mf_lda)
print('False Positive :',FP_mf_lda)
print('False Negative :',FN_mf_lda)
print('Precision - micro : ',round(precision_mf_lda_micro,2),'\nPrecision - weighted :',round(precision_mf_lda_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_lda_micro,2),'\nRecall - weighted :',round(recall_mf_lda_weighted,2),' -----> TARGET\n')

print('RANDOM FOREST')
print('non available\n')

print('NAIVE BAYES')
print('True Positive :',TP_mf_nby,' ------------> TARGET')
print('True Negative :',TN_mf_nby)
print('Precision - micro : ',round(precision_mf_nby_micro,2),'\nPrecision - weighted :',round(precision_mf_nby_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_nby_micro,2),'\nRecall - weighted :',round(recall_mf_nby_weighted,2),' -----> TARGET\n')

print('XGB')
print('True Positive :',TP_mf_xgb,' ------------> TARGET')
print('True Negative :',TN_mf_xgb)
print('Precision - micro : ',round(precision_mf_xgb_micro,2),'\nPrecision - weighted :',round(precision_mf_xgb_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_xgb_micro,2),'\nRecall - weighted :',round(recall_mf_xgb_weighted,2),' -----> TARGET\n')

print('SVC')
print('True Positive :',TP_mf_svc,' ------------> TARGET')
print('True Negative :',TN_mf_svc)
print('Precision - micro : ',round(precision_mf_svc_micro,2),'\nPrecision - weighted :',round(precision_mf_svc_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_svc_micro,2),'\nRecall - weighted :',round(recall_mf_svc_weighted,2),' -----> TARGET\n')


# ### TEST SALE_CL : Classification

# #### 1. CREATE THE TRAINING SET FOR THE SALE_CL TARGET

# In[758]:


CL_ = df2.loc[:, df2.columns.str.endswith("CL")]
print(CL_.head(0))


# In[759]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_cl = df2[['Sale_CL']]


# In[760]:


import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cl, x_test_cl, y_train_cl,y_test_cl = train_test_split(x,y_cl,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_cl.shape)
print('Training Labels Shape:', y_train_cl.shape)
print('Testing Features Shape:', x_test_cl.shape)
print('Testing Labels Shape:', y_test_cl.shape)


# #### 2. EVALUATE ACCURACY OF MODELS ON THE SALE_CL TARGET

# ##### DECISION TREE REGRESSOR

# In[761]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
regressor = DecisionTreeRegressor()

#Plug the model to the training set
cl_model_dt = regressor.fit(x_train_cl,y_train_cl)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_dt_cl = cross_val_score(cl_model_dt, x_train_cl, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the DT model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_dt_cl.mean(), scores_cross_val_dt_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_dt_test = cl_model_dt.predict(x_test_cl)
cl_acc_dt = round(accuracy_score(y_test_cl,cl_predictions_dt_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_dt)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:',round(metrics.mean_absolute_error(y_test_cl, cl_predictions_dt_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_dt_test),3),'\n\n')
#print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_dt_test),3)))

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_dt = confusion_matrix(y_test_cl,cl_predictions_dt_test)
TP_cl_dt = cmx_cl_dt[1,1]
TN_cl_dt = cmx_cl_dt[0,0]
print(cmx_cl_dt,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_dt[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_dt[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_dt[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_dt[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cl_dt_micro = precision_score(y_test_cl, cl_predictions_dt_test, average='micro')
precision_cl_dt_weighted = precision_score(y_test_cl, cl_predictions_dt_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_dt_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_dt_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_dt_micro = recall_score(y_test_mf, cl_predictions_dt_test, average='micro')
recall_cl_dt_weighted = recall_score(y_test_mf, cl_predictions_dt_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_dt_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_dt_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### K-NEAREST NEIGHBOR MODEL

# In[762]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 4) #4 is the best classifier in this case

#Plug the model to the training set
cl_model_knn = knn.fit(x_train_cl,y_train_cl.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_knn_cl = cross_val_score(cl_model_knn, x_train_cl, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the KNN model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_knn_cl.mean(), scores_cross_val_knn_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_knn_test = cl_model_knn.predict(x_test_cl)
cl_acc_knn = round(accuracy_score(y_test_cl,cl_predictions_knn_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_knn)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_knn_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_knn_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_knn_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_knn = confusion_matrix(y_test_cl,cl_predictions_knn_test)
TP_cl_knn = cmx_cl_knn[1,1]
TN_cl_knn = cmx_cl_knn[0,0]
print(cmx_cl_knn,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_knn[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_knn[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_knn[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_knn[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cl_knn_micro = precision_score(y_test_cl, cl_predictions_knn_test, average='micro')
precision_cl_knn_weighted = precision_score(y_test_cl, cl_predictions_knn_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_knn_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_knn_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_knn_micro = recall_score(y_test_cl, cl_predictions_knn_test, average='micro')
recall_cl_knn_weighted = recall_score(y_test_cl, cl_predictions_knn_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_knn_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_knn_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### LOGISTIC REGRESSION

# In[763]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logmodel = LogisticRegression(max_iter=1000) #solver sklearn
import warnings
warnings.filterwarnings('ignore')

#Plug the model to the training set
cl_model_lr = logmodel.fit(x_train_cl,y_train_cl.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_lr_cl = cross_val_score(cl_model_lr, x_train_cl, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the Logistic Regression model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_lr_cl.mean(), scores_cross_val_lr_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_lr_test = cl_model_lr.predict(x_test_cl)
cl_acc_lr = round(accuracy_score(y_test_cl,cl_predictions_lr_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_lr)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_lr_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_lr_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_lr_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lr = confusion_matrix(y_test_cl,cl_predictions_lr_test)
TP_cl_lr = cmx_cl_lr[1,1]
TN_cl_lr = cmx_cl_lr[0,0]
print(cmx_cl_lr,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_lr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_lr[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cl_lr_micro = precision_score(y_test_cl, cl_predictions_lr_test, average='micro')
precision_cl_lr_weighted = precision_score(y_test_cl, cl_predictions_lr_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_lr_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_lr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_lr_micro = recall_score(y_test_cl, cl_predictions_lr_test, average='micro')
recall_cl_lr_weighted = recall_score(y_test_cl, cl_predictions_lr_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_lr_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_lr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVM

# In[764]:


from sklearn import svm
from sklearn.metrics import accuracy_score
svmmodel = svm.SVC()

#Plug the model to the training set
cl_model_svm = svmmodel.fit(x_train_cl,y_train_cl.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_svm_cl = cross_val_score(cl_model_svm, x_train_cl, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the SVM model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_svm_cl.mean(), scores_cross_val_svm_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_svm_test = cl_model_svm.predict(x_test_cl)
cl_acc_svm = round(accuracy_score(y_test_cl,cl_predictions_svm_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_svm)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_svm_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_svm_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_svm_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_svm = confusion_matrix(y_test_cl,cl_predictions_svm_test)
TP_cl_svm = cmx_cl_svm[1,1]
TN_cl_svm = cmx_cl_svm[0,0]
print(cmx_cl_svm,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_svm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_svm[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_svm[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_svm[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cl_svm_micro = precision_score(y_test_cl, cl_predictions_svm_test, average='micro')
precision_cl_svm_weighted = precision_score(y_test_cl, cl_predictions_svm_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_svm_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_svm_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_svm_micro = recall_score(y_test_cl, cl_predictions_svm_test, average='micro')
recall_cl_svm_weighted = recall_score(y_test_cl, cl_predictions_svm_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_svm_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_svm_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### LDA

# In[765]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_cl_sc = sc.fit_transform(x_train_cl)
x_test_cl_sc = sc.transform(x_test_cl)

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_train_cl_sc = lda.fit_transform(x_train_cl_sc, y_train_cl)
x_test_cl_sc = lda.transform(x_test_cl_sc)

# Comparison w/Random Forest
from sklearn.ensemble import RandomForestClassifier
cl_classifier = RandomForestClassifier(max_depth=2, random_state=0)
cl_classifier_lda = cl_classifier.fit(x_train_cl_sc, y_train_cl)#.values.ravel()

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_lda_cl = cross_val_score(cl_classifier_lda, x_train_cl_sc, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the LDA model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_lda_cl.mean(), scores_cross_val_lda_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_lda_test = cl_classifier_lda.predict(x_test_cl_sc)
cl_acc_lda = round(accuracy_score(y_test_cl,cl_predictions_lda_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_lda)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_lda_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_lda_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_lda_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lda = confusion_matrix(y_test_cl,cl_predictions_lda_test)
TP_cl_lda = cmx_cl_lda[1,1]
TN_cl_lda = cmx_cl_lda[0,0]
print(cmx_cl_lda,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lda[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lda[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_lda[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_lda[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cl_lda_micro = precision_score(y_test_cl, cl_predictions_lda_test, average='micro')
precision_cl_lda_weighted = precision_score(y_test_cl, cl_predictions_lda_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_lda_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_lda_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_lda_micro = recall_score(y_test_cl, cl_predictions_lda_test, average='micro')
recall_cl_lda_weighted = recall_score(y_test_cl, cl_predictions_lda_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_lda_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_lda_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### RANDOM FOREST

# In[766]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_cl = sc.fit_transform(x_train_cl)
x_test_cl = sc.transform(x_test_cl)

#definition 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)

#Plug the model to the training set
cl_model_rdf = regressor.fit(x_train_cl, y_train_cl)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_rdf_cl = cross_val_score(cl_model_rdf, x_train_cl, y_train_cl, cv=5)#, scoring='accuracy')
print('not sure it is correct accuracy number as we cannot pass the parameter scoring')
print("CROSS VALIDATION\nAccuracy Score of the LDA model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_rdf_cl.mean(), scores_cross_val_rdf_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_rdf_test = cl_model_rdf.predict(x_test_cl)
#cl_acc_rdf = round(accuracy_score(y_test_cl,cl_predictions_rdf_test),3)
#print("TEST SET\nAccuracy Score :", cl_acc_rdf)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_rdf_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_rdf_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_rdf_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
#print("CONFUSION MATRIX (Test Set)")
print("CONFUSION MATRIX UNAVAILABLE")
#cmx_cl_rdf = confusion_matrix(y_test_cl,cl_predictions_rdf_test)
#TP_cl_rdf = cmx_cl_rdf[1,1]
#TN_cl_rdf = cmx_cl_rdf[0,0]
#print(cmx_cl_rdf,'\n')
#print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_rdf[1,1])
#print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_rdf[0,0])
#print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_rdf[0,1])
#print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_rdf[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
#precision_cl_rdf_micro = precision_score(y_test_cl, cl_predictions_rdf_test, average='micro')
#precision_cl_rdf_weighted = precision_score(y_test_cl, cl_predictions_rdf_test, average='weighted')
#print('Precision Rate - Micro :',round(precision_cl_rdf_micro,3))
#print('Precision Rate - Weighted :',round(precision_cl_rdf_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
#recall_cl_rdf_micro = recall_score(y_test_cl, cl_predictions_rdf_test, average='micro')
#recall_cl_rdf_weighted = recall_score(y_test_cl, cl_predictions_rdf_test, average='weighted')
#print('Recall Rate - Micro :',round(recall_cl_rdf_micro,3))
#print('Recall Rate - Weighted :',round(recall_cl_rdf_weighted,3),'\n\n')

#print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')



# ##### NAIVE BAYES

# In[767]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#feature scaling
from sklearn.preprocessing import MinMaxScaler
mmsc = MinMaxScaler()
x_train_cl_nby = mmsc.fit_transform(x_train_cl)
x_test_cl_nby = mmsc.transform(x_test_cl)

#Plug the model to the training set
mnb = MultinomialNB()
cl_model_nby = mnb.fit(x_train_cl_nby, y_train_cl)#.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_nby_cl = cross_val_score(cl_model_nby, x_train_cl_nby, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the NAIVE BAYERS model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_nby_cl.mean(), scores_cross_val_nby_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_nby_test = cl_model_nby.predict(x_test_cl_nby)
cl_acc_nby = round(accuracy_score(y_test_cl,cl_predictions_nby_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_nby)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_nby_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_nby_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_nby_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_nby = confusion_matrix(y_test_cl,cl_predictions_nby_test)
TP_cl_nby = cmx_cl_nby[1,1]
TN_cl_nby = cmx_cl_nby[0,0]
print(cmx_cl_nby,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_nby[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_nby[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_nby[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_nby[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_nby_micro = precision_score(y_test_cl, cl_predictions_nby_test, average='micro')
precision_cl_nby_weighted = precision_score(y_test_cl, cl_predictions_nby_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_nby_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_nby_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_nby_micro = recall_score(y_test_cl, cl_predictions_nby_test, average='micro')
recall_cl_nby_weighted = recall_score(y_test_cl, cl_predictions_nby_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_nby_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_nby_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### XGB Classifier

# In[768]:


#!pip install xgboost
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
cl_xgb_clf = XGBClassifier()

#Plug the model to the training set
cl_model_xgb = cl_xgb_clf.fit(x_train_cl, y_train_cl)#.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_xgb_cl = cross_val_score(cl_model_xgb, x_train_cl, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the XGB model on Sale_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_xgb_cl.mean(), scores_cross_val_xgb_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_xgb_test = cl_model_xgb.predict(x_test_cl)
cl_acc_xgb = round(accuracy_score(y_test_cl,cl_predictions_xgb_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_xgb)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_xgb_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_xgb_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_xgb_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_xgb = confusion_matrix(y_test_cl,cl_predictions_xgb_test)
TP_cl_xgb = cmx_cl_xgb[1,1]
TN_cl_xgb = cmx_cl_xgb[0,0]
print(cmx_cl_xgb,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_xgb[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_xgb[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_xgb[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_xgb[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_xgb_micro = precision_score(y_test_cl, cl_predictions_xgb_test, average='micro')
precision_cl_xgb_weighted = precision_score(y_test_cl, cl_predictions_xgb_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_xgb_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_xgb_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_xgb_micro = recall_score(y_test_cl, cl_predictions_xgb_test, average='micro')
recall_cl_xgb_weighted = recall_score(y_test_cl, cl_predictions_xgb_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_xgb_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_xgb_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVC 

# In[769]:


## USUALLY TAKES FOREVER TO RUN

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',max_iter=500000)

#Plug the model to the training set
cl_model_svc = svclassifier.fit(x_train_cl, y_train_cl)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_svc_cl = cross_val_score(cl_model_svc, x_train_cl, y_train_cl, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the XGB model on Sale_MF : %0.2f (+/- %0.2f)" % (scores_cross_val_svc_cl.mean(), scores_cross_val_svc_cl.std() * 2),'\n')

#Evaluate the model on the test set
cl_predictions_svc_test = cl_model_svc.predict(x_test_cl)
cl_acc_svc = round(accuracy_score(y_test_cl,cl_predictions_svc_test),3)
print("TEST SET\nAccuracy Score :", cl_acc_svc)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl, cl_predictions_svc_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl, cl_predictions_svc_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl, cl_predictions_svc_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_svc = confusion_matrix(y_test_cl,cl_predictions_svc_test)
TP_cl_svc = cmx_cl_svc[1,1]
TN_cl_svc = cmx_cl_svc[0,0]
print(cmx_cl_svc,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_svc[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_svc[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_svc[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_svc[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_svc_micro = precision_score(y_test_cl, cl_predictions_svc_test, average='micro')
precision_cl_svc_weighted = precision_score(y_test_cl, cl_predictions_svc_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_svc_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_svc_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_svc_micro = recall_score(y_test_cl, cl_predictions_svc_test, average='micro')
recall_cl_svc_weighted = recall_score(y_test_cl, cl_predictions_svc_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_svc_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_svc_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# #### 3. RESULT SALE_CL  : LOGISTIC REGRESSION

# In[771]:


print('CROSS VALIDATION ACCURACY (SALE_CL) :\n')
print('- DECISION TREE :            %0.2f (+/- %0.2f)' % (scores_cross_val_dt_cl.mean(), scores_cross_val_dt_cl.std() * 2))
print('- KNN :                      %0.2f (+/- %0.2f)' % (scores_cross_val_knn_cl.mean(), scores_cross_val_knn_cl.std() * 2))
print('- Logistic Regression :      %0.2f (+/- %0.2f)' % (scores_cross_val_lr_cl.mean(), scores_cross_val_lr_cl.std() * 2))
print('- SVM :                      %0.2f (+/- %0.2f)' % (scores_cross_val_svm_cl.mean(), scores_cross_val_svm_cl.std() * 2))
print('- LDA :                      %0.2f (+/- %0.2f)' % (scores_cross_val_lda_cl.mean(), scores_cross_val_lda_cl.std() * 2))
print('- Random Forest (not sure)  %0.2f (+/- %0.2f)' % (scores_cross_val_rdf_cl.mean(), scores_cross_val_rdf_cl.std() * 2))
print('- Naive Bayers :             %0.2f (+/- %0.2f)' % (scores_cross_val_nby_cl.mean(), scores_cross_val_nby_cl.std() * 2))
print('- XGB :                      %0.2f (+/- %0.2f)' % (scores_cross_val_xgb_cl.mean(), scores_cross_val_xgb_cl.std() * 2))
print('- SVC :                      %0.2f (+/- %0.2f)' % (scores_cross_val_xgb_cl.mean(), scores_cross_val_xgb_cl.std() * 2))


# In[772]:


print('CONFUSION MATRIX POST TEST ANALYSIS (SALE_CL) :\n')
print('DECISION TREES')
print('True Positive :',TP_cl_dt,' ------------> TARGET')
print('True Negative :',TN_cl_dt)
#print('False Positive :',FP_cl_dt)
#print('False Negative :',FN_cl_dt)
print('Precision - micro : ',round(precision_cl_dt_micro,2),'\nPrecision - weighted :',round(precision_cl_dt_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_dt_micro,2),'\nRecall - weighted :',round(recall_cl_dt_weighted,2),' -----> TARGET\n')

print('KNN')
print('True Positive :',TP_cl_knn,' ------------> TARGET')
print('True Negative :',TN_cl_knn)
print('Precision - micro : ',round(precision_cl_knn_micro,2),'\nPrecision - weighted :',round(precision_cl_knn_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_knn_micro,2),'\nRecall - weighted :',round(recall_cl_knn_weighted,2),' -----> TARGET\n')

print('LOGISTIC REGRESSION')
print('True Positive :',TP_cl_lr,' ------------> TARGET')
print('True Negative :',TN_cl_lr)
print('Precision - micro : ',round(precision_cl_lr_micro,2),'\nPrecision - weighted :',round(precision_cl_lr_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_lr_micro,2),'\nRecall - weighted :',round(recall_cl_lr_weighted,2),' -----> TARGET\n')

print('SVM')
print('True Positive :',TP_cl_svm,' ------------> TARGET')
print('True Negative :',TN_cl_svm)
print('Precision - micro : ',round(precision_cl_svm_micro,2),'\nPrecision - weighted :',round(precision_cl_svm_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_svm_micro,2),'\nRecall - weighted :',round(recall_cl_svm_weighted,2),' -----> TARGET\n')

print('LDA')
print('True Positive :',TP_cl_lda,' ------------> TARGET')
print('True Negative :',TN_cl_lda)
#print('False Positive :',FP_cl_lda)
#print('False Negative :',FN_cl_lda)
print('Precision - micro : ',round(precision_cl_lda_micro,2),'\nPrecision - weighted :',round(precision_cl_lda_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_lda_micro,2),'\nRecall - weighted :',round(recall_cl_lda_weighted,2),' -----> TARGET\n')

print('RANDOM FOREST')
print('non available\n')

print('NAIVE BAYES')
print('True Positive :',TP_cl_nby,' ------------> TARGET')
print('True Negative :',TN_cl_nby)
print('Precision - micro : ',round(precision_cl_nby_micro,2),'\nPrecision - weighted :',round(precision_cl_nby_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_nby_micro,2),'\nRecall - weighted :',round(recall_cl_nby_weighted,2),' -----> TARGET\n')

print('XGB')
print('True Positive :',TP_cl_xgb,' ------------> TARGET')
print('True Negative :',TN_cl_xgb)
print('Precision - micro : ',round(precision_cl_xgb_micro,2),'\nPrecision - weighted :',round(precision_cl_xgb_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_xgb_micro,2),'\nRecall - weighted :',round(recall_cl_xgb_weighted,2),' -----> TARGET\n')

print('SVC')
print('True Positive :',TP_cl_svc,' ------------> TARGET')
print('True Negative :',TN_cl_svc)
print('Precision - micro : ',round(precision_cl_svc_micro,2),'\nPrecision - weighted :',round(precision_cl_svc_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_svc_micro,2),'\nRecall - weighted :',round(recall_cl_svc_weighted,2),' -----> TARGET\n')


# ### TEST SALE_CC : Classification

# #### 1. CREATE THE TRAINING SET FOR THE SALE_CC TARGET

# In[773]:


CC_ = df2.loc[:, df2.columns.str.endswith("CC")]
print(CC_.head(0))


# In[774]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_cc = df2[['Sale_CC']]


# In[775]:


import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cc, x_test_cc, y_train_cc,y_test_cc = train_test_split(x,y_cc,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_cc.shape)
print('Training Labels Shape:', y_train_cc.shape)
print('Testing Features Shape:', x_test_cc.shape)
print('Testing Labels Shape:', y_test_cc.shape)


# #### 2. EVALUATE ACCURACY OF MODELS ON THE SALE_CC TARGET

# ##### DECISION TREE REGRESSOR

# In[776]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
regressor = DecisionTreeRegressor()

#Plug the model to the training set
cc_model_dt = regressor.fit(x_train_cc,y_train_cc)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_dt_cc = cross_val_score(cc_model_dt, x_train_cc, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the DT model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_dt_cc.mean(), scores_cross_val_dt_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_dt_test = cc_model_dt.predict(x_test_cc)
cc_acc_dt = round(accuracy_score(y_test_mf,cc_predictions_dt_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_dt)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:',round(metrics.mean_absolute_error(y_test_cc, cc_predictions_dt_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_dt_test),3),'\n\n')
#print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_dt_test),3)))

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_dt = confusion_matrix(y_test_cc,cc_predictions_dt_test)
TP_cc_dt = cmx_cc_dt[1,1]
TN_cc_dt = cmx_cc_dt[0,0]
print(cmx_cc_dt,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_dt[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_dt[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_dt[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_dt[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cc_dt_micro = precision_score(y_test_cc, cc_predictions_dt_test, average='micro')
precision_cc_dt_weighted = precision_score(y_test_cc, cc_predictions_dt_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_dt_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_dt_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_dt_micro = recall_score(y_test_cc, cc_predictions_dt_test, average='micro')
recall_cc_dt_weighted = recall_score(y_test_cc, cc_predictions_dt_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_dt_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_dt_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### K-NEAREST NEIGHBOR MODEL

# In[777]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 4) #4 is the best classifier in this case

#Plug the model to the training set
cc_model_knn = knn.fit(x_train_cc,y_train_cc.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_knn_cc = cross_val_score(cc_model_knn, x_train_cc, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the KNN model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_knn_cc.mean(), scores_cross_val_knn_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_knn_test = cc_model_knn.predict(x_test_cc)
cc_acc_knn = round(accuracy_score(y_test_cc,cc_predictions_knn_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_knn)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_knn_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_knn_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_knn_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_knn = confusion_matrix(y_test_cc,cc_predictions_knn_test)
TP_cc_knn = cmx_cc_knn[1,1]
TN_cc_knn = cmx_cc_knn[0,0]
print(cmx_cc_knn,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_knn[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_knn[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_knn[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_knn[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cc_knn_micro = precision_score(y_test_cc, cc_predictions_knn_test, average='micro')
precision_cc_knn_weighted = precision_score(y_test_cc, cc_predictions_knn_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_knn_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_knn_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_knn_micro = recall_score(y_test_cc, cc_predictions_knn_test, average='micro')
recall_cc_knn_weighted = recall_score(y_test_cc, cc_predictions_knn_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_knn_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_knn_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### LOGISTIC REGRESSION

# In[778]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logmodel = LogisticRegression(max_iter=1000) #solver sklearn
import warnings
warnings.filterwarnings('ignore')

#Plug the model to the training set
cc_model_lr = logmodel.fit(x_train_cc,y_train_cc.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_lr_cc = cross_val_score(cc_model_lr, x_train_cc, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the Logistic Regression model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_lr_cc.mean(), scores_cross_val_lr_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_lr_test = cc_model_lr.predict(x_test_cc)
cc_acc_lr = round(accuracy_score(y_test_cc,cc_predictions_lr_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_lr)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_lr_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_lr_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_lr_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_lr = confusion_matrix(y_test_cc,cc_predictions_lr_test)
print(cmx_cc_lr,'\n')
TP_cc_lr = cmx_cc_lr[1,1]
TN_cc_lr = cmx_cc_lr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_lr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_lr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_lr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_lr[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cc_lr_micro = precision_score(y_test_cc, cc_predictions_lr_test, average='micro')
precision_cc_lr_weighted = precision_score(y_test_cc, cc_predictions_lr_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_lr_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_lr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_lr_micro = recall_score(y_test_cc, cc_predictions_lr_test, average='micro')
recall_cc_lr_weighted = recall_score(y_test_cc, cc_predictions_lr_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_lr_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_lr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVM

# In[779]:


from sklearn import svm
from sklearn.metrics import accuracy_score
svmmodel = svm.SVC()

#Plug the model to the training set
cc_model_svm = svmmodel.fit(x_train_cc,y_train_cc)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_svm_cc = cross_val_score(cc_model_svm, x_train_cc, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the SVM model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_svm_cc.mean(), scores_cross_val_svm_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_svm_test = cc_model_svm.predict(x_test_cc)
cc_acc_svm = round(accuracy_score(y_test_cc,cc_predictions_svm_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_svm)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_svm_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_svm_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_svm_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_svm = confusion_matrix(y_test_cc,cc_predictions_svm_test)
print(cmx_cc_svm,'\n')
TP_cc_svm = cmx_cc_svm[1,1]
TN_cc_svm = cmx_cc_svm[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_svm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_svm[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_svm[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_svm[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cc_svm_micro = precision_score(y_test_cc, cc_predictions_svm_test, average='micro')
precision_cc_svm_weighted = precision_score(y_test_cc, cc_predictions_svm_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_svm_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_svm_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_svm_micro = recall_score(y_test_cc, cc_predictions_svm_test, average='micro')
recall_cc_svm_weighted = recall_score(y_test_cc, cc_predictions_svm_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_svm_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_svm_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### LDA

# In[780]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_cc = sc.fit_transform(x_train_cc)
x_test_cc = sc.transform(x_test_cc)

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_train_cc = lda.fit_transform(x_train_cc, y_train_cc)
x_test_cc = lda.transform(x_test_cc)

# Comparison w/Random Forest
from sklearn.ensemble import RandomForestClassifier
cc_classifier = RandomForestClassifier(max_depth=2, random_state=0)
cc_classifier_lda = cc_classifier.fit(x_train_cc, y_train_cc)#.values.ravel()

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_lda_cc = cross_val_score(cc_classifier_lda, x_train_cc, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the LDA model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_lda_cc.mean(), scores_cross_val_lda_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_lda_test = cc_classifier_lda.predict(x_test_cc)
cc_acc_lda = round(accuracy_score(y_test_cc,cc_predictions_lda_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_lda)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_lda_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_lda_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_lda_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_lda = confusion_matrix(y_test_cc,cc_predictions_lda_test)
TP_cc_lda = cmx_cc_lda[1,1]
TN_cc_lda = cmx_cc_lda[0,0]
print(cmx_cc_lda,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_lda[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_lda[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_lda[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_lda[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cc_lda_micro = precision_score(y_test_cc, cc_predictions_lda_test, average='micro')
precision_cc_lda_weighted = precision_score(y_test_cc, cc_predictions_lda_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_lda_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_lda_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_lda_micro = recall_score(y_test_cc, cc_predictions_lda_test, average='micro')
recall_cc_lda_weighted = recall_score(y_test_cc, cc_predictions_lda_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_lda_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_lda_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### RANDOM FOREST

# In[781]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_cc = sc.fit_transform(x_train_cc)
x_test_cc = sc.transform(x_test_cc)

#definition 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)

#Plug the model to the training set
cc_model_rdf = regressor.fit(x_train_cc, y_train_cc)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_rdf_cc = cross_val_score(cc_model_rdf, x_train_cc, y_train_cc,cv=5)#,scoring='accuracy') #==> Doesnt work with accuracy
print('not sure it is correct accuracy number as we cannot pass the parameter scoring')
print("CROSS VALIDATION\nAccuracy Score of the LDA model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_rdf_cc.mean(), scores_cross_val_rdf_cc.std() * 2),'\n')


#Evaluate the model on the test set
cc_predictions_rdf_test = cc_model_rdf.predict(x_test_cc)
#mf_acc_rdf = round(accuracy_score(y_test_mf,mf_predictions_rdf_test),3)
#print("TEST SET\nAccuracy Score :", mf_acc_rdf)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_rdf_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_rdf_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_rdf_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX UNAVAILABLE")
#print("CONFUSION MATRIX (Test Set)")
#cmx_mf_rdf = confusion_matrix(y_test_mf, mf_predictions_rdf_test)
#TP_cc_rdf = cmx_cc_rdf[1,1]
#TN_cc_rdf = cmx_cc_rdf[0,0]
#print(cmx_mf_rdf,'\n')
#print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_rdf[1,1])
#print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_rdf[0,0])
#print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_rdf[0,1])
#print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_rdf[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
#precision_mf_rdf_micro = precision_score(y_test_mf, mf_predictions_rdf_test, average='micro')
#precision_mf_rdf_weighted = precision_score(y_test_mf, mf_predictions_rdf_test, average='weighted')
#print('Precision Rate - Micro :',round(precision_mf_rdf_micro,3))
#print('Precision Rate - Weighted :',round(precision_mf_rdf_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
#recall_mf_rdf_micro = recall_score(y_test_mf, mf_predictions_rdf_test, average='micro')
#recall_mf_rdf_weighted = recall_score(y_test_mf, mf_predictions_rdf_test, average='weighted')
#print('Recall Rate - Micro :',round(recall_mf_rdf_micro,3))
#print('Recall Rate - Weighted :',round(recall_mf_rdf_weighted,3),'\n\n')

#print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### NAIVE BAYES

# In[782]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#feature scaling
from sklearn.preprocessing import MinMaxScaler
mmsc = MinMaxScaler()
x_train_cc_nby = mmsc.fit_transform(x_train_cc)
x_test_cc_nby = mmsc.transform(x_test_cc)

#Plug the model to the training set
mnb = MultinomialNB()
cc_model_nby = mnb.fit(x_train_cc_nby, y_train_cc)#.values.ravel()

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_nby_cc = cross_val_score(cc_model_nby, x_train_cc_nby, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the NAIVE BAYERS model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_nby_cc.mean(), scores_cross_val_nby_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_nby_test = cc_model_nby.predict(x_test_cc_nby)
cc_acc_nby = round(accuracy_score(y_test_cc,cc_predictions_nby_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_nby)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_nby_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_nby_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_nby_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_nby = confusion_matrix(y_test_cc,cc_predictions_nby_test)
TP_cc_nby = cmx_cc_nby[1,1]
TN_cc_nby = cmx_cc_nby[0,0]
print(cmx_cc_nby,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_nby[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_nby[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_nby[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_nby[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_nby_micro = precision_score(y_test_cc, cc_predictions_nby_test, average='micro')
precision_cc_nby_weighted = precision_score(y_test_cc, cc_predictions_nby_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_nby_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_nby_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_nby_micro = recall_score(y_test_cc, cc_predictions_nby_test, average='micro')
recall_cc_nby_weighted = recall_score(y_test_cc, cc_predictions_nby_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_nby_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_nby_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### XGB Classifier

# In[783]:


#!pip install xgboost
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
xgb_clf = XGBClassifier()

#Plug the model to the training set
cc_model_xgb = xgb_clf.fit(x_train_cc, y_train_cc)#.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_xgb_cc = cross_val_score(cc_model_xgb, x_train_cc, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the XGB model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_xgb_cc.mean(), scores_cross_val_xgb_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_xgb_test = cc_model_xgb.predict(x_test_cc)
cc_acc_xgb = round(accuracy_score(y_test_cc,cc_predictions_xgb_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_xgb)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_xgb_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_xgb_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_xgb_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_xgb = confusion_matrix(y_test_cc,cc_predictions_xgb_test)
TP_cc_xgb = cmx_cc_xgb[1,1]
TN_cc_xgb = cmx_cc_xgb[0,0]
print(cmx_cc_xgb,'\n')
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_xgb[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_xgb[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_xgb[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_xgb[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_xgb_micro = precision_score(y_test_cc, cc_predictions_xgb_test, average='micro')
precision_cc_xgb_weighted = precision_score(y_test_cc, cc_predictions_xgb_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_xgb_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_xgb_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_xgb_micro = recall_score(y_test_cc, cc_predictions_xgb_test, average='micro')
recall_cc_xgb_weighted = recall_score(y_test_cc, cc_predictions_xgb_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_xgb_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_xgb_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVC

# In[784]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',max_iter=500000)

#Plug the model to the training set
cc_model_svc = svclassifier.fit(x_train_cc, y_train_cc)

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_svc_cc = cross_val_score(cc_model_svc, x_train_cc, y_train_cc, cv=5, scoring='accuracy')
print("CROSS VALIDATION\nAccuracy Score of the XGB model on Sale_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_svc_cc.mean(), scores_cross_val_svc_cc.std() * 2),'\n')

#Evaluate the model on the test set
cc_predictions_svc_test = cc_model_svc.predict(x_test_cc)
cc_acc_svc = round(accuracy_score(y_test_cc,cc_predictions_svc_test),3)
print("TEST SET\nAccuracy Score :", cc_acc_svc)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc, cc_predictions_svc_test),3))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc, cc_predictions_svc_test),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc, cc_predictions_svc_test)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_svc = confusion_matrix(y_test_cc,cc_predictions_svc_test)
print(cmx_cc_svc,'\n')
TP_cc_svc = cmx_cc_svc[1,1]
TN_cc_svc = cmx_cc_svc[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_svc[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_svc[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_svc[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_svc[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_svc_micro = precision_score(y_test_cc, cc_predictions_svc_test, average='micro')
precision_cc_svc_weighted = precision_score(y_test_cc, cc_predictions_svc_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_svc_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_svc_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_svc_micro = recall_score(y_test_cc, cc_predictions_svc_test, average='micro')
recall_cc_svc_weighted = recall_score(y_test_cc, cc_predictions_svc_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_svc_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_svc_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# #### 3. RESULT SALE_CC  :  

# In[786]:


print('CROSS VALIDATION ACCURACY (SALE_CC) :\n')
print('- DECISION TREE :            %0.2f (+/- %0.2f)' % (scores_cross_val_dt_cc.mean(), scores_cross_val_dt_cc.std() * 2))
print('- KNN :                      %0.2f (+/- %0.2f)' % (scores_cross_val_knn_cc.mean(), scores_cross_val_knn_cc.std() * 2))
print('- Logistic Regression :      %0.2f (+/- %0.2f)' % (scores_cross_val_lr_cc.mean(), scores_cross_val_lr_cc.std() * 2))
print('- SVM :                      %0.2f (+/- %0.2f)' % (scores_cross_val_svm_cc.mean(), scores_cross_val_svm_cc.std() * 2))
print('- LDA :                      %0.2f (+/- %0.2f)' % (scores_cross_val_lda_cc.mean(), scores_cross_val_lda_cc.std() * 2))
print('- Random Forest (not sure)  %0.2f (+/- %0.2f)' % (scores_cross_val_rdf_cc.mean(), scores_cross_val_rdf_cc.std() * 2))
print('- Naive Bayers :             %0.2f (+/- %0.2f)' % (scores_cross_val_nby_cc.mean(), scores_cross_val_nby_cc.std() * 2))
print('- XGB :                      %0.2f (+/- %0.2f)' % (scores_cross_val_xgb_cc.mean(), scores_cross_val_xgb_cc.std() * 2))
print('- SVC :                      %0.2f (+/- %0.2f)' % (scores_cross_val_xgb_cc.mean(), scores_cross_val_xgb_cc.std() * 2))


# In[787]:


print('CONFUSION MATRIX POST TEST ANALYSIS (SALE_CC) :\n')
print('DECISION TREES')
print('True Positive :',TP_cc_dt,' ------------> TARGET')
print('True Negative :',TN_cc_dt)
#print('False Positive :',FP_cc_dt)
#print('False Negative :',FN_cc_dt)
print('Precision - micro : ',round(precision_cc_dt_micro,2),'\nPrecision - weighted :',round(precision_cc_dt_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_dt_micro,2),'\nRecall - weighted :',round(recall_cc_dt_weighted,2),' -----> TARGET\n')

print('KNN')
print('True Positive :',TP_cc_knn,' ------------> TARGET')
print('True Negative :',TN_cc_knn)
print('Precision - micro : ',round(precision_cc_knn_micro,2),'\nPrecision - weighted :',round(precision_cc_knn_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_knn_micro,2),'\nRecall - weighted :',round(recall_cc_knn_weighted,2),' -----> TARGET\n')

print('LOGISTIC REGRESSION')
print('True Positive :',TP_cc_lr,' ------------> TARGET')
print('True Negative :',TN_cc_lr)
print('Precision - micro : ',round(precision_cc_lr_micro,2),'\nPrecision - weighted :',round(precision_cc_lr_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_lr_micro,2),'\nRecall - weighted :',round(recall_cc_lr_weighted,2),' -----> TARGET\n')

print('SVM')
print('True Positive :',TP_cc_svm,' ------------> TARGET')
print('True Negative :',TN_cc_svm)
print('Precision - micro : ',round(precision_cc_svm_micro,2),'\nPrecision - weighted :',round(precision_cc_svm_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_svm_micro,2),'\nRecall - weighted :',round(recall_cc_svm_weighted,2),' -----> TARGET\n')

print('LDA')
print('True Positive :',TP_cc_lda,' ------------> TARGET')
print('True Negative :',TN_cc_lda)
#print('False Positive :',FP_cc_lda)
#print('False Negative :',FN_cc_lda)
print('Precision - micro : ',round(precision_cc_lda_micro,2),'\nPrecision - weighted :',round(precision_cc_lda_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_lda_micro,2),'\nRecall - weighted :',round(recall_cc_lda_weighted,2),' -----> TARGET\n')

print('RANDOM FOREST')
print('non available\n')

print('NAIVE BAYES')
print('True Positive :',TP_cc_nby,' ------------> TARGET')
print('True Negative :',TN_cc_nby)
print('Precision - micro : ',round(precision_cc_nby_micro,2),'\nPrecision - weighted :',round(precision_cc_nby_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_nby_micro,2),'\nRecall - weighted :',round(recall_cc_nby_weighted,2),' -----> TARGET\n')

print('XGB')
print('True Positive :',TP_cc_xgb,' ------------> TARGET')
print('True Negative :',TN_cc_xgb)
print('Precision - micro : ',round(precision_cc_xgb_micro,2),'\nPrecision - weighted :',round(precision_cc_xgb_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_xgb_micro,2),'\nRecall - weighted :',round(recall_cc_xgb_weighted,2),' -----> TARGET\n')

print('SVC')
print('True Positive :',TP_cc_svc,' ------------> TARGET')
print('True Negative :',TN_cc_svc)
print('Precision - micro : ',round(precision_cc_svc_micro,2),'\nPrecision - weighted :',round(precision_cc_svc_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_svc_micro,2),'\nRecall - weighted :',round(recall_cc_svc_weighted,2),' -----> TARGET\n')


# ### SUMMARY :
#     1. Sale_MF : test with Logistic regression and LDA
#     2. Sale_CL : test with Logistic regression and LDA
#     3. Sale_CC : test with Logistic regression and SVM
# 

# # 4. MODELS OPTIMIZATION :
# Exploration over the two most promising models for each Sale objective

# ## MODEL SALE_MF : PARAMETERS AND DEPLOYMENT

# ##### SELECTION OF TARGET METRIC + EXCLUSION OF METRICS

# In[788]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_mf = df2[['Sale_MF']]


# ##### SPLIT OF THE DATASET INTO TRAIN / CROSS VALIDATION / TEST

# In[789]:


#import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_mf_deploy, x_test_mf_deploy, y_train_mf_deploy,y_test_mf_deploy = train_test_split(x,y_mf,test_size=0.2,random_state=42)
x_train_mf_deploy, x_cv_mf_deploy, y_train_mf_deploy, y_cv_mf_deploy = train_test_split(x_train_mf_deploy,y_train_mf_deploy,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_mf_deploy.shape)
print('Training Labels Shape:', y_train_mf_deploy.shape)
print('Cross Validation Testing Features Shape:', x_cv_mf_deploy.shape)
print('Cross Validation Testing Labels Shape:', y_cv_mf_deploy.shape)
print('Testing Features Shape:', x_test_mf_deploy.shape)
print('Testing Labels Shape:', y_test_mf_deploy.shape,'\n')

print('Right Split with an overall matching number of observations :',x_train_mf_deploy.shape[0] + x_cv_mf_deploy.shape[0] + x_test_mf_deploy.shape[0] == x.shape[0])
print('Right Split with an overall matching number of variables :', (x_train_mf_deploy.shape[1] == x_cv_mf_deploy.shape[1] == x_test_mf_deploy.shape[1]) == y_mf.shape[1])


# ### LOGISTIC REGRESSION

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[790]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logmodel = LogisticRegression(max_iter=1000)

#Plug the model to the training set
mf_model_lr_deploy_train = logmodel.fit(x_train_mf_deploy,y_train_mf_deploy.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
mf_predictions_lr_test_deploy = mf_model_lr_deploy_train.predict(x_test_mf_deploy)
from sklearn.model_selection import cross_val_score
mf_acc_lr_deploy = round(accuracy_score(y_test_mf_deploy,mf_predictions_lr_test_deploy),3)
print("TEST SET\nAccuracy Score :", mf_acc_lr_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_deploy, mf_predictions_lr_test_deploy),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lr_weighted_deploy = precision_score(y_test_mf_deploy, mf_predictions_lr_test_deploy, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lr_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lr_weighted_deploy = recall_score(y_test_mf_deploy, mf_predictions_lr_test_deploy, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lr_weighted_deploy,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lr = confusion_matrix(y_test_mf_deploy, mf_predictions_lr_test_deploy)
print(cmx_mf_lr,'\n')
TP_mf_lr = cmx_mf_lr[1,1]
TN_mf_lr = cmx_mf_lr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lr[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lr[1,1]+cmx_mf_lr[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_deploy['Sale_MF'], mf_predictions_lr_test_deploy)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_deploy['Sale_MF'],mf_predictions_lr_test_deploy),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[791]:


logmodel = LogisticRegression(max_iter=1000)

#Plug the model to the cross validation set
mf_model_lr_deploy_cv = logmodel.fit(x_cv_mf_deploy,y_cv_mf_deploy.values.ravel())
print('------ CROSS VALIDATION ------\n')

#Evaluate the model on the test set
mf_predictions_lr_test_deploy_cv = mf_model_lr_deploy_cv.predict(x_test_mf_deploy)

from sklearn.model_selection import cross_val_score
mf_acc_lr_deploy = round(accuracy_score(y_test_mf_deploy,mf_predictions_lr_test_deploy_cv),3)
print("TEST SET\nAccuracy Score :", mf_acc_lr_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_deploy, mf_predictions_lr_test_deploy_cv),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lr_weighted_deploy = precision_score(y_test_mf_deploy, mf_predictions_lr_test_deploy_cv, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lr_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lr_weighted_deploy = recall_score(y_test_mf_deploy, mf_predictions_lr_test_deploy_cv, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lr_weighted_deploy,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lr = confusion_matrix(y_test_mf_deploy, mf_predictions_lr_test_deploy_cv)
print(cmx_mf_lr,'\n')
TP_mf_lr = cmx_mf_lr[1,1]
TN_mf_lr = cmx_mf_lr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lr[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lr[1,1]+cmx_mf_lr[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_deploy['Sale_MF'], mf_predictions_lr_test_deploy_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_deploy['Sale_MF'],mf_predictions_lr_test_deploy_cv),3))
plt.show()


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[792]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


# In[793]:


#After few trials, 19 is the number optimizing the best our KPI
selector = SelectKBest(chi2, k=19)
x_train_mf_new_params = selector.fit_transform(x_train_mf_deploy,y_train_mf_deploy)
x_train_mf_new_params.shape


# In[794]:


x_test_mf_new_params = selector.fit_transform(x_test_mf_deploy,y_test_mf_deploy)
x_test_mf_new_params.shape


# In[795]:


x_cv_mf_new_params = selector.fit_transform(x_cv_mf_deploy,y_cv_mf_deploy)
x_cv_mf_new_params.shape


# In[796]:


#for visualising which columns have been selected
cols = selector.get_support(indices=True)
features_df_new = x_train_mf_deploy.iloc[:,cols]
features_df_new.columns


# ###### TEST ON THE TRAINING DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[797]:


logmodel = LogisticRegression(max_iter=1000)

y_train_mf_new_params = y_train_mf_deploy
y_test_mf_new_params = y_test_mf_deploy

#Plug the model to the training set
mf_model_lr_optimized = logmodel.fit(x_train_mf_new_params,y_train_mf_new_params.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
mf_predictions_lr_optimized = mf_model_lr_optimized.predict(x_test_mf_new_params)
from sklearn.model_selection import cross_val_score

mf_acc_lr_optimized = round(accuracy_score(y_test_mf_new_params,mf_predictions_lr_optimized),3)
print("TEST SET\nAccuracy Score :", mf_acc_lr_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_new_params,mf_predictions_lr_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lr_weighted_optimized = precision_score(y_test_mf_new_params,mf_predictions_lr_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lr_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lr_weighted_optimized = recall_score(y_test_mf_new_params,mf_predictions_lr_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lr_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lr_op = confusion_matrix(y_test_mf_new_params,mf_predictions_lr_optimized)
print(cmx_mf_lr_op,'\n')
TP_mf_lr_op = cmx_mf_lr_op[1,1]
TN_mf_lr_op = cmx_mf_lr_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lr_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lr_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lr_op[1,1]+cmx_mf_lr_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_new_params['Sale_MF'], mf_predictions_lr_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_new_params['Sale_MF'],mf_predictions_lr_optimized),3))
plt.show()


# ###### TEST ON THE CROSS VALIDATION DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[798]:


logmodel = LogisticRegression(max_iter=1000)

y_cv_mf_new_params = y_cv_mf_deploy
y_test_mf_new_params = y_test_mf_deploy

#Plug the model to thecross validation set
mf_model_lr_optimized = logmodel.fit(x_cv_mf_new_params,y_cv_mf_new_params.values.ravel())
print('------ CROSS VALIDATION SET ------\n')

#Evaluate the model on the test set
mf_predictions_lr_optimized = mf_model_lr_optimized.predict(x_test_mf_new_params)
from sklearn.model_selection import cross_val_score

mf_acc_lr_optimized = round(accuracy_score(y_test_mf_new_params,mf_predictions_lr_optimized),3)
print("TEST SET\nAccuracy Score :", mf_acc_lr_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_new_params,mf_predictions_lr_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lr_weighted_optimized = precision_score(y_test_mf_new_params,mf_predictions_lr_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lr_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lr_weighted_optimized = recall_score(y_test_mf_new_params,mf_predictions_lr_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lr_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lr_op = confusion_matrix(y_test_mf_new_params,mf_predictions_lr_optimized)
print(cmx_mf_lr_op,'\n')
TP_mf_lr_op = cmx_mf_lr_op[1,1]
TN_mf_lr_op = cmx_mf_lr_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lr_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lr_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lr_op[1,1]+cmx_mf_lr_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_new_params['Sale_MF'], mf_predictions_lr_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_new_params['Sale_MF'],mf_predictions_lr_optimized),3))
plt.show()


# ### LDA

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[799]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_mf_deploy = sc.fit_transform(x_train_mf_deploy)
x_test_mf_deploy = sc.fit_transform(x_test_mf_deploy)

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_train_mf_deploy = lda.fit_transform(x_train_mf_deploy, y_train_mf_deploy)
x_test_mf_deploy = lda.transform(x_test_mf_deploy)

from sklearn.ensemble import RandomForestClassifier
mf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
mf_classifier = mf_classifier.fit(x_train_mf_deploy, y_train_mf_deploy)
mf_model_lda_deploy_train = mf_classifier.fit(x_train_mf_deploy,y_train_mf_deploy.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
mf_predictions_lda_test_deploy = mf_model_lda_deploy_train.predict(x_test_mf_deploy)
from sklearn.model_selection import cross_val_score
mf_acc_lda_deploy = round(accuracy_score(y_test_mf_deploy,mf_predictions_lda_test_deploy),3)
print("TEST SET\nAccuracy Score :", mf_acc_lda_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_deploy, mf_predictions_lda_test_deploy),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lda_weighted_deploy = precision_score(y_test_mf_deploy, mf_predictions_lda_test_deploy, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lda_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lda_weighted_deploy = recall_score(y_test_mf_deploy, mf_predictions_lda_test_deploy, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lda_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_deploy['Sale_MF'], mf_predictions_lda_test_deploy)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_deploy['Sale_MF'],mf_predictions_lda_test_deploy),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[800]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_cv_mf_deploy = sc.fit_transform(x_cv_mf_deploy)
x_test_mf_deploy = sc.fit_transform(x_test_mf_deploy)

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_cv_mf_deploy = lda.fit_transform(x_cv_mf_deploy, y_cv_mf_deploy)
x_test_mf_deploy = lda.transform(x_test_mf_deploy)

from sklearn.ensemble import RandomForestClassifier
mf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
mf_classifier = mf_classifier.fit(x_cv_mf_deploy, y_cv_mf_deploy)
mf_model_lda_deploy_train_cv = mf_classifier.fit(x_cv_mf_deploy,y_cv_mf_deploy.values.ravel())
print('------ CROSS VALIDATION ------\n')

#Evaluate the model on the test set
mf_predictions_lda_test_deploy_cv = mf_model_lda_deploy_train_cv.predict(x_test_mf_deploy)
from sklearn.model_selection import cross_val_score
mf_acc_lda_deploy = round(accuracy_score(y_test_mf_deploy,mf_predictions_lda_test_deploy_cv),3)
print("TEST SET\nAccuracy Score :", mf_acc_lda_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_deploy, mf_predictions_lda_test_deploy_cv),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lda_weighted_deploy = precision_score(y_test_mf_deploy, mf_predictions_lda_test_deploy_cv, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lda_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lda_weighted_deploy = recall_score(y_test_mf_deploy, mf_predictions_lda_test_deploy_cv, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lda_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_deploy['Sale_MF'], mf_predictions_lda_test_deploy_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_deploy['Sale_MF'],mf_predictions_lda_test_deploy_cv),3))
plt.show()


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[801]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


# In[802]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
x = df2[features]
y_mf = df2[['Sale_MF']]
x_train_mf_deploy, x_test_mf_deploy, y_train_mf_deploy,y_test_mf_deploy = train_test_split(x,y_mf,test_size=0.2,random_state=42)
x_train_mf_deploy, x_cv_mf_deploy, y_train_mf_deploy, y_cv_mf_deploy = train_test_split(x_train_mf_deploy,y_train_mf_deploy,test_size=0.2,random_state=42)


#After few trials, 21 is the number optimizing the best our KPI
selector = SelectKBest(chi2, k=21)
x_train_mf_new_params_lda = selector.fit_transform(x_train_mf_deploy,y_train_mf_deploy)
x_train_mf_new_params_lda.shape


# In[803]:


x_test_mf_new_params_lda = selector.fit_transform(x_test_mf_deploy,y_test_mf_deploy)
x_test_mf_new_params_lda.shape


# In[804]:


x_cv_mf_new_params_lda = selector.fit_transform(x_cv_mf_deploy,y_cv_mf_deploy)
x_cv_mf_new_params_lda.shape


# In[805]:


#for visualising which columns have been selected
cols = selector.get_support(indices=True)
features_df_new = x_train_mf_deploy.iloc[:,cols]
features_df_new.columns


# ###### TEST ON THE TRAINING DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[806]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_mf_deploy = sc.fit_transform(x_train_mf_deploy)
x_test_mf_deploy = sc.fit_transform(x_test_mf_deploy)

y_train_mf_new_params_lda = y_train_mf_deploy
y_test_mf_new_params_lda = y_test_mf_deploy

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_train_mf_new_params_lda = lda.fit_transform(x_train_mf_new_params_lda, y_train_mf_new_params_lda)
x_test_mf_new_params_lda = lda.transform(x_test_mf_new_params_lda)

from sklearn.ensemble import RandomForestClassifier
mf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
mf_classifier = mf_classifier.fit(x_train_mf_new_params_lda, y_train_mf_new_params_lda)
mf_model_lda_optimized = mf_classifier.fit(x_train_mf_new_params_lda,y_train_mf_new_params_lda.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
mf_predictions_lda_optimized = mf_model_lda_optimized.predict(x_test_mf_new_params_lda)
from sklearn.model_selection import cross_val_score
mf_acc_lda_optimized = round(accuracy_score(y_test_mf_new_params_lda,mf_predictions_lda_optimized),3)
print("TEST SET\nAccuracy Score :", mf_acc_lda_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_new_params_lda,mf_predictions_lda_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lda_weighted_optimized = precision_score(y_test_mf_new_params_lda,mf_predictions_lda_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lda_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lda_weighted_optimized = recall_score(y_test_mf_new_params_lda,mf_predictions_lda_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lda_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lda_op = confusion_matrix(y_test_mf_new_params_lda,mf_predictions_lda_optimized)
print(cmx_mf_lda_op,'\n')
TP_mf_lda_op = cmx_mf_lda_op[1,1]
TN_mf_lda_op = cmx_mf_lda_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lda_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lda_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lda_op[1,1]+cmx_mf_lda_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_new_params_lda['Sale_MF'], mf_predictions_lda_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_new_params_lda['Sale_MF'],mf_predictions_lda_optimized),3))
plt.show()


# ###### TEST ON THE CROSS VALIDATION DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[807]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_cv_mf_deploy = sc.fit_transform(x_cv_mf_deploy)
x_test_mf_deploy = sc.fit_transform(x_test_mf_deploy)

y_cv_mf_new_params_lda = y_cv_mf_deploy
y_test_mf_new_params_lda = y_test_mf_deploy

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_cv_mf_new_params_lda = lda.fit_transform(x_cv_mf_new_params_lda, y_cv_mf_new_params_lda)
x_test_mf_new_params_lda = lda.transform(x_test_mf_new_params_lda)

from sklearn.ensemble import RandomForestClassifier
mf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
mf_classifier = mf_classifier.fit(x_cv_mf_new_params_lda, y_cv_mf_new_params_lda)
mf_model_lda_optimized = mf_classifier.fit(x_cv_mf_new_params_lda,y_cv_mf_new_params_lda.values.ravel())
print('------ CROSS VALIDATION SET ------\n')

#Evaluate the model on the test set
mf_predictions_lda_optimized = mf_model_lda_optimized.predict(x_test_mf_new_params_lda)
from sklearn.model_selection import cross_val_score

mf_acc_lda_optimized = round(accuracy_score(y_test_mf_new_params_lda,mf_predictions_lda_optimized),3)
print("TEST SET\nAccuracy Score :", mf_acc_lda_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_mf_new_params_lda,mf_predictions_lda_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_mf_lda_weighted_optimized = precision_score(y_test_mf_new_params_lda,mf_predictions_lda_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_mf_lda_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_mf_lda_weighted_optimized = recall_score(y_test_mf_new_params_lda,mf_predictions_lda_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_mf_lda_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lda_op = confusion_matrix(y_test_mf_new_params_lda,mf_predictions_lda_optimized)
print(cmx_mf_lda_op,'\n')
TP_mf_lda_op = cmx_mf_lda_op[1,1]
TN_mf_lda_op = cmx_mf_lda_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lda_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lda_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lda_op[1,1]+cmx_mf_lda_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_mf_new_params_lda['Sale_MF'], mf_predictions_lda_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_mf_new_params_lda['Sale_MF'],mf_predictions_lda_optimized),3))
plt.show()


# ##### ELECTING THE FINAL MODEL 

# In[808]:


print('MODEL 1 LOGISTIC REGRESSION -----------------')
print('Precision Rate - Weighted :',round(precision_mf_lr_weighted_optimized,3))
print('Recall Rate - Weighted :',round(recall_mf_lr_weighted_optimized,3))
print('AUC :', round(roc_auc_score(y_test_mf_new_params['Sale_MF'],mf_predictions_lr_optimized),3),'\n')
print('TOTAL TRUE :',cmx_mf_lr_op[1,1]+cmx_mf_lr_op[0,0])
print(cmx_mf_lr_op,'\n')

print('MODEL 2 LDA ---------------------------------')
print('Precision Rate - Weighted :',round(precision_mf_lda_weighted_optimized,3))
print('Recall Rate - Weighted :',round(recall_mf_lda_weighted_optimized,3))
print('AUC :', round(roc_auc_score(y_test_mf_new_params_lda['Sale_MF'],mf_predictions_lda_optimized),3),'\n')
print('TOTAL TRUE :',cmx_mf_lda_op[1,1]+cmx_mf_lda_op[0,0])
print(cmx_mf_lda_op)


# ######## BASED ON THE RESULTS : the chosen model for projecting SALE_CC is LDA

# In[ ]:





# ## MODEL SALE_CL : PARAMETERS AND DEPLOYMENT

# ##### SELECTION OF TARGET METRIC + EXCLUSION OF METRICS

# In[809]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_cl = df2[['Sale_CL']]


# ##### SPLIT OF THE DATASET INTO TRAIN / CROSS VALIDATION / TEST

# In[810]:


#import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cl_deploy, x_test_cl_deploy, y_train_cl_deploy,y_test_cl_deploy = train_test_split(x,y_cl,test_size=0.2,random_state=42)
x_train_cl_deploy, x_cv_cl_deploy, y_train_cl_deploy, y_cv_cl_deploy = train_test_split(x_train_cl_deploy,y_train_cl_deploy,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_cl_deploy.shape)
print('Training Labels Shape:', y_train_cl_deploy.shape)
print('Cross Validation Testing Features Shape:', x_cv_cl_deploy.shape)
print('Cross Validation Testing Labels Shape:', y_cv_cl_deploy.shape)
print('Testing Features Shape:', x_test_cl_deploy.shape)
print('Testing Labels Shape:', y_test_cl_deploy.shape,'\n')

print('Right Split with an overall matching number of observations :',x_train_cl_deploy.shape[0] + x_cv_cl_deploy.shape[0] + x_test_cl_deploy.shape[0] == x.shape[0])
print('Right Split with an overall matching number of variables :', (x_train_cl_deploy.shape[1] == x_cv_cl_deploy.shape[1] == x_test_cl_deploy.shape[1]) == y_cl.shape[1])


# ### LOGISTIC REGRESSION

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[811]:


logmodel = LogisticRegression(max_iter=1000)

#Plug the model to the training set
cl_model_lr_deploy_train = logmodel.fit(x_train_cl_deploy,y_train_cl_deploy.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cl_predictions_lr_test_deploy = cl_model_lr_deploy_train.predict(x_test_cl_deploy)
from sklearn.model_selection import cross_val_score
cl_acc_lr_deploy = round(accuracy_score(y_test_cl_deploy,cl_predictions_lr_test_deploy),3)
print("TEST SET\nAccuracy Score :", cl_acc_lr_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_deploy, cl_predictions_lr_test_deploy),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lr_weighted_deploy = precision_score(y_test_cl_deploy, cl_predictions_lr_test_deploy, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lr_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lr_weighted_deploy = recall_score(y_test_cl_deploy, cl_predictions_lr_test_deploy, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lr_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_deploy['Sale_CL'], cl_predictions_lr_test_deploy)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_deploy['Sale_CL'],cl_predictions_lr_test_deploy),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[812]:


logmodel = LogisticRegression(max_iter=1000)

#Plug the model to the cross validation set
cl_model_lr_deploy_cv = logmodel.fit(x_cv_cl_deploy,y_cv_cl_deploy.values.ravel())
print('------ CROSS VALIDATION ------\n')

#Evaluate the model on the test set
cl_predictions_lr_test_deploy_cv = cl_model_lr_deploy_cv.predict(x_test_cl_deploy)

from sklearn.model_selection import cross_val_score
cl_acc_lr_deploy = round(accuracy_score(y_test_cl_deploy,cl_predictions_lr_test_deploy_cv),3)
print("TEST SET\nAccuracy Score :", cl_acc_lr_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_deploy, cl_predictions_lr_test_deploy_cv),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lr_weighted_deploy = precision_score(y_test_cl_deploy, cl_predictions_lr_test_deploy_cv, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lr_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lr_weighted_deploy = recall_score(y_test_cl_deploy, cl_predictions_lr_test_deploy_cv, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lr_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_deploy['Sale_CL'], cl_predictions_lr_test_deploy_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_deploy['Sale_CL'],cl_predictions_lr_test_deploy_cv),3))
plt.show()


# ###### TEST ON THE TRAINING DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[813]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


# In[814]:


#After few trials, 14 is the number optimizing the best our KPI
selector = SelectKBest(chi2, k=14)
x_train_cl_new_params = selector.fit_transform(x_train_cl_deploy,y_train_cl_deploy)
x_train_cl_new_params.shape


# In[815]:


x_test_cl_new_params = selector.fit_transform(x_test_cl_deploy,y_test_cl_deploy)
x_test_cl_new_params.shape


# In[816]:


x_cv_cl_new_params = selector.fit_transform(x_cv_cl_deploy,y_cv_cl_deploy)
x_cv_cl_new_params.shape


# In[817]:


#for visualising which columns have been selected
cols = selector.get_support(indices=True)
features_df_new = x_train_cl_deploy.iloc[:,cols]
features_df_new.columns


# In[1142]:


logmodel = LogisticRegression(max_iter=1000)

y_train_cl_new_params = y_train_cl_deploy
y_test_cl_new_params = y_test_cl_deploy

#Plug the model to the training set
cl_model_lr_optimized = logmodel.fit(x_train_cl_new_params,y_train_cl_new_params.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cl_predictions_lr_optimized = cl_model_lr_optimized.predict(x_test_cl_new_params)
from sklearn.model_selection import cross_val_score

cl_acc_lr_optimized = round(accuracy_score(y_test_cl_new_params,cl_predictions_lr_optimized),3)
print("TEST SET\nAccuracy Score :", cl_acc_lr_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_new_params,cl_predictions_lr_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lr_weighted_optimized = precision_score(y_test_cl_new_params,cl_predictions_lr_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lr_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lr_weighted_optimized = recall_score(y_test_cl_new_params,cl_predictions_lr_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lr_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lr_op = confusion_matrix(y_test_cl_new_params,cl_predictions_lr_optimized)
print(cmx_cl_lr_op,'\n')
TP_cl_lr_op = cmx_cl_lr_op[1,1]
TN_cl_lr_op = cmx_cl_lr_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lr_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lr_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_lr_op[1,1]+cmx_cl_lr_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_new_params['Sale_CL'], cl_predictions_lr_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_new_params['Sale_CL'],cl_predictions_lr_optimized),3))
plt.show()


# ###### TEST ON THE CROSS VALIDATION DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[819]:


logmodel = LogisticRegression(max_iter=1000)

y_cv_cl_new_params = y_cv_cl_deploy
y_test_cl_new_params = y_test_cl_deploy

#Plug the model to thecross validation set
cl_model_lr_optimized = logmodel.fit(x_cv_cl_new_params,y_cv_cl_new_params.values.ravel())
print('------ CROSS VALIDATION SET ------\n')

#Evaluate the model on the test set
cl_predictions_lr_optimized = cl_model_lr_optimized.predict(x_test_cl_new_params)
from sklearn.model_selection import cross_val_score

cl_acc_lr_optimized = round(accuracy_score(y_test_cl_new_params,cl_predictions_lr_optimized),3)
print("TEST SET\nAccuracy Score :", cl_acc_lr_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_new_params,cl_predictions_lr_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lr_weighted_optimized = precision_score(y_test_cl_new_params,cl_predictions_lr_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lr_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lr_weighted_optimized = recall_score(y_test_cl_new_params,cl_predictions_lr_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lr_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lr_op = confusion_matrix(y_test_cl_new_params,cl_predictions_lr_optimized)
print(cmx_cl_lr_op,'\n')
TP_cl_lr_op = cmx_cl_lr_op[1,1]
TN_cl_lr_op = cmx_cl_lr_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lr_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lr_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_lr_op[1,1]+cmx_cl_lr_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_new_params['Sale_CL'], cl_predictions_lr_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_new_params['Sale_CL'],cl_predictions_lr_optimized),3))
plt.show()


# ### LDA

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[820]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_cl_deploy = sc.fit_transform(x_train_cl_deploy)
x_test_cl_deploy = sc.fit_transform(x_test_cl_deploy)

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_train_cl_deploy = lda.fit_transform(x_train_cl_deploy, y_train_cl_deploy)
x_test_cl_deploy = lda.transform(x_test_cl_deploy)

from sklearn.ensemble import RandomForestClassifier
cl_classifier = RandomForestClassifier(max_depth=2, random_state=0)
cl_classifier = cl_classifier.fit(x_train_cl_deploy, y_train_cl_deploy)
cl_model_lda_deploy_train = cl_classifier.fit(x_train_cl_deploy,y_train_cl_deploy.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cl_predictions_lda_test_deploy = cl_model_lda_deploy_train.predict(x_test_cl_deploy)
from sklearn.model_selection import cross_val_score
cl_acc_lda_deploy = round(accuracy_score(y_test_cl_deploy,cl_predictions_lda_test_deploy),3)
print("TEST SET\nAccuracy Score :", cl_acc_lda_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_deploy, cl_predictions_lda_test_deploy),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lda_weighted_deploy = precision_score(y_test_cl_deploy, cl_predictions_lda_test_deploy, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lda_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lda_weighted_deploy = recall_score(y_test_cl_deploy, cl_predictions_lda_test_deploy, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lda_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_deploy['Sale_CL'], cl_predictions_lda_test_deploy)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_deploy['Sale_CL'],cl_predictions_lda_test_deploy),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[821]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_cv_cl_deploy = sc.fit_transform(x_cv_cl_deploy)
x_test_cl_deploy = sc.fit_transform(x_test_cl_deploy)

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_cv_cl_deploy = lda.fit_transform(x_cv_cl_deploy, y_cv_cl_deploy)
x_test_cl_deploy = lda.transform(x_test_cl_deploy)

from sklearn.ensemble import RandomForestClassifier
cl_classifier = RandomForestClassifier(max_depth=2, random_state=0)
cl_classifier = cl_classifier.fit(x_cv_cl_deploy, y_cv_cl_deploy)
cl_model_lda_deploy_train_cv = cl_classifier.fit(x_cv_cl_deploy,y_cv_cl_deploy.values.ravel())
print('------ CROSS VALIDATION ------\n')

#Evaluate the model on the test set
cl_predictions_lda_test_deploy_cv = cl_model_lda_deploy_train_cv.predict(x_test_cl_deploy)
from sklearn.model_selection import cross_val_score
cl_acc_lda_deploy = round(accuracy_score(y_test_cl_deploy,cl_predictions_lda_test_deploy_cv),3)
print("TEST SET\nAccuracy Score :", cl_acc_lda_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_deploy, cl_predictions_lda_test_deploy_cv),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lda_weighted_deploy = precision_score(y_test_cl_deploy, cl_predictions_lda_test_deploy_cv, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lda_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lda_weighted_deploy = recall_score(y_test_cl_deploy, cl_predictions_lda_test_deploy_cv, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lda_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_deploy['Sale_CL'], cl_predictions_lda_test_deploy_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_deploy['Sale_CL'],cl_predictions_lda_test_deploy_cv),3))
plt.show()


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[822]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


# In[823]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
x = df2[features]
y_mf = df2[['Sale_CL']]
x_train_cl_deploy, x_test_cl_deploy, y_train_cl_deploy,y_test_cl_deploy = train_test_split(x,y_cl,test_size=0.2,random_state=42)
x_train_cl_deploy, x_cv_cl_deploy, y_train_cl_deploy, y_cv_cl_deploy = train_test_split(x_train_cl_deploy,y_train_cl_deploy,test_size=0.2,random_state=42)


#After few trials, 20 is the number optimizing the best our KPI
selector = SelectKBest(chi2, k=20)
x_train_cl_new_params_lda = selector.fit_transform(x_train_cl_deploy,y_train_cl_deploy)
x_train_cl_new_params_lda.shape


# In[824]:


x_test_cl_new_params_lda = selector.fit_transform(x_test_cl_deploy,y_test_cl_deploy)
x_test_cl_new_params_lda.shape


# In[825]:


x_cv_cl_new_params_lda = selector.fit_transform(x_cv_cl_deploy,y_cv_cl_deploy)
x_cv_cl_new_params_lda.shape


# In[826]:


#for visualising which columns have been selected
cols = selector.get_support(indices=True)
features_df_new = x_train_cl_deploy.iloc[:,cols]
features_df_new.columns


# ###### TEST ON THE TRAINING DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[827]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_cl_deploy = sc.fit_transform(x_train_cl_deploy)
x_test_cl_deploy = sc.fit_transform(x_test_cl_deploy)

y_train_cl_new_params_lda = y_train_cl_deploy
y_test_cl_new_params_lda = y_test_cl_deploy

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_train_cl_new_params_lda = lda.fit_transform(x_train_cl_new_params_lda, y_train_cl_new_params_lda)
x_test_cl_new_params_lda = lda.transform(x_test_cl_new_params_lda)

from sklearn.ensemble import RandomForestClassifier
cl_classifier = RandomForestClassifier(max_depth=2, random_state=0)
cl_classifier = cl_classifier.fit(x_train_cl_new_params_lda, y_train_cl_new_params_lda)
cl_model_lda_optimized = mf_classifier.fit(x_train_cl_new_params_lda,y_train_cl_new_params_lda.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cl_predictions_lda_optimized = cl_model_lda_optimized.predict(x_test_cl_new_params_lda)
from sklearn.model_selection import cross_val_score
cl_acc_lda_optimized = round(accuracy_score(y_test_cl_new_params_lda,cl_predictions_lda_optimized),3)
print("TEST SET\nAccuracy Score :", cl_acc_lda_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_new_params_lda,cl_predictions_lda_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lda_weighted_optimized = precision_score(y_test_cl_new_params_lda,cl_predictions_lda_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lda_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lda_weighted_optimized = recall_score(y_test_cl_new_params_lda,cl_predictions_lda_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lda_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lda_op = confusion_matrix(y_test_cl_new_params_lda,cl_predictions_lda_optimized)
print(cmx_cl_lda_op,'\n')
TP_cl_lda_op = cmx_cl_lda_op[1,1]
TN_cl_lda_op = cmx_cl_lda_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lda_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lda_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_lda_op[1,1]+cmx_cl_lda_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_new_params_lda['Sale_CL'], cl_predictions_lda_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_new_params_lda['Sale_CL'],cl_predictions_lda_optimized),3))
plt.show()


# ###### TEST ON THE CROSS VALIDATION DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[828]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_cv_cl_deploy = sc.fit_transform(x_cv_cl_deploy)
x_test_cl_deploy = sc.fit_transform(x_test_cl_deploy)

y_cv_cl_new_params_lda = y_cv_cl_deploy
y_test_cl_new_params_lda = y_test_cl_deploy

#definition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)

# Fit the classifier to the data
x_cv_cl_new_params_lda = lda.fit_transform(x_cv_cl_new_params_lda, y_cv_cl_new_params_lda)
x_test_cl_new_params_lda = lda.transform(x_test_cl_new_params_lda)

from sklearn.ensemble import RandomForestClassifier
cl_classifier = RandomForestClassifier(max_depth=2, random_state=0)
cl_classifier = cl_classifier.fit(x_cv_cl_new_params_lda, y_cv_cl_new_params_lda)
cl_model_lda_optimized = cl_classifier.fit(x_cv_cl_new_params_lda,y_cv_cl_new_params_lda.values.ravel())
print('------ CROSS VALIDATION SET ------\n')

#Evaluate the model on the test set
cl_predictions_lda_optimized = cl_model_lda_optimized.predict(x_test_cl_new_params_lda)
from sklearn.model_selection import cross_val_score

cl_acc_lda_optimized = round(accuracy_score(y_test_cl_new_params_lda,cl_predictions_lda_optimized),3)
print("TEST SET\nAccuracy Score :", cl_acc_lda_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cl_new_params_lda,cl_predictions_lda_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cl_lda_weighted_optimized = precision_score(y_test_cl_new_params_lda,cl_predictions_lda_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cl_lda_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cl_lda_weighted_optimized = recall_score(y_test_cl_new_params_lda,cl_predictions_lda_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cl_lda_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lda_op = confusion_matrix(y_test_cl_new_params_lda,cl_predictions_lda_optimized)
print(cmx_cl_lda_op,'\n')
TP_cl_lda_op = cmx_cl_lda_op[1,1]
TN_cl_lda_op = cmx_cl_lda_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lda_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lda_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_lda_op[1,1]+cmx_cl_lda_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cl_new_params_lda['Sale_CL'], cl_predictions_lda_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cl_new_params_lda['Sale_CL'],cl_predictions_lda_optimized),3))
plt.show()


# ##### ELECTING THE FINAL MODEL 

# In[829]:


print('MODEL 1 LOGISTIC REGRESSION -----------------')
print('Precision Rate - Weighted :',round(precision_cl_lr_weighted_optimized,3))
print('Recall Rate - Weighted :',round(recall_cl_lr_weighted_optimized,3))
print('AUC :', round(roc_auc_score(y_test_cl_new_params['Sale_CL'],cl_predictions_lr_optimized),3),'\n')
print('TOTAL TRUE :',cmx_cl_lr_op[1,1]+cmx_cl_lr_op[0,0])
print(cmx_cl_lr_op,'\n')

print('MODEL 2 LDA ---------------------------------')
print('Precision Rate - Weighted :',round(precision_cl_lda_weighted_optimized,3))
print('Recall Rate - Weighted :',round(recall_cl_lda_weighted_optimized,3))
print('AUC :', round(roc_auc_score(y_test_cl_new_params_lda['Sale_CL'],cl_predictions_lda_optimized),3),'\n')
print('TOTAL TRUE :',cmx_cl_lda_op[1,1]+cmx_cl_lda_op[0,0])
print(cmx_cl_lda_op)


# ######## BASED ON THE RESULTS : the chosen model for projecting SALE_CL is LDA

# ## MODEL SALE_CC : PARAMETERS AND DEPLOYMENT

# ##### SELECTION OF TARGET METRIC + EXCLUSION OF METRICS

# In[830]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_cc = df2[['Sale_CC']]


# ##### SPLIT OF THE DATASET INTO TRAIN / CROSS VALIDATION / TEST

# In[831]:


#import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cc_deploy, x_test_cc_deploy, y_train_cc_deploy,y_test_cc_deploy = train_test_split(x,y_cc,test_size=0.2,random_state=42)
x_train_cc_deploy, x_cv_cc_deploy, y_train_cc_deploy, y_cv_cc_deploy = train_test_split(x_train_cc_deploy,y_train_cc_deploy,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_cc_deploy.shape)
print('Training Labels Shape:', y_train_cc_deploy.shape)
print('Cross Validation Testing Features Shape:', x_cv_cc_deploy.shape)
print('Cross Validation Testing Labels Shape:', y_cv_cc_deploy.shape)
print('Testing Features Shape:', x_test_cc_deploy.shape)
print('Testing Labels Shape:', y_test_cc_deploy.shape,'\n')

print('Right Split with an overall matching number of observations :',x_train_cc_deploy.shape[0] + x_cv_cc_deploy.shape[0] + x_test_cc_deploy.shape[0] == x.shape[0])
print('Right Split with an overall matching number of variables :', (x_train_cc_deploy.shape[1] == x_cv_cc_deploy.shape[1] == x_test_cc_deploy.shape[1]) == y_cc.shape[1])


# ### LOGISTIC REGRESSION

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[832]:


logmodel = LogisticRegression(max_iter=1000)

#Plug the model to the training set
cc_model_lr_deploy_train = logmodel.fit(x_train_cc_deploy,y_train_cc_deploy.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cc_predictions_lr_test_deploy = cc_model_lr_deploy_train.predict(x_test_cc_deploy)
from sklearn.model_selection import cross_val_score
cc_acc_lr_deploy = round(accuracy_score(y_test_cc_deploy,cc_predictions_lr_test_deploy),3)
print("TEST SET\nAccuracy Score :", cc_acc_lr_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_deploy, cc_predictions_lr_test_deploy),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_lr_weighted_deploy = precision_score(y_test_cc_deploy, cc_predictions_lr_test_deploy, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_lr_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cc_lr_weighted_deploy = recall_score(y_test_cc_deploy, cc_predictions_lr_test_deploy, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_lr_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_deploy['Sale_CC'], cc_predictions_lr_test_deploy)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_deploy['Sale_CC'],cc_predictions_lr_test_deploy),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[833]:


logmodel = LogisticRegression(max_iter=1000)

#Plug the model to the cross validation set
cc_model_lr_deploy_cv = logmodel.fit(x_cv_cc_deploy,y_cv_cc_deploy.values.ravel())
print('------ CROSS VALIDATION ------\n')

#Evaluate the model on the test set
cc_predictions_lr_test_deploy_cv = cc_model_lr_deploy_cv.predict(x_test_cc_deploy)

from sklearn.model_selection import cross_val_score
cc_acc_lr_deploy = round(accuracy_score(y_test_cc_deploy,cc_predictions_lr_test_deploy_cv),3)
print("TEST SET\nAccuracy Score :", cc_acc_lr_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_deploy, cc_predictions_lr_test_deploy_cv),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_lr_weighted_deploy = precision_score(y_test_cc_deploy, cc_predictions_lr_test_deploy_cv, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_lr_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cc_lr_weighted_deploy = recall_score(y_test_cc_deploy, cc_predictions_lr_test_deploy_cv, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_lr_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_deploy['Sale_CC'], cc_predictions_lr_test_deploy_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_deploy['Sale_CC'],cc_predictions_lr_test_deploy_cv),3))
plt.show()


# ###### TEST ON THE TRAINING DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[834]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


# In[835]:


#After few trials, 21 is the number optimizing the best our KPI
selector = SelectKBest(chi2, k=21)
x_train_cc_new_params = selector.fit_transform(x_train_cc_deploy,y_train_cc_deploy)
x_train_cc_new_params.shape


# In[836]:


x_test_cc_new_params = selector.fit_transform(x_test_cc_deploy,y_test_cc_deploy)
x_test_cc_new_params.shape


# In[837]:


x_cv_cc_new_params = selector.fit_transform(x_cv_cc_deploy,y_cv_cc_deploy)
x_cv_cc_new_params.shape


# In[838]:


#for visualising which columns have been selected
cols = selector.get_support(indices=True)
features_df_new = x_train_cc_deploy.iloc[:,cols]
features_df_new.columns


# In[839]:


logmodel = LogisticRegression(max_iter=1000)

y_train_cc_new_params = y_train_cc_deploy
y_test_cc_new_params = y_test_cc_deploy

#Plug the model to the training set
cc_model_lr_optimized = logmodel.fit(x_train_cc_new_params,y_train_cc_new_params.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cc_predictions_lr_optimized = cc_model_lr_optimized.predict(x_test_cc_new_params)
from sklearn.model_selection import cross_val_score

cc_acc_lr_optimized = round(accuracy_score(y_test_cc_new_params,cc_predictions_lr_optimized),3)
print("TEST SET\nAccuracy Score :", cc_acc_lr_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_new_params,cc_predictions_lr_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_lr_weighted_optimized = precision_score(y_test_cc_new_params,cc_predictions_lr_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_lr_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cc_lr_weighted_optimized = recall_score(y_test_cc_new_params,cc_predictions_lr_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_lr_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_lr_op = confusion_matrix(y_test_cc_new_params,cc_predictions_lr_optimized)
print(cmx_cc_lr_op,'\n')
TP_cc_lr_op = cmx_cc_lr_op[1,1]
TN_cc_lr_op = cmx_cc_lr_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_lr_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_lr_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_lr_op[1,1]+cmx_cc_lr_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_new_params['Sale_CC'], cc_predictions_lr_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_new_params['Sale_CC'],cc_predictions_lr_optimized),3))
plt.show()


# ###### TEST ON THE CROSS VALIDATION DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[840]:


logmodel = LogisticRegression(max_iter=1000)

y_cv_cc_new_params = y_cv_cc_deploy
y_test_cc_new_params = y_test_cc_deploy

#Plug the model to thecross validation set
cc_model_lr_optimized = logmodel.fit(x_cv_cc_new_params,y_cv_cc_new_params.values.ravel())
print('------ CROSS VALIDATION SET ------\n')

#Evaluate the model on the test set
cc_predictions_lr_optimized = cc_model_lr_optimized.predict(x_test_cc_new_params)
from sklearn.model_selection import cross_val_score

cc_acc_lr_optimized = round(accuracy_score(y_test_cc_new_params,cc_predictions_lr_optimized),3)
print("TEST SET\nAccuracy Score :", cc_acc_lr_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_new_params,cc_predictions_lr_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_lr_weighted_optimized = precision_score(y_test_cc_new_params,cc_predictions_lr_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_lr_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cc_lr_weighted_optimized = recall_score(y_test_cc_new_params,cc_predictions_lr_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_lr_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_lr_op = confusion_matrix(y_test_cc_new_params,cc_predictions_lr_optimized)
print(cmx_cc_lr_op,'\n')
TP_cc_lr_op = cmx_cc_lr_op[1,1]
TN_cc_lr_op = cmx_cc_lr_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_lr_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_lr_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_lr_op[1,1]+cmx_cc_lr_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_new_params['Sale_CC'], cc_predictions_lr_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_new_params['Sale_CC'],cc_predictions_lr_optimized),3))
plt.show()


# ### SVM

# ###### TEST ON THE TRAINING DATASET

# In[841]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
svmmodel = svm.SVC()

#Plug the model to the training set
cc_model_svm_deploy_train = svmmodel.fit(x_train_cc_deploy,y_train_cc_deploy.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cc_predictions_svm_test_deploy = cc_model_svm_deploy_train.predict(x_test_cc_deploy)


from sklearn.model_selection import cross_val_score
cc_acc_svm_deploy = round(accuracy_score(y_test_cc_deploy,cc_predictions_svm_test_deploy),3)
print("TEST SET\nAccuracy Score :", cc_acc_svm_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_deploy, cc_predictions_svm_test_deploy),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_svm_weighted_deploy = precision_score(y_test_cc_deploy, cc_predictions_svm_test_deploy, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_svm_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cc_svm_weighted_deploy = recall_score(y_test_cc_deploy, cc_predictions_svm_test_deploy, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_svm_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_deploy['Sale_CC'], cc_predictions_svm_test_deploy)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_deploy['Sale_CC'],cc_predictions_svm_test_deploy),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[842]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
svmmodel = svm.SVC()

#Plug the model to the cross validation set
cc_model_svm_deploy_cv = svmmodel.fit(x_cv_cc_deploy,y_cv_cc_deploy.values.ravel())
print('------ CROSS VALIDATION ------\n')

#Evaluate the model on the test set
cc_predictions_svm_test_deploy_cv = cc_model_svm_deploy_cv.predict(x_test_cc_deploy)

from sklearn.model_selection import cross_val_score
cc_acc_svm_deploy = round(accuracy_score(y_test_cc_deploy,cc_predictions_svm_test_deploy_cv),3)
print("TEST SET\nAccuracy Score :", cc_acc_svm_deploy)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_deploy, cc_predictions_svm_test_deploy_cv),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_svm_weighted_deploy = precision_score(y_test_cc_deploy, cc_predictions_svm_test_deploy_cv, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_svm_weighted_deploy,3))

#recall
from sklearn.metrics import recall_score
recall_cc_svm_weighted_deploy = recall_score(y_test_cc_deploy, cc_predictions_svm_test_deploy_cv, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_svm_weighted_deploy,3),'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_deploy['Sale_CC'], cc_predictions_svm_test_deploy_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_deploy['Sale_CC'],cc_predictions_svm_test_deploy_cv),3))
plt.show()


# ###### TEST ON THE TRAINING DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[843]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


# In[844]:


#After few trials, 12 is the number optimizing the best our KPI
selector = SelectKBest(chi2, k=12)
x_train_cc_new_params_svm = selector.fit_transform(x_train_cc_deploy,y_train_cc_deploy)
x_train_cc_new_params_svm.shape


# In[845]:


x_test_cc_new_params_svm = selector.fit_transform(x_test_cc_deploy,y_test_cc_deploy)
x_test_cc_new_params_svm.shape


# In[846]:


x_cv_cc_new_params_svm = selector.fit_transform(x_cv_cc_deploy,y_cv_cc_deploy)
x_cv_cc_new_params_svm.shape


# In[847]:


#for visualising which columns have been selected
cols = selector.get_support(indices=True)
features_df_new = x_train_cc_deploy.iloc[:,cols]
features_df_new.columns


# In[848]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
svmmodel = svm.SVC()

y_train_cc_new_params_svm = y_train_cc_deploy
y_test_cc_new_params_svm = y_test_cc_deploy

#Plug the model to the training set
cc_model_svm_optimized = svmmodel.fit(x_train_cc_new_params_svm,y_train_cc_new_params_svm.values.ravel())
print('------ TRAINING SET ------\n')

#Evaluate the model on the test set
cc_predictions_svm_optimized = cc_model_svm_optimized.predict(x_test_cc_new_params_svm)
from sklearn.model_selection import cross_val_score

cc_acc_svm_optimized = round(accuracy_score(y_test_cc_new_params_svm,cc_predictions_svm_optimized),3)
print("TEST SET\nAccuracy Score :", cc_acc_svm_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_new_params_svm,cc_predictions_svm_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_svm_weighted_optimized = precision_score(y_test_cc_new_params_svm,cc_predictions_svm_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_svm_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cc_svm_weighted_optimized = recall_score(y_test_cc_new_params_svm,cc_predictions_svm_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_svm_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_svm_op = confusion_matrix(y_test_cc_new_params_svm,cc_predictions_svm_optimized)
print(cmx_cc_svm_op,'\n')
TP_cc_svm_op = cmx_cc_svm_op[1,1]
TN_cc_svm_op = cmx_cc_svm_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_svm_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_svm_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_svm_op[1,1]+cmx_cc_svm_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_new_params_svm['Sale_CC'], cc_predictions_svm_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_new_params_svm['Sale_CC'],cc_predictions_svm_optimized),3))
plt.show()


# ###### TEST ON THE CROSS VALIDATION DATASET WITH SELECTED FEATURES (RUNNING OPTIMISED MODEL)

# In[849]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
svmmodel = svm.SVC()

y_cv_cc_new_params_svm = y_cv_cc_deploy
y_test_cc_new_params_svm = y_test_cc_deploy

#Plug the model to thecross validation set
cc_model_svm_optimized = svmmodel.fit(x_cv_cc_new_params_svm,y_cv_cc_new_params_svm.values.ravel())
print('------ CROSS VALIDATION SET ------\n')

#Evaluate the model on the test set
cc_predictions_svm_optimized = cc_model_svm_optimized.predict(x_test_cc_new_params_svm)
from sklearn.model_selection import cross_val_score

cc_acc_svm_optimized = round(accuracy_score(y_test_cc_new_params_svm,cc_predictions_svm_optimized),3)
print("TEST SET\nAccuracy Score :", cc_acc_svm_optimized)

#Evaluating the results
from sklearn import metrics
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test_cc_new_params_svm,cc_predictions_svm_optimized),3))

# precision 
from sklearn.metrics import precision_score
precision_cc_svm_weighted_optimized = precision_score(y_test_cc_new_params_svm,cc_predictions_svm_optimized, average='weighted')
print('Precision Rate - Weighted :',round(precision_cc_svm_weighted_optimized,3))

#recall
from sklearn.metrics import recall_score
recall_cc_svm_weighted_optimized = recall_score(y_test_cc_new_params_svm,cc_predictions_svm_optimized, average='weighted')
print('Recall Rate - Weighted :',round(recall_cc_svm_weighted_optimized,3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_svm_op = confusion_matrix(y_test_cc_new_params_svm,cc_predictions_svm_optimized)
print(cmx_cc_svm_op,'\n')
TP_cc_svm_op = cmx_cc_svm_op[1,1]
TN_cc_svm_op = cmx_cc_svm_op[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_svm_op[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_svm_op[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_svm_op[1,1]+cmx_cc_svm_op[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_cc_new_params_svm['Sale_CC'], cc_predictions_svm_optimized)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_cc_new_params_svm['Sale_CC'],cc_predictions_svm_optimized),3))
plt.show()


# ##### ELECTING THE FINAL MODEL 

# In[850]:


print('MODEL 1 LOGISTIC REGRESSION -----------------')
print('Precision Rate - Weighted :',round(precision_cc_lr_weighted_optimized,3))
print('Recall Rate - Weighted :',round(recall_cc_lr_weighted_optimized,3))
print('AUC :', round(roc_auc_score(y_test_cc_new_params['Sale_CC'],cc_predictions_lr_optimized),3),'\n')
print('TOTAL TRUE :',cmx_cc_lr_op[1,1]+cmx_cc_lr_op[0,0])
print(cmx_cc_lr_op,'\n')

print('MODEL 2 SVM ---------------------------------')
print('Precision Rate - Weighted :',round(precision_cc_svm_weighted_optimized,3))
print('Recall Rate - Weighted :',round(recall_cc_svm_weighted_optimized,3))
print('AUC :', round(roc_auc_score(y_test_cc_new_params_svm['Sale_CC'],cc_predictions_svm_optimized),3),'\n')
print('TOTAL TRUE :',cmx_cc_svm_op[1,1]+cmx_cc_svm_op[0,0])
print(cmx_cc_svm_op)


# ######## BASED ON THE RESULTS : the chosen model for projecting SALE_CC is LOGISTIC REGRESSION

# ### MODELS ELECTED FOR DEPLOYMENT :
# >  - Sale_MF : LDA
#  - Sale_CL : LDA
#  - Sale_CC : LOGISTIC
# 

# # 5. MODELS DEPLOYMENT TO PREDICT 3 SALES TARGETS

# ##### SALE_MF : DEPLOYING THE MODEL TO CREATE PREDICTIONS

# ##### Definition of the target over the target dataset

# In[877]:


df_2_40_3 = df_2_40_3.set_value(0, 'Sale_MF', 1) 
df_2_40_3[['Sale_MF']].nunique()


# In[878]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


# In[879]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}

features_60 = [a for a in df2 if a not in dropers]
x_mf_df_2_60 = df2[features_60]
y_mf_df_2_60 = df2[['Sale_MF']]

features_40 = [a for a in df_2_40_3 if a not in dropers]
x_mf_df_2_40 = df_2_40_3[features_40]
y_mf_df_2_40 = df_2_40_3[['Sale_MF']]

selector = SelectKBest(chi2, k=21)
x_mf_df_2_60 = selector.fit_transform(x_mf_df_2_60,y_mf_df_2_60)
x_mf_df_2_40 = selector.fit_transform(x_mf_df_2_40,y_mf_df_2_40)

print(y_mf_df_2_40.nunique())
print('x 60% :', x_mf_df_2_60.shape,'\ny 60% :', y_mf_df_2_60.shape,'\n\nx 40%:', x_mf_df_2_40.shape,'\ny 40%:', y_mf_df_2_40.shape)


# 
# ###### Model definition : LDA

# In[880]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
D = sc.fit_transform(x_mf_df_2_60)

# Fit the classifier to the data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
lda = LDA(n_components=1)
D = lda.fit_transform(D,y_mf_df_2_60)

mf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
mf_model_lda_deployed = mf_classifier.fit(D,y_mf_df_2_60.values.ravel())

#project values
O = sc.fit_transform(x_mf_df_2_40)
O = lda.fit_transform(O,y_mf_df_2_40)
y_pred_mf_lda_ok = mf_model_lda_deployed.predict(O)
pred_labels_mf = {'Client': df_2_40_3['Client'], 'Sale_MF_Pred': y_pred_mf_lda_ok}
pred_mf_lda = pd.DataFrame(pred_labels_mf,columns= ['Client', 'Sale_MF_Pred']).sort_values(by=['Client']).set_index('Client')

print('Verification Projection :',pred_mf_lda['Sale_MF_Pred'].unique(),' with nunique() of',pred_mf_lda['Sale_MF_Pred'].nunique())
print('Number of Sales projected :',pred_mf_lda['Sale_MF_Pred'].sum(),' / out of ',pred_mf_lda['Sale_MF_Pred'].count())
pred_mf_lda


# ##### SALE_CL : DEPLOYING THE MODEL TO CREATE PREDICTIONS

# ##### Definition of the target over the target dataset

# In[881]:


df_2_40_3 = df_2_40_3.set_value(0, 'Sale_CL', 1) 
df_2_40_3[['Sale_CL']].nunique()


# In[882]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


# In[883]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}

features_60 = [a for a in df2 if a not in dropers]
x_cl_df_2_60 = df2[features_60]
y_cl_df_2_60 = df2[['Sale_CL']]

features_40 = [a for a in df_2_40_3 if a not in dropers]
x_cl_df_2_40 = df_2_40_3[features_40]
y_cl_df_2_40 = df_2_40_3[['Sale_CL']]

selector = SelectKBest(chi2, k=20)
x_cl_df_2_60 = selector.fit_transform(x_cl_df_2_60,y_cl_df_2_60)
x_cl_df_2_40 = selector.fit_transform(x_cl_df_2_40,y_cl_df_2_40)

print(y_cl_df_2_40.nunique())
print('x 60% :', x_cl_df_2_60.shape,'\ny 60% :', y_cl_df_2_60.shape,'\n\nx 40%:', x_cl_df_2_40.shape,'\ny 40%:', y_cl_df_2_40.shape)


# 
# ###### Model definition : LDA

# In[884]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
D = sc.fit_transform(x_cl_df_2_60)

# Fit the classifier to the data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
lda = LDA(n_components=1)
D = lda.fit_transform(D,y_cl_df_2_60)

cl_classifier = RandomForestClassifier(max_depth=2, random_state=0)
cl_model_lda_deployed = cl_classifier.fit(D,y_cl_df_2_60.values.ravel())

#project values
O = sc.fit_transform(x_cl_df_2_40)
O = lda.fit_transform(O,y_cl_df_2_40)
y_pred_cl_lda_ok = cl_model_lda_deployed.predict(O)
pred_labels_cl = {'Client': df_2_40_3['Client'], 'Sale_CL_Pred': y_pred_cl_lda_ok}
pred_cl_lda = pd.DataFrame(pred_labels_cl,columns= ['Client', 'Sale_CL_Pred']).sort_values(by=['Client']).set_index('Client')

print('Verification Projection :',pred_cl_lda['Sale_CL_Pred'].unique(),' with nunique() of',pred_cl_lda['Sale_CL_Pred'].nunique())
print('Number of Sales projected :',pred_cl_lda['Sale_CL_Pred'].sum(),' / out of ',pred_cl_lda['Sale_CL_Pred'].count())
pred_cl_lda


# ##### SALE_CC : DEPLOYING THE MODEL TO CREATE PREDICTIONS

# ##### Definition of the target over the target dataset

# In[885]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


# In[886]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}

features_60 = [a for a in df2 if a not in dropers]
x_cc_df_2_60 = df2[features_60]
y_cc_df_2_60 = df2[['Sale_CC']]

features_40 = [a for a in df_2_40_3 if a not in dropers]
x_cc_df_2_40 = df_2_40_3[features_40]
y_cc_df_2_40 = df_2_40_3[['Sale_CC']]

selector = SelectKBest(chi2, k=21)
x_cc_df_2_60 = selector.fit_transform(x_cc_df_2_60,y_cc_df_2_60)
x_cc_df_2_40 = selector.fit_transform(x_cc_df_2_40,y_cc_df_2_40)

print(y_cc_df_2_40.nunique())
print('x 60% :', x_cc_df_2_60.shape,'\ny 60% :', y_cc_df_2_60.shape,'\n\nx 40%:', x_cc_df_2_40.shape,'\ny 40%:', y_cc_df_2_40.shape)


# ###### Model definition : Logistic Regression

# In[887]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=10000)
logreg.fit(x_cc_df_2_60,y_cc_df_2_60)
y_pred_cc=logreg.predict(x_cc_df_2_40)

pred_labels = {'Client': df_2_40_3['Client'], 'Sale_CC_Pred': y_pred_cc}

pred_cc_lr = pd.DataFrame(pred_labels,columns= ['Client', 'Sale_CC_Pred']).sort_values(by=['Client']).set_index('Client') 
print('Verification Projection :',pred_cc_lr['Sale_CC_Pred'].unique(),' with nunique() of',pred_cc_lr['Sale_CC_Pred'].nunique())
print('Number of Sales projected :',pred_cc_lr['Sale_CC_Pred'].sum(),' / out of ',pred_cc_lr['Sale_CC_Pred'].count())
pred_cc_lr


# ### FINAL PROJECTIONS FOR SALES

# #### DATASET 40%

# In[888]:


M1 = pd.merge(pred_mf_lda, pred_cl_lda, on='Client', how='left')
df_forecast_sales = pd.merge(M1,pred_cc_lr, on='Client', how='left')
df_forecast_sales


# In[890]:


df_2_40_Sales_Pred = pd.merge(df_forecast_sales,df_2_40,on="Client",how='left')
df_2_40_Sales_Pred

#Exportation of the dataset
#df_2_40_Sales_Pred.to_pickle('df_2_40_Sales_Pred.dat')


# # ------------ REVENUES

# # 6. MODELS EXPLORATION : Which to choose for each Revenue objective

# In[1388]:


#df2 = df2.drop(['Responding_Client','Sale_Total','Revenue_Total'], axis=1)
print(df2.head(0))


# ### FOCUS 1 : REVENUE_MF 

# #### 1. CREATE THE TRAINING SET FOR THE REVENUE_MF TARGET

# In[1389]:


MF_ = df2.loc[:, df2.columns.str.endswith("MF")]
#print(MF_.head(0))


# In[1390]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_mf_rv = df2[['Revenue_MF']]


# In[1391]:


import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_mf_rv, x_test_mf_rv, y_train_mf_rv,y_test_mf_rv = train_test_split(x,y_mf_rv,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_mf_rv.shape)
print('Training Labels Shape:', y_train_mf_rv.shape)
print('Testing Features Shape:', x_test_mf_rv.shape)
print('Testing Labels Shape:', y_test_mf_rv.shape)


# #### 2. EVALUATE ACCURACY OF MODELS ON THE REVENUE_MF TARGET

# ##### LINEAR REGRESSION

# In[1392]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv = lm.fit(x_train_mf_rv, y_train_mf_rv.values.ravel())

#Evaluate the model on the test set
mf_predictions_lm_rv = model_lm_rv.score(x_test_mf_rv, y_test_mf_rv)
mf_predictions_lm_rv = model_lm_rv.predict(x_test_mf_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_lm = round(metrics.mean_absolute_error(y_test_mf_rv, mf_predictions_lm_rv),3)
print('Mean Absolute Error:',mean_abs_error_mf_lm)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf_rv, mf_predictions_lm_rv),3),'\n\n')


#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_mf_lm = np.zeros_like(mf_predictions_lm_rv)    # initialise a matrix full with zeros
y_pred_classes_mf_lm[mf_predictions_lm_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_mf_lm = np.zeros_like(y_test_mf_rv)
y_test_classes_mf_lm[y_test_mf_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lm = confusion_matrix(y_test_classes_mf_lm, y_pred_classes_mf_lm)
print(cmx_mf_lm,'\n')
TP_mf_lm = cmx_mf_lm[1,1]
TN_mf_lm = cmx_mf_lm[0,0]
FP_mf_lm = cmx_mf_lm[0,1]
FN_mf_lm = cmx_mf_lm[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lm[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_lm[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_lm[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_lm_micro = precision_score(y_test_classes_mf_lm, y_pred_classes_mf_lm, average='micro')
precision_mf_lm_weighted = precision_score(y_test_classes_mf_lm, y_pred_classes_mf_lm, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_lm_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_lm_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_lm_micro = recall_score(y_test_classes_mf_lm, y_pred_classes_mf_lm, average='micro')
recall_mf_lm_weighted = recall_score(y_test_classes_mf_lm, y_pred_classes_mf_lm, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_lm_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_lm_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### DECISION TREE REGRESSOR

# In[1393]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
regressor = DecisionTreeRegressor()

#Plug the model to the training set
mf_model_dt_rv = regressor.fit(x_train_mf_rv,y_train_mf_rv.values.ravel())

#Evaluate the model on the test set
mf_predictions_dt_rv = mf_model_dt_rv.score(x_test_mf_rv, y_test_mf_rv)
mf_predictions_dt_rv = mf_model_dt_rv.predict(x_test_mf_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_dt = round(metrics.mean_absolute_error(y_test_mf_rv, mf_predictions_dt_rv),3)
print('Mean Absolute Error:',mean_abs_error_mf_dt)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf_rv, mf_predictions_dt_rv),3),'\n\n')


#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_mf_dt = np.zeros_like(mf_predictions_dt_rv)    # initialise a matrix full with zeros
y_pred_classes_mf_dt[mf_predictions_dt_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_mf_dt = np.zeros_like(y_test_mf_rv)
y_test_classes_mf_dt[y_test_mf_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_dt = confusion_matrix(y_test_classes_mf_dt, y_pred_classes_mf_dt)
print(cmx_mf_lm,'\n')
TP_mf_dt = cmx_mf_dt[1,1]
TN_mf_dt = cmx_mf_dt[0,0]
FP_mf_dt = cmx_mf_dt[0,1]
FN_mf_dt = cmx_mf_dt[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_dt[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_dt[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_dt[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_dt[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_dt_micro = precision_score(y_test_classes_mf_dt, y_pred_classes_mf_dt, average='micro')
precision_mf_dt_weighted = precision_score(y_test_classes_mf_dt, y_pred_classes_mf_dt, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_dt_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_dt_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_dt_micro = recall_score(y_test_classes_mf_dt, y_pred_classes_mf_dt, average='micro')
recall_mf_dt_weighted = recall_score(y_test_classes_mf_dt, y_pred_classes_mf_dt, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_dt_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_dt_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### NEURAL NETWORK REGRESSION

# In[1394]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
nnr = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

#Plug the model to the training set
mf_model_nnr_rv = nnr.fit(x_train_mf_rv,y_train_mf_rv)#.values.ravel())

#Evaluate the model on the test set
nnr = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')
mf_predictions_nnr_rv = mf_model_nnr_rv.predict(x_test_mf_rv)
mf_acc_nnr = round(accuracy_score(y_test_mf_rv,mf_predictions_nnr_rv),3)
print("TEST SET\nAccuracy Score :", mf_acc_nnr)

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_nnr = round(metrics.mean_absolute_error(y_test_mf_rv, mf_predictions_nnr_rv),3)
print('Mean Absolute Error:',mean_abs_error_mf_nnr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_mf_rv, mf_predictions_nnr_rv),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_mf_rv, mf_predictions_nnr_rv)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_nnr = confusion_matrix(y_test_mf_rv,mf_predictions_nnr_rv)
print(cmx_mf_nnr,'\n')
TP_mf_nnr = cmx_mf_nnr[1,1]
TN_mf_nnr = cmx_mf_nnr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_nnr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_nnr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_nnr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_nnr[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_mf_knn_micro = precision_score(y_test_mf, mf_predictions_knn_test, average='micro')
precision_mf_knn_weighted = precision_score(y_test_mf, mf_predictions_knn_test, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_knn_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_knn_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_knn_micro = recall_score(y_test_mf, mf_predictions_knn_test, average='micro')
recall_mf_knn_weighted = recall_score(y_test_mf, mf_predictions_knn_test, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_knn_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_knn_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVR

# In[1395]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score

svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
mf_model_svr_rv = svr.fit(x_train_mf_rv,y_train_mf_rv.values.ravel())


#Evaluate the model on the test set
mf_predictions_svr_rv = mf_model_svr_rv.predict(x_test_mf_rv)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_mf_svr = np.zeros_like(mf_predictions_svr_rv)    # initialise a matrix full with zeros
y_pred_classes_mf_svr[mf_predictions_svr_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_mf_svr = np.zeros_like(y_test_mf_rv)
y_test_classes_mf_svr[y_test_mf_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_svr = round(metrics.mean_absolute_error(y_test_classes_mf_svr, y_pred_classes_mf_svr),3)
print('Mean Absolute Error:',mean_abs_error_mf_svr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_classes_mf_svr, y_pred_classes_mf_svr),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_classes_mf_svr, y_pred_classes_mf_svr)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_svr = confusion_matrix(y_test_classes_mf_svr,y_test_classes_mf_svr)
print(cmx_mf_svr,'\n')
TP_mf_svr = cmx_mf_svr[1,1]
TN_mf_svr = cmx_mf_svr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_svr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_svr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_mf_svr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_mf_svr[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_svr_micro = precision_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='micro')
precision_mf_svr_weighted = precision_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_svr_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_svr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_svr_micro = recall_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='micro')
recall_mf_svr_weighted = recall_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_svr_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_svr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')

from sklearn.model_selection import KFold
import numpy as np
acc_score = []

#overfitting
#kf = KFold(n_splits=5)
#for train_index, test_index in kf.split(X):
#x_train_mf_rv, x_test_mf_rv = X[train_index], X[test_index]
#y_train_mf_rv, y_test_mf_rv= y[train_index], y[test_index]
#svm_model.fit(x_train_mf_rv,y_train_mf_rv)
#predictions = svm_model.predict(y_test_mf_rv)
#acc_score.append(accuracy_score(predictions, y_test_mf_rv))
#np.mean(acc_score)

print('\nOVERFITTING :')
import numpy as np
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
kf.get_n_splits(x_train_mf_rv)
print(kf)

KFold(n_splits=2, random_state=None, shuffle=False)
X = mf_predictions_svr_rv
y = mf_predictions_svr_rv
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
return print("TRAIN:", train_index, "TEST:", test_index)


# #### 3. RESULT REVENUE_MF  :

# In[1396]:


print('ABSOLUTE ERROR (REVENUE_CL) :\n')
print('- LINEAR Mean Absolute Error:',mean_abs_error_mf_lm)
print('- DT Mean Absolute Error:',mean_abs_error_mf_dt)
#print('- NNR Mean Absolute Error:',mean_abs_error_mf_nnr)
print('- SVR Mean Absolute Error:',mean_abs_error_mf_svr)


# In[1397]:


print('CONFUSION MATRIX POST TEST ANALYSIS (REVENUE_MF) :\n')
print('LINEAR REGRESSION')
print('True Positive :',TP_mf_lm,' ------------> TARGET')
print('True Negative :',TN_mf_lm)
#print('False Positive :',FP_cl_dt)
#print('False Negative :',FN_cl_dt)
print('Precision - micro : ',round(precision_mf_lm_micro,2),'\nPrecision - weighted :',round(precision_mf_lm_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_lm_micro,2),'\nRecall - weighted :',round(recall_mf_lm_weighted,2),' -----> TARGET\n')

print('DECISION TREE REGRESSION')
print('True Positive :',TP_mf_dt,' ------------> TARGET')
print('True Negative :',TN_mf_dt)
print('Precision - micro : ',round(precision_mf_dt_micro,2),'\nPrecision - weighted :',round(precision_mf_dt_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_dt_micro,2),'\nRecall - weighted :',round(recall_mf_dt_weighted,2),' -----> TARGET\n')

print('NEURAL NETWORK REGRESSION')
print('Not available\n')

print('SVR')
print('True Positive :',TP_mf_svr,' ------------> TARGET')
print('True Negative :',TN_mf_svr)
print('Precision - micro : ',round(precision_mf_svr_micro,2),'\nPrecision - weighted :',round(precision_mf_svr_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_mf_svr_micro,2),'\nRecall - weighted :',round(recall_mf_svr_weighted,2),' -----> TARGET\n')


# ### FOCUS 2 : REVENUE_CL

# #### 1. CREATE THE TRAINING SET FOR THE REVENUE_CL TARGET

# In[1398]:


CL_ = df2.loc[:, df2.columns.str.endswith("CL")]
#print(MF_.head(0))


# In[1399]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_cl_rv = df2[['Revenue_CL']]


# In[1400]:


import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cl_rv, x_test_cl_rv, y_train_cl_rv,y_test_cl_rv = train_test_split(x,y_cl_rv,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_cl_rv.shape)
print('Training Labels Shape:', y_train_cl_rv.shape)
print('Testing Features Shape:', x_test_cl_rv.shape)
print('Testing Labels Shape:', y_test_cl_rv.shape)


# #### 2. EVALUATE ACCURACY OF MODELS ON THE REVENUE_CL TARGET

# ##### LINEAR REGRESSION

# In[1401]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv = lm.fit(x_train_cl_rv, y_train_cl_rv.values.ravel())

#Evaluate the model on the test set
cl_predictions_lm_rv = model_lm_rv.score(x_test_cl_rv, y_test_cl_rv)
cl_predictions_lm_rv = model_lm_rv.predict(x_test_cl_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_lm = round(metrics.mean_absolute_error(y_test_cl_rv, cl_predictions_lm_rv),3)
print('Mean Absolute Error:',mean_abs_error_cl_lm)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl_rv, cl_predictions_lm_rv),3),'\n\n')


#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cl_lm = np.zeros_like(cl_predictions_lm_rv)    # initialise a matrix full with zeros
y_pred_classes_cl_lm[cl_predictions_lm_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cl_lm = np.zeros_like(y_test_cl_rv)
y_test_classes_cl_lm[y_test_cl_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lm = confusion_matrix(y_test_classes_cl_lm, y_pred_classes_cl_lm)
print(cmx_cl_lm,'\n')
TP_cl_lm = cmx_cl_lm[1,1]
TN_cl_lm = cmx_cl_lm[0,0]
FP_cl_lm = cmx_cl_lm[0,1]
FN_cl_lm = cmx_cl_lm[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lm[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_lm[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_lm[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_lm_micro = precision_score(y_test_classes_cl_lm, y_pred_classes_cl_lm, average='micro')
precision_cl_lm_weighted = precision_score(y_test_classes_cl_lm, y_pred_classes_cl_lm, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_lm_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_lm_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_lm_micro = recall_score(y_test_classes_cl_lm, y_pred_classes_cl_lm, average='micro')
recall_cl_lm_weighted = recall_score(y_test_classes_cl_lm, y_pred_classes_cl_lm, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_lm_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_lm_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### DECISION TREE REGRESSOR

# In[1402]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
regressor = DecisionTreeRegressor()

#Plug the model to the training set
cl_model_dt_rv = regressor.fit(x_train_cl_rv,y_train_cl_rv.values.ravel())

#Evaluate the model on the test set
cl_predictions_dt_rv = cl_model_dt_rv.score(x_test_cl_rv, y_test_cl_rv)
cl_predictions_dt_rv = cl_model_dt_rv.predict(x_test_cl_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_dt = round(metrics.mean_absolute_error(y_test_cl_rv, cl_predictions_dt_rv),3)
print('Mean Absolute Error:',mean_abs_error_cl_dt)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl_rv, cl_predictions_dt_rv),3),'\n\n')


#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cl_dt = np.zeros_like(cl_predictions_dt_rv)    # initialise a matrix full with zeros
y_pred_classes_cl_dt[cl_predictions_dt_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cl_dt = np.zeros_like(y_test_cl_rv)
y_test_classes_cl_dt[y_test_cl_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_dt = confusion_matrix(y_test_classes_cl_dt, y_pred_classes_cl_dt)
print(cmx_cl_dt,'\n')
TP_cl_dt = cmx_cl_dt[1,1]
TN_cl_dt = cmx_cl_dt[0,0]
FP_cl_dt = cmx_cl_dt[0,1]
FN_cl_dt = cmx_cl_dt[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_dt[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_dt[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_dt[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_dt[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_dt_micro = precision_score(y_test_classes_cl_dt, y_pred_classes_cl_dt, average='micro')
precision_cl_dt_weighted = precision_score(y_test_classes_cl_dt, y_pred_classes_cl_dt, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_dt_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_dt_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_dt_micro = recall_score(y_test_classes_cl_dt, y_pred_classes_cl_dt, average='micro')
recall_cl_dt_weighted = recall_score(y_test_classes_cl_dt, y_pred_classes_cl_dt, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_dt_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_dt_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### NEURAL NETWORK REGRESSION

# In[1403]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
nnr = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

#Plug the model to the training set
cl_model_nnr_rv = nnr.fit(x_train_cl_rv,y_train_cl_rv)#.values.ravel())

#Assesing the accuracy of the model on the cross validation set
from sklearn.model_selection import cross_val_score
scores_cross_val_nnr_cl_rv = cross_val_score(cl_model_nnr_rv, x_train_cl_rv, y_train_cl_rv, cv=5)#, scoring='accuracy')
scval_nnr_cl = print("CROSS VALIDATION--- not sure as it cannot pass the scoring parameter\nAccuracy Score of the Neural Network Regression model on Revenue_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_nnr_cl_rv.mean(), scores_cross_val_nnr_cl_rv.std() * 2),'\n')
scval_nnr_cl 

#Evaluate the model on the test set
nnr = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')
cl_predictions_nnr_rv = cl_model_nnr_rv.predict(x_test_cl_rv)
cl_acc_nnr = round(accuracy_score(y_test_cl_rv,cl_predictions_nnr_rv),3)
print("TEST SET\nAccuracy Score :", cl_acc_nnr)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_nnr = round(metrics.mean_absolute_error(y_test_cl_rv, cl_predictions_nnr_rv),3)
print('Mean Absolute Error:',mean_abs_error_cl_nnr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cl_rv, cl_predictions_nnr_rv),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cl_rv, cl_predictions_nnr_rv)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_nnr = confusion_matrix(y_test_cl_rv,cl_predictions_nnr_rv)
print(cmx_cl_nnr,'\n')
TP_cl_nnr = cmx_cl_nnr[1,1]
TN_cl_nnr = cmx_cl_nnr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_nnr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_nnr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_nnr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_nnr[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cl_nnr_micro = precision_score(y_test_cl, cl_predictions_nnr_test, average='micro')
precision_cl_nnr_weighted = precision_score(y_test_cl, cl_predictions_nnr_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_nnr_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_nnr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_nnr_micro = recall_score(y_test_cl, cl_predictions_nnr_test, average='micro')
recall_cl_nnr_weighted = recall_score(y_test_cl, cl_predictions_nnr_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_nnr_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_nnr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVR

# In[1461]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score

svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
cl_model_svr_rv = svr.fit(x_train_cl_rv,y_train_cl_rv.values.ravel())

#Assesing the accuracy of the model on the cross validation set
#from sklearn.model_selection import cross_val_score
#scores_cross_val_svr_cl_rv = cross_val_score(cl_model_svr_rv, x_train_cl_rv, y_train_cl_rv, cv=5)#, scoring='accuracy')
#scval_svr_cl= print("CROSS VALIDATION--- not sure as it cannot pass the scoring parameter\nAccuracy Score of the SVM model on Revenue_CL : %0.2f (+/- %0.2f)" % (scores_cross_val_svr_cl_rv.mean(), scores_cross_val_svr_cl_rv.std() * 2),'\n')
#scval_svr_cl

#Evaluate the model on the test set
cl_predictions_svr_rv = cl_model_svr_rv.predict(x_test_cl_rv)
#mf_acc_svr = round(accuracy_score(y_test_mf_rv,mf_predictions_svr_rv),3)
#print("TEST SET\nAccuracy Score :", mf_acc_svr)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cl_svr = np.zeros_like(cl_predictions_svr_rv)    # initialise a matrix full with zeros
y_pred_classes_cl_svr[cl_predictions_svr_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cl_svr = np.zeros_like(y_test_cl_rv)
y_test_classes_cl_svr[y_test_cl_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_svr = round(metrics.mean_absolute_error(y_test_classes_cl_svr, y_pred_classes_cl_svr),3)
print('Mean Absolute Error:',mean_abs_error_cl_svr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_classes_cl_svr, y_pred_classes_cl_svr),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_classes_cl_svr, y_pred_classes_cl_svr)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_svr = confusion_matrix(y_test_classes_cl_svr,y_test_classes_cl_svr)
print(cmx_cl_svr,'\n')
TP_cl_svr = cmx_cl_svr[1,1]
TN_cl_svr = cmx_cl_svr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_svr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_svr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cl_svr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cl_svr[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_svr_micro = precision_score(y_test_classes_cl_svr, y_test_classes_cl_svr, average='micro')
precision_cl_svr_weighted = precision_score(y_test_classes_cl_svr, y_test_classes_cl_svr, average='weighted')
print('Precision Rate - Micro :',round(precision_cl_svr_micro,3))
print('Precision Rate - Weighted :',round(precision_cl_svr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cl_svr_micro = recall_score(y_test_classes_cl_svr, y_test_classes_cl_svr, average='micro')
recall_cl_svr_weighted = recall_score(y_test_classes_cl_svr, y_test_classes_cl_svr, average='weighted')
print('Recall Rate - Micro :',round(recall_cl_svr_micro,3))
print('Recall Rate - Weighted :',round(recall_cl_svr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')

from sklearn.model_selection import KFold
import numpy as np
acc_score = []

#overfitting
#kf = KFold(n_splits=5)
#for train_index, test_index in kf.split(X):
#x_train_mf_rv, x_test_mf_rv = X[train_index], X[test_index]
#y_train_mf_rv, y_test_mf_rv= y[train_index], y[test_index]
#svm_model.fit(x_train_mf_rv,y_train_mf_rv)
#predictions = svm_model.predict(y_test_mf_rv)
#acc_score.append(accuracy_score(predictions, y_test_mf_rv))
#np.mean(acc_score)

print('\nOVERFITTING :')
import numpy as np
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
kf.get_n_splits(x_train_cl_rv)
print(kf)

KFold(n_splits=2, random_state=None, shuffle=False)
X = cl_predictions_svr_rv
y = cl_predictions_svr_rv
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
return print("TRAIN:", train_index, "TEST:", test_index)


# #### 3. RESULT REVENUE_CL  :

# In[1405]:


print('ABSOLUTE ERROR (REVENUE_CL) :\n')
print('- LINEAR Mean Absolute Error:',mean_abs_error_cl_lm)
print('- DT Mean Absolute Error:',mean_abs_error_cl_dt)
#print('- NNR Mean Absolute Error:',mean_abs_error_cl_nnr)
print('- SVR Mean Absolute Error:',mean_abs_error_cl_svr)


# In[1406]:


print('CONFUSION MATRIX POST TEST ANALYSIS (REVENUE_CL) :\n')
print('LINEAR REGRESSION')
print('True Positive :',TP_cl_lm,' ------------> TARGET')
print('True Negative :',TN_cl_lm)
#print('False Positive :',FP_cl_dt)
#print('False Negative :',FN_cl_dt)
print('Precision - micro : ',round(precision_cl_lm_micro,2),'\nPrecision - weighted :',round(precision_cl_lm_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_lm_micro,2),'\nRecall - weighted :',round(recall_cl_lm_weighted,2),' -----> TARGET\n')

print('DECISION TREE REGRESSION')
print('True Positive :',TP_cl_dt,' ------------> TARGET')
print('True Negative :',TN_cl_dt)
print('Precision - micro : ',round(precision_cl_dt_micro,2),'\nPrecision - weighted :',round(precision_cl_dt_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_dt_micro,2),'\nRecall - weighted :',round(recall_cl_dt_weighted,2),' -----> TARGET\n')

print('NEURAL NETWORK REGRESSION')
print('Not available\n')

print('SVR')
print('True Positive :',TP_cl_svr,' ------------> TARGET')
print('True Negative :',TN_cl_svr)
print('Precision - micro : ',round(precision_cl_svr_micro,2),'\nPrecision - weighted :',round(precision_cl_svr_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cl_svr_micro,2),'\nRecall - weighted :',round(recall_cl_svr_weighted,2),' -----> TARGET\n')


# ### FOCUS 3 : REVENUE_CC

# #### 1. CREATE THE TRAINING SET FOR THE REVENUE_CC TARGET

# In[1407]:


CC_ = df2.loc[:, df2.columns.str.endswith("CC")]
#print(MF_.head(0))


# In[1408]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_cc_rv = df2[['Revenue_CC']]


# In[1409]:


import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cc_rv, x_test_cc_rv, y_train_cc_rv,y_test_cc_rv = train_test_split(x,y_cc_rv,test_size=0.2,random_state=42)
#use K as for the cross validation or use this function twice

print('Training Features Shape:', x_train_cc_rv.shape)
print('Training Labels Shape:', y_train_cc_rv.shape)
print('Testing Features Shape:', x_test_cc_rv.shape)
print('Testing Labels Shape:', y_test_cc_rv.shape)


# #### 2. EVALUATE ACCURACY OF MODELS ON THE REVENUE_CC TARGET

# ##### LINEAR REGRESSION

# In[1412]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv = lm.fit(x_train_cc_rv, y_train_cc_rv.values.ravel())

#Evaluate the model on the test set
cc_predictions_lm_rv = model_lm_rv.score(x_test_cc_rv, y_test_cc_rv)
cc_predictions_lm_rv = model_lm_rv.predict(x_test_cc_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_lm = round(metrics.mean_absolute_error(y_test_cc_rv, cc_predictions_lm_rv),3)
print('Mean Absolute Error:',mean_abs_error_cc_lm)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc_rv, cc_predictions_lm_rv),3),'\n\n')


#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cc_lm = np.zeros_like(cc_predictions_lm_rv)    # initialise a matrix full with zeros
y_pred_classes_cc_lm[cc_predictions_lm_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cc_lm = np.zeros_like(y_test_cc_rv)
y_test_classes_cc_lm[y_test_cc_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_lm = confusion_matrix(y_test_classes_cc_lm, y_pred_classes_cc_lm)
print(cmx_cc_lm,'\n')
TP_cc_lm = cmx_cc_lm[1,1]
TN_cc_lm = cmx_cc_lm[0,0]
FP_cc_lm = cmx_cc_lm[0,1]
FN_cc_lm = cmx_cc_lm[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_lm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_lm[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_lm[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_lm[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_lm_micro = precision_score(y_test_classes_cc_lm, y_pred_classes_cc_lm, average='micro')
precision_cc_lm_weighted = precision_score(y_test_classes_cc_lm, y_pred_classes_cc_lm, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_lm_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_lm_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_lm_micro = recall_score(y_test_classes_cc_lm, y_pred_classes_cc_lm, average='micro')
recall_cc_lm_weighted = recall_score(y_test_classes_cc_lm, y_pred_classes_cc_lm, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_lm_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_lm_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### DECISION TREE REGRESSOR

# In[1413]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
regressor = DecisionTreeRegressor()

#Plug the model to the training set
cc_model_dt_rv = regressor.fit(x_train_cc_rv,y_train_cc_rv.values.ravel())

#Evaluate the model on the test set
cc_predictions_dt_rv = cc_model_dt_rv.score(x_test_cc_rv, y_test_cc_rv)
cc_predictions_dt_rv = cc_model_dt_rv.predict(x_test_cc_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_dt = round(metrics.mean_absolute_error(y_test_cc_rv, cc_predictions_dt_rv),3)
print('Mean Absolute Error:',mean_abs_error_cc_dt)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc_rv, cc_predictions_dt_rv),3),'\n\n')


#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cc_dt = np.zeros_like(cc_predictions_dt_rv)    # initialise a matrix full with zeros
y_pred_classes_cc_dt[cc_predictions_dt_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cc_dt = np.zeros_like(y_test_cc_rv)
y_test_classes_cc_dt[y_test_cc_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_dt = confusion_matrix(y_test_classes_cc_dt, y_pred_classes_cc_dt)
print(cmx_cc_dt,'\n')
TP_cc_dt = cmx_cc_dt[1,1]
TN_cc_dt = cmx_cc_dt[0,0]
FP_cc_dt = cmx_cc_dt[0,1]
FN_cc_dt = cmx_cc_dt[1,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_dt[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_dt[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_dt[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_dt[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_dt_micro = precision_score(y_test_classes_cc_dt, y_pred_classes_cc_dt, average='micro')
precision_cc_dt_weighted = precision_score(y_test_classes_cc_dt, y_pred_classes_cc_dt, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_dt_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_dt_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_dt_micro = recall_score(y_test_classes_cc_dt, y_pred_classes_cc_dt, average='micro')
recall_cc_dt_weighted = recall_score(y_test_classes_cc_dt, y_pred_classes_cc_dt, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_dt_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_dt_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### NEURAL NETWORK REGRESSION

# In[1414]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
nnr = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

#Plug the model to the training set
cc_model_nnr_rv = nnr.fit(x_train_cc_rv,y_train_cc_rv)#.values.ravel())

#Evaluate the model on the test set
nnr = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')
cc_predictions_nnr_rv = cc_model_nnr_rv.predict(x_test_cc_rv)
cc_acc_nnr = round(accuracy_score(y_test_cc_rv,cc_predictions_nnr_rv),3)
print("TEST SET\nAccuracy Score :", cc_acc_nnr)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_nnr = round(metrics.mean_absolute_error(y_test_cc_rv, cc_predictions_nnr_rv),3)
print('Mean Absolute Error:',mean_abs_error_cc_nnr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_cc_rv, cc_predictions_nnr_rv),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_cc_rv, cc_predictions_nnr_rv)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_nnr = confusion_matrix(y_test_cc_rv,cc_predictions_nnr_rv)
print(cmx_cc_nnr,'\n')
TP_cc_nnr = cmx_cc_nnr[1,1]
TN_cc_nnr = cmx_cc_nnr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_nnr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_nnr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_nnr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_nnr[1,0],'\n\n')


# precision 
from sklearn.metrics import precision_score
precision_cc_nnr_micro = precision_score(y_test_cc, cc_predictions_nnr_test, average='micro')
precision_cc_nnr_weighted = precision_score(y_test_cc, cc_predictions_nnr_test, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_nnr_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_nnr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_nnr_micro = recall_score(y_test_cc, cc_predictions_nnr_test, average='micro')
recall_cc_nnr_weighted = recall_score(y_test_cc, cc_predictions_nnr_test, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_nnr_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_nnr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')


# ##### SVR

# In[1415]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score

svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
cc_model_svr_rv = svr.fit(x_train_cc_rv,y_train_cc_rv.values.ravel())

#Assesing the accuracy of the model on the cross validation set
#from sklearn.model_selection import cross_val_score
#scores_cross_val_svr_cc_rv = cross_val_score(cc_model_svr_rv, x_train_cc_rv, y_train_cc_rv, cv=5)#, scoring='accuracy')
#scval_svr_cc= print("CROSS VALIDATION--- not sure as it cannot pass the scoring parameter\nAccuracy Score of the SVM model on Revenue_CC : %0.2f (+/- %0.2f)" % (scores_cross_val_svr_cc_rv.mean(), scores_cross_val_svr_cc_rv.std() * 2),'\n')
#scval_svr_cc

#Evaluate the model on the test set
cc_predictions_svr_rv = cc_model_svr_rv.predict(x_test_cc_rv)
#cc_acc_svr = round(accuracy_score(y_test_cc_rv,cc_predictions_svr_rv),3)
#print("TEST SET\nAccuracy Score :", cc_acc_svr)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cc_svr = np.zeros_like(cc_predictions_svr_rv)    # initialise a matrix full with zeros
y_pred_classes_cc_svr[cc_predictions_svr_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cc_svr = np.zeros_like(y_test_cc_rv)
y_test_classes_cc_svr[y_test_cc_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_svr = round(metrics.mean_absolute_error(y_test_classes_cc_svr, y_pred_classes_cc_svr),3)
print('Mean Absolute Error:',mean_abs_error_cc_svr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_classes_cc_svr, y_pred_classes_cc_svr),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_classes_cc_svr, y_pred_classes_cc_svr)),3),'\n')

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_svr = confusion_matrix(y_test_classes_cc_svr,y_test_classes_cc_svr)
print(cmx_cc_svr,'\n')
TP_cc_svr = cmx_cc_svr[1,1]
TN_cc_svr = cmx_cc_svr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cc_svr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cc_svr[0,0])
print('FP (False Positive) ---- error -> Positive responding clients wrongly predicted :',cmx_cc_svr[0,1])
print('FN (False Negative) ---- error -> Negative responding clients wrongly predicted :',cmx_cc_svr[1,0],'\n\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_svr_micro = precision_score(y_test_classes_cc_svr, y_test_classes_cc_svr, average='micro')
precision_cc_svr_weighted = precision_score(y_test_classes_cc_svr, y_test_classes_cc_svr, average='weighted')
print('Precision Rate - Micro :',round(precision_cc_svr_micro,3))
print('Precision Rate - Weighted :',round(precision_cc_svr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_cc_svr_micro = recall_score(y_test_classes_cc_svr, y_test_classes_cc_svr, average='micro')
recall_cc_svr_weighted = recall_score(y_test_classes_cc_svr, y_test_classes_cc_svr, average='weighted')
print('Recall Rate - Micro :',round(recall_cc_svr_micro,3))
print('Recall Rate - Weighted :',round(recall_cc_svr_weighted,3),'\n\n')

print('MICRO : Calculate metrics globally by counting the total true positives, false negatives and false positives.\nWEIGHTED : Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).')

from sklearn.model_selection import KFold
import numpy as np
acc_score = []

#overfitting
#kf = KFold(n_splits=5)
#for train_index, test_index in kf.split(X):
#x_train_mf_rv, x_test_mf_rv = X[train_index], X[test_index]
#y_train_mf_rv, y_test_mf_rv= y[train_index], y[test_index]
#svm_model.fit(x_train_mf_rv,y_train_mf_rv)
#predictions = svm_model.predict(y_test_mf_rv)
#acc_score.append(accuracy_score(predictions, y_test_mf_rv))
#np.mean(acc_score)

print('\nOVERFITTING :')
import numpy as np
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
kf.get_n_splits(x_train_cc_rv)
print(kf)

KFold(n_splits=2, random_state=None, shuffle=False)
X = cc_predictions_svr_rv
y = cc_predictions_svr_rv
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
return print("TRAIN:", train_index, "TEST:", test_index)


# #### 3. RESULT REVENUE_CL  :

# In[1416]:


print('ABSOLUTE ERROR (REVENUE_CL) :\n')
print('- LINEAR Mean Absolute Error:',mean_abs_error_cc_lm)
print('- DT Mean Absolute Error:',mean_abs_error_cc_dt)
#print('- NNR Mean Absolute Error:',mean_abs_error_cc_nnr)
print('- SVR Mean Absolute Error:',mean_abs_error_cc_svr)


# In[1417]:


print('CONFUSION MATRIX POST TEST ANALYSIS (REVENUE_CC) :\n')
print('LINEAR REGRESSION')
print('True Positive :',TP_cc_lm,' ------------> TARGET')
print('True Negative :',TN_cc_lm)
#print('False Positive :',FP_cc_dt)
#print('False Negative :',FN_cc_dt)
print('Precision - micro : ',round(precision_cc_lm_micro,2),'\nPrecision - weighted :',round(precision_cc_lm_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_lm_micro,2),'\nRecall - weighted :',round(recall_cc_lm_weighted,2),' -----> TARGET\n')

print('DECISION TREE REGRESSION')
print('True Positive :',TP_cc_dt,' ------------> TARGET')
print('True Negative :',TN_cc_dt)
print('Precision - micro : ',round(precision_cc_dt_micro,2),'\nPrecision - weighted :',round(precision_cc_dt_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_dt_micro,2),'\nRecall - weighted :',round(recall_cc_dt_weighted,2),' -----> TARGET\n')

print('NEURAL NETWORK REGRESSION')
print('Not available\n')

print('SVR')
print('True Positive :',TP_cc_svr,' ------------> TARGET')
print('True Negative :',TN_cc_svr)
print('Precision - micro : ',round(precision_cc_svr_micro,2),'\nPrecision - weighted :',round(precision_cc_svr_weighted,2),' --> TARGET')
print('Recall - micro : ',round(recall_cc_svr_micro,2),'\nRecall - weighted :',round(recall_cc_svr_weighted,2),' -----> TARGET\n')


# ### SUMMARY :
#     1. Revenue_MF : test with Linear regression and SVR
#     2. Revenue_CL : test with Linear regression and SVR
#     3. Revenue_CC : test with Linear regression and SVR
# 

# # 7. MODELS OPTIMIZATION :
# Exploration over the two most promising models for each Revenue objective

# ## MODEL REVENUE_MF : PARAMETERS AND DEPLOYMENT

# ##### SELECTION OF TARGET METRIC + EXCLUSION OF METRICS

# In[1418]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_mf_rv = df2[['Revenue_MF']]


# ##### SPLIT OF THE DATASET INTO TRAIN / CROSS VALIDATION / TEST

# In[1419]:


#import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_mf_deploy_rv, x_test_mf_deploy_rv, y_train_mf_deploy_rv,y_test_mf_deploy_rv = train_test_split(x,y_mf_rv,test_size=0.2,random_state=42)
x_train_mf_deploy_rv, x_cv_mf_deploy_rv, y_train_mf_deploy_rv, y_cv_mf_deploy_rv = train_test_split(x_train_mf_deploy_rv,y_train_mf_deploy_rv,test_size=0.2,random_state=42)

print('Training Features Shape:', x_train_mf_deploy_rv.shape)
print('Training Labels Shape:', y_train_mf_deploy_rv.shape)
print('Cross Validation Testing Features Shape:', x_cv_mf_deploy_rv.shape)
print('Cross Validation Testing Labels Shape:', y_cv_mf_deploy_rv.shape)
print('Testing Features Shape:', x_test_mf_deploy_rv.shape)
print('Testing Labels Shape:', y_test_mf_deploy_rv.shape,'\n')

print('Right Split with an overall matching number of observations :',x_train_mf_deploy_rv.shape[0] + x_cv_mf_deploy_rv.shape[0] + x_test_mf_deploy_rv.shape[0] == x.shape[0])
print('Right Split with an overall matching number of variables :', (x_train_mf_deploy_rv.shape[1] == x_cv_mf_deploy_rv.shape[1] == x_test_mf_deploy_rv.shape[1]) == y_mf_rv.shape[1])


# ### LINEAR REGRESSION

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[1420]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv_eval_mf = lm.fit(x_train_mf_deploy_rv, y_train_mf_deploy_rv.values.ravel())

#Evaluate the model on the test set
mf_predictions_lm_rv = model_lm_rv_eval_mf.score(x_train_mf_deploy_rv, y_train_mf_deploy_rv)
mf_predictions_lm_rv = model_lm_rv_eval_mf.predict(x_train_mf_deploy_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_lm = round(metrics.mean_absolute_error(y_train_mf_deploy_rv, mf_predictions_lm_rv),3)
print('Mean Absolute Error:',mean_abs_error_mf_lm)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_train_mf_deploy_rv, mf_predictions_lm_rv),3),'\n\n')

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_mf_lm = np.zeros_like(mf_predictions_lm_rv)    # initialise a matrix full with zeros
y_pred_classes_mf_lm[mf_predictions_lm_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_mf_lm = np.zeros_like(y_train_mf_deploy_rv)
y_test_classes_mf_lm[y_train_mf_deploy_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lm = confusion_matrix(y_test_classes_mf_lm, y_pred_classes_mf_lm)
print(cmx_mf_lm,'\n')
TP_mf_lm = cmx_mf_lm[1,1]
TN_mf_lm = cmx_mf_lm[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lm[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lm[1,1]+cmx_mf_lm[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_classes_mf_lm,y_pred_classes_mf_lm)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_classes_mf_lm,y_pred_classes_mf_lm),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[1421]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv_eval_mf_cv = lm.fit(x_cv_mf_deploy_rv, y_cv_mf_deploy_rv.values.ravel())

#Evaluate the model on the test set
mf_predictions_lm_rv_cv = model_lm_rv_eval_mf.score(x_cv_mf_deploy_rv, y_cv_mf_deploy_rv)
mf_predictions_lm_rv_cv = model_lm_rv_eval_mf.predict(x_cv_mf_deploy_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_lm_cv = round(metrics.mean_absolute_error(y_cv_mf_deploy_rv, mf_predictions_lm_rv_cv),3)
print('Mean Absolute Error:',mean_abs_error_mf_lm_cv)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_cv_mf_deploy_rv, mf_predictions_lm_rv_cv),3),'\n\n')

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_mf_lm_cv = np.zeros_like(mf_predictions_lm_rv_cv)    # initialise a matrix full with zeros
y_pred_classes_mf_lm_cv[mf_predictions_lm_rv_cv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_mf_lm_cv = np.zeros_like(y_cv_mf_deploy_rv)
y_test_classes_mf_lm_cv[y_cv_mf_deploy_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_lm_cv = confusion_matrix(y_test_classes_mf_lm_cv, y_pred_classes_mf_lm_cv)
print(cmx_mf_lm_cv,'\n')
TP_mf_lm_cv = cmx_mf_lm_cv[1,1]
TN_mf_lm_cv = cmx_mf_lm_cv[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_lm_cv[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_lm_cv[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_lm_cv[1,1]+cmx_mf_lm_cv[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_classes_mf_lm_cv,y_pred_classes_mf_lm_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_classes_mf_lm_cv,y_pred_classes_mf_lm_cv),3))
plt.show()


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[1422]:


#####TEST OPTIMISATION TO PROCEED BUT NO TIME


# ## SVR

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[1423]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score
svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
model_svr_rv_eval_mf = svr.fit(x_train_mf_deploy_rv, y_train_mf_deploy_rv.values.ravel())

#Evaluate the model on the test set
mf_predictions_svr_rv = model_svr_rv_eval_mf.score(x_train_mf_deploy_rv, y_train_mf_deploy_rv)
mf_predictions_svr_rv = model_svr_rv_eval_mf.predict(x_train_mf_deploy_rv)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_mf_svr = np.zeros_like(mf_predictions_svr_rv)    # initialise a matrix full with zeros
y_pred_classes_mf_svr[mf_predictions_svr_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_mf_svr = np.zeros_like(y_train_mf_deploy_rv)
y_test_classes_mf_svr[y_train_mf_deploy_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_svr = round(metrics.mean_absolute_error(y_test_classes_mf_svr, y_pred_classes_mf_svr),3)
print('Mean Absolute Error:',mean_abs_error_mf_svr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_classes_mf_svr, y_pred_classes_mf_svr),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_classes_mf_svr, y_pred_classes_mf_svr)),3),'\n')


#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_svr = confusion_matrix(y_test_classes_mf_svr,y_test_classes_mf_svr)
print(cmx_mf_svr,'\n')
TP_mf_svr = cmx_mf_svr[1,1]
TN_mf_svr = cmx_mf_svr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_svr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_svr[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_svr[1,1]+cmx_mf_svr[0,0],'\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_svr_micro = precision_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='micro')
precision_mf_svr_weighted = precision_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_svr_micro,3))
print('Precision Rate - Weighted :',round(precision_mf_svr_weighted,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_svr_micro = recall_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='micro')
recall_mf_svr_weighted = recall_score(y_test_classes_mf_svr, y_test_classes_mf_svr, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_svr_micro,3))
print('Recall Rate - Weighted :',round(recall_mf_svr_weighted,3),'\n\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_classes_mf_svr, y_test_classes_mf_svr)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_classes_mf_svr, y_test_classes_mf_svr),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[1424]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score
svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
model_svr_rv_eval_mf_cv = svr.fit(x_cv_mf_deploy_rv, y_cv_mf_deploy_rv.values.ravel())

#Evaluate the model on the test set
mf_predictions_svr_rv_cv = model_svr_rv_eval_mf_cv.score(x_cv_mf_deploy_rv, y_cv_mf_deploy_rv)
mf_predictions_svr_rv_cv = model_svr_rv_eval_mf_cv.predict(x_cv_mf_deploy_rv)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_mf_svr_cv = np.zeros_like(mf_predictions_svr_rv_cv)    # initialise a matrix full with zeros
y_pred_classes_mf_svr_cv[mf_predictions_svr_rv_cv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_mf_svr_cv = np.zeros_like(y_cv_mf_deploy_rv)
y_test_classes_mf_svr_cv[y_cv_mf_deploy_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_mf_svr_cv = round(metrics.mean_absolute_error(y_test_classes_mf_svr_cv, y_pred_classes_mf_svr_cv),3)
print('Mean Absolute Error:',mean_abs_error_mf_svr_cv)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_classes_mf_svr_cv, y_pred_classes_mf_svr_cv),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_classes_mf_svr_cv, y_pred_classes_mf_svr_cv)),3),'\n')


#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_mf_svr_cv = confusion_matrix(y_test_classes_mf_svr_cv,y_test_classes_mf_svr_cv)
print(cmx_mf_svr_cv,'\n')
TP_mf_svr_cv = cmx_mf_svr_cv[1,1]
TN_mf_svr_cv = cmx_mf_svr_cv[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_mf_svr_cv[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_mf_svr_cv[0,0])
print('                                                                     TOTAL TRUE :',cmx_mf_svr_cv[1,1]+cmx_mf_svr_cv[0,0],'\n')

# precision 
from sklearn.metrics import precision_score
precision_mf_svr_micro_cv = precision_score(y_test_classes_mf_svr_cv, y_test_classes_mf_svr_cv, average='micro')
precision_mf_svr_weighted_cv = precision_score(y_test_classes_mf_svr_cv, y_test_classes_mf_svr_cv, average='weighted')
print('Precision Rate - Micro :',round(precision_mf_svr_micro_cv,3))
print('Precision Rate - Weighted :',round(precision_mf_svr_weighted_cv,3),'\n')

#recall
from sklearn.metrics import recall_score
recall_mf_svr_micro_cv = recall_score(y_test_classes_mf_svr_cv, y_test_classes_mf_svr_cv, average='micro')
recall_mf_svr_weighted_cv = recall_score(y_test_classes_mf_svr_cv, y_test_classes_mf_svr_cv, average='weighted')
print('Recall Rate - Micro :',round(recall_mf_svr_micro_cv,3))
print('Recall Rate - Weighted :',round(recall_mf_svr_weighted_cv,3),'\n\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_classes_mf_svr_cv, y_test_classes_mf_svr_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_classes_mf_svr_cv, y_test_classes_mf_svr_cv),3))
plt.show()


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[1425]:


#####TEST OPTIMISATION TO PROCEED BUT NO TIME


# ##### CONCLUSION FOR REVENUE_MF

# In[1426]:


print('MODEL 1 LINEAR REGRESSION -----------------')
#print('Precision Rate - Weighted :',round(precision_mf_lm_weighted_optimized,3))
#print('Recall Rate - Weighted :',round(recall_mf_lm_weighted_optimized,3))
print('Mean Absolute Error :',mean_abs_error_mf_lm)
print('AUC :', round(roc_auc_score(y_test_classes_mf_lm,y_pred_classes_mf_lm),3))
print('TOTAL TRUE :',cmx_mf_lm[1,1]+cmx_mf_lm[0,0])
print(cmx_mf_lm,'\n')


print('MODEL 2 SVR ---------------------------------')

print('Mean Absolute Error :',mean_abs_error_mf_svr)
print('AUC :', round(roc_auc_score(y_test_classes_mf_svr,y_pred_classes_mf_svr),3))
print('TOTAL TRUE :',cmx_mf_svr[1,1]+cmx_mf_svr[0,0])
print(cmx_mf_svr,'\n')


# ######## BASED ON THE RESULTS : the most performing model for projecting REVENUE_MF is SVR but suspicion of overfitting
# Linear regression appears to have results with great benchmark to SVR : the surer

# In[ ]:





# ## MODEL REVENUE_CL : PARAMETERS AND DEPLOYMENT

# ##### SELECTION OF TARGET METRIC + EXCLUSION OF METRICS

# In[1427]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_mf_rv = df2[['Revenue_CL']]


# ##### SPLIT OF THE DATASET INTO TRAIN / CROSS VALIDATION / TEST

# In[1428]:


#import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cl_deploy_rv, x_test_cl_deploy_rv, y_train_cl_deploy_rv,y_test_cl_deploy_rv = train_test_split(x,y_cl_rv,test_size=0.2,random_state=42)
x_train_cl_deploy_rv, x_cv_cl_deploy_rv, y_train_cl_deploy_rv, y_cv_cl_deploy_rv = train_test_split(x_train_cl_deploy_rv,y_train_cl_deploy_rv,test_size=0.2,random_state=42)

print('Training Features Shape:', x_train_cl_deploy_rv.shape)
print('Training Labels Shape:', y_train_cl_deploy_rv.shape)
print('Cross Validation Testing Features Shape:', x_cv_cl_deploy_rv.shape)
print('Cross Validation Testing Labels Shape:', y_cv_cl_deploy_rv.shape)
print('Testing Features Shape:', x_test_cl_deploy_rv.shape)
print('Testing Labels Shape:', y_test_cl_deploy_rv.shape,'\n')

print('Right Split with an overall matching number of observations :',x_train_cl_deploy_rv.shape[0] + x_cv_cl_deploy_rv.shape[0] + x_test_cl_deploy_rv.shape[0] == x.shape[0])
print('Right Split with an overall matching number of variables :', (x_train_cl_deploy_rv.shape[1] == x_cv_cl_deploy_rv.shape[1] == x_test_cl_deploy_rv.shape[1]) == y_cl_rv.shape[1])


# ### LINEAR REGRESSION

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[1429]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv_eval_cl = lm.fit(x_train_cl_deploy_rv, y_train_cl_deploy_rv.values.ravel())

#Evaluate the model on the test set
cl_predictions_lm_rv = model_lm_rv_eval_cl.score(x_train_cl_deploy_rv, y_train_cl_deploy_rv)
cl_predictions_lm_rv = model_lm_rv_eval_cl.predict(x_train_cl_deploy_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_lm = round(metrics.mean_absolute_error(y_train_cl_deploy_rv, cl_predictions_lm_rv),3)
print('Mean Absolute Error:',mean_abs_error_cl_lm)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_train_cl_deploy_rv, cl_predictions_lm_rv),3),'\n\n')

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cl_lm = np.zeros_like(cl_predictions_lm_rv)    # initialise a matrix full with zeros
y_pred_classes_cl_lm[cl_predictions_lm_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cl_lm = np.zeros_like(y_train_cl_deploy_rv)
y_test_classes_cl_lm[y_train_cl_deploy_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lm = confusion_matrix(y_test_classes_cl_lm, y_pred_classes_cl_lm)
print(cmx_cl_lm,'\n')
TP_cl_lm = cmx_cl_lm[1,1]
TN_cl_lm = cmx_cl_lm[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lm[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lm[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_lm[1,1]+cmx_cl_lm[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_classes_cl_lm,y_pred_classes_cl_lm)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_classes_cl_lm,y_pred_classes_cl_lm),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[1430]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv_eval_cl_cv = lm.fit(x_cv_cl_deploy_rv, y_cv_cl_deploy_rv.values.ravel())

#Evaluate the model on the test set
cl_predictions_lm_rv_cv = model_lm_rv_eval_cl.score(x_cv_cl_deploy_rv, y_cv_cl_deploy_rv)
cl_predictions_lm_rv_cv = model_lm_rv_eval_cl.predict(x_cv_cl_deploy_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_lm_cv = round(metrics.mean_absolute_error(y_cv_cl_deploy_rv, cl_predictions_lm_rv_cv),3)
print('Mean Absolute Error:',mean_abs_error_cl_lm_cv)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_cv_cl_deploy_rv, cl_predictions_lm_rv_cv),3),'\n\n')

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cl_lm_cv = np.zeros_like(cl_predictions_lm_rv_cv)    # initialise a matrix full with zeros
y_pred_classes_cl_lm_cv[cl_predictions_lm_rv_cv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cl_lm_cv = np.zeros_like(y_cv_cl_deploy_rv)
y_test_classes_cl_lm_cv[y_cv_cl_deploy_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_lm_cv = confusion_matrix(y_test_classes_cl_lm_cv, y_pred_classes_cl_lm_cv)
print(cmx_cl_lm_cv,'\n')
TP_cl_lm_cv = cmx_cl_lm_cv[1,1]
TN_cl_lm_cv = cmx_cl_lm_cv[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_lm_cv[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_lm_cv[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_lm_cv[1,1]+cmx_cl_lm_cv[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_classes_cl_lm_cv,y_pred_classes_cl_lm_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_classes_cl_lm_cv,y_pred_classes_cl_lm_cv),3))
plt.show()


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[1431]:


#####TEST OPTIMISATION TO PROCEED BUT NO TIME


# ## SVR

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[1432]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score
svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
model_svr_rv_eval_cl = svr.fit(x_train_cl_deploy_rv, y_train_cl_deploy_rv.values.ravel())

#Evaluate the model on the test set
cl_predictions_svr_rv = model_svr_rv_eval_cl.score(x_train_cl_deploy_rv, y_train_cl_deploy_rv)
cl_predictions_svr_rv = model_svr_rv_eval_cl.predict(x_train_cl_deploy_rv)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cl_svr = np.zeros_like(cl_predictions_svr_rv)    # initialise a matrix full with zeros
y_pred_classes_cl_svr[cl_predictions_svr_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cl_svr = np.zeros_like(y_train_cl_deploy_rv)
y_test_classes_cl_svr[y_train_cl_deploy_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_svr = round(metrics.mean_absolute_error(y_test_classes_cl_svr, y_pred_classes_cl_svr),3)
print('Mean Absolute Error:',mean_abs_error_cl_svr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_classes_cl_svr, y_pred_classes_cl_svr),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_classes_cl_svr, y_pred_classes_cl_svr)),3),'\n')


#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_svr = confusion_matrix(y_test_classes_cl_svr,y_test_classes_cl_svr)
print(cmx_cl_svr,'\n')
TP_cl_svr = cmx_cl_svr[1,1]
TN_cl_svr = cmx_cl_svr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_svr[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_svr[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_svr[1,1]+cmx_cl_svr[0,0],'\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_svr_micro = precision_score(y_test_classes_cl_svr, y_test_classes_cl_svr, average='micro')
precision_cl_svr_weighted = precision_score(y_test_classes_cl_svr, y_test_classes_cl_svr, average='weighted')


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[1433]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score
svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
model_svr_rv_eval_cl_cv = svr.fit(x_cv_cl_deploy_rv, y_cv_cl_deploy_rv.values.ravel())

#Evaluate the model on the test set
cl_predictions_svr_rv_cv = model_svr_rv_eval_cl_cv.score(x_cv_cl_deploy_rv, y_cv_cl_deploy_rv)
cl_predictions_svr_rv_cv = model_svr_rv_eval_cl_cv.predict(x_cv_cl_deploy_rv)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_classes_cl_svr_cv = np.zeros_like(cl_predictions_svr_rv_cv)    # initialise a matrix full with zeros
y_pred_classes_cl_svr_cv[cl_predictions_svr_rv_cv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_classes_cl_svr_cv = np.zeros_like(y_cv_cl_deploy_rv)
y_test_classes_cl_svr_cv[y_cv_cl_deploy_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_cl_svr_cv = round(metrics.mean_absolute_error(y_test_classes_cl_svr_cv, y_pred_classes_cl_svr_cv),3)
print('Mean Absolute Error:',mean_abs_error_cl_svr_cv)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_classes_cl_svr_cv, y_pred_classes_cl_svr_cv),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_classes_cl_svr_cv, y_pred_classes_cl_svr_cv)),3),'\n')


#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cl_svr_cv = confusion_matrix(y_test_classes_cl_svr_cv,y_test_classes_cl_svr_cv)
print(cmx_cl_svr_cv,'\n')
TP_cl_svr_cv = cmx_cl_svr_cv[1,1]
TN_cl_svr_cv = cmx_cl_svr_cv[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding clients well predicted :',cmx_cl_svr_cv[1,1])
print('TN (True Negative)                   Negative responding clients well predicted :',cmx_cl_svr_cv[0,0])
print('                                                                     TOTAL TRUE :',cmx_cl_svr_cv[1,1]+cmx_cl_svr_cv[0,0],'\n')

# precision 
from sklearn.metrics import precision_score
precision_cl_svr_micro_cv = precision_score(y_test_classes_cl_svr_cv, y_test_classes_cl_svr_cv, average='micro')
precision_cl_svr_weighted_cv = precision_score(y_test_classes_cl_svr_cv, y_test_classes_cl_svr_cv, average='weighted')


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[1434]:


#####TEST OPTIMISATION TO PROCEED BUT NO TIME


# ##### CONCLUSION FOR REVENUE_CL

# In[1435]:


print('MODEL 1 LINEAR REGRESSION -----------------')
#print('Precision Rate - Weighted :',round(precision_cl_lm_weighted_optimized,3))
#print('Recall Rate - Weighted :',round(recall_cl_lm_weighted_optimized,3))
print('Mean Absolute Error :',mean_abs_error_cl_lm)
print('AUC :', round(roc_auc_score(y_test_classes_cl_lm,y_pred_classes_cl_lm),3))
print('TOTAL TRUE :',cmx_cl_lm[1,1]+cmx_cl_lm[0,0])
print(cmx_cl_lm,'\n')


print('MODEL 2 SVR ---------------------------------')

print('Mean Absolute Error :',mean_abs_error_cl_svr)
print('AUC :', round(roc_auc_score(y_test_classes_cl_svr,y_pred_classes_cl_svr),3))
print('TOTAL TRUE :',cmx_cl_svr[1,1]+cmx_cl_svr[0,0])
print(cmx_cl_svr,'\n')


# ######## BASED ON THE RESULTS : the most performing model for projecting REVENUE_CL is SVR but suspicion of overfitting
# Linear regression appears to have results with great benchmark to SVR : the surer
# 

# In[ ]:





# ## MODEL REVENUE_CC : PARAMETERS AND DEPLOYMENT

# ##### SELECTION OF TARGET METRIC + EXCLUSION OF METRICS

# In[1436]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}
features = [a for a in df2 if a not in dropers]
print(features)

x = df2[features]
y_mf_rv = df2[['Revenue_CC']]


# ##### SPLIT OF THE DATASET INTO TRAIN / CROSS VALIDATION / TEST

# In[1437]:


#import numpy as np
from sklearn.model_selection import train_test_split
# Split data in train and test (80% of data for training and 20% for testing).
x_train_cc_deploy_rv, x_test_cc_deploy_rv, y_train_cc_deploy_rv,y_test_cc_deploy_rv = train_test_split(x,y_cc_rv,test_size=0.2,random_state=42)
x_train_cc_deploy_rv, x_cv_cc_deploy_rv, y_train_cc_deploy_rv, y_cv_cc_deploy_rv = train_test_split(x_train_cc_deploy_rv,y_train_cc_deploy_rv,test_size=0.2,random_state=42)

print('Training Features Shape:', x_train_cc_deploy_rv.shape)
print('Training Labels Shape:', y_train_cc_deploy_rv.shape)
print('Cross Validation Testing Features Shape:', x_cv_cc_deploy_rv.shape)
print('Cross Validation Testing Labels Shape:', y_cv_cc_deploy_rv.shape)
print('Testing Features Shape:', x_test_cc_deploy_rv.shape)
print('Testing Labels Shape:', y_test_cc_deploy_rv.shape,'\n')

print('Right Split with an overall matching number of observations :',x_train_cc_deploy_rv.shape[0] + x_cv_cc_deploy_rv.shape[0] + x_test_cc_deploy_rv.shape[0] == x.shape[0])
print('Right Split with an overall matching number of variables :', (x_train_cc_deploy_rv.shape[1] == x_cv_cc_deploy_rv.shape[1] == x_test_cc_deploy_rv.shape[1]) == y_cc_rv.shape[1])


# ### LINEAR REGRESSION

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[1438]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv_eval_cc = lm.fit(x_train_cc_deploy_rv, y_train_cc_deploy_rv.values.ravel())

#Evaluate the model on the test set
cc_predictions_lm_rv = model_lm_rv_eval_cc.score(x_train_cc_deploy_rv, y_train_cc_deploy_rv)
cc_predictions_lm_rv = model_lm_rv_eval_cc.predict(x_train_cc_deploy_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_lm = round(metrics.mean_absolute_error(y_train_cc_deploy_rv, cc_predictions_lm_rv),3)
print('Mean Absolute Error:',mean_abs_error_cc_lm)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_train_cc_deploy_rv, cc_predictions_lm_rv),3),'\n\n')

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_ccasses_cc_lm = np.zeros_like(cc_predictions_lm_rv)    # initialise a matrix full with zeros
y_pred_ccasses_cc_lm[cc_predictions_lm_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_ccasses_cc_lm = np.zeros_like(y_train_cc_deploy_rv)
y_test_ccasses_cc_lm[y_train_cc_deploy_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_lm = confusion_matrix(y_test_ccasses_cc_lm, y_pred_ccasses_cc_lm)
print(cmx_cc_lm,'\n')
TP_cc_lm = cmx_cc_lm[1,1]
TN_cc_lm = cmx_cc_lm[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding ccients well predicted :',cmx_cc_lm[1,1])
print('TN (True Negative)                   Negative responding ccients well predicted :',cmx_cc_lm[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_lm[1,1]+cmx_cc_lm[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_ccasses_cc_lm,y_pred_ccasses_cc_lm)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_ccasses_cc_lm,y_pred_ccasses_cc_lm),3))
plt.show()


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[1439]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

#Plug the model to the training set
model_lm_rv_eval_cc_cv = lm.fit(x_cv_cc_deploy_rv, y_cv_cc_deploy_rv.values.ravel())

#Evaluate the model on the test set
cc_predictions_lm_rv_cv = model_lm_rv_eval_cc.score(x_cv_cc_deploy_rv, y_cv_cc_deploy_rv)
cc_predictions_lm_rv_cv = model_lm_rv_eval_cc.predict(x_cv_cc_deploy_rv)

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_lm_cv = round(metrics.mean_absolute_error(y_cv_cc_deploy_rv, cc_predictions_lm_rv_cv),3)
print('Mean Absolute Error:',mean_abs_error_cc_lm_cv)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_cv_cc_deploy_rv, cc_predictions_lm_rv_cv),3),'\n\n')

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_ccasses_cc_lm_cv = np.zeros_like(cc_predictions_lm_rv_cv)    # initialise a matrix full with zeros
y_pred_ccasses_cc_lm_cv[cc_predictions_lm_rv_cv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_ccasses_cc_lm_cv = np.zeros_like(y_cv_cc_deploy_rv)
y_test_ccasses_cc_lm_cv[y_cv_cc_deploy_rv > cutoff] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_lm_cv = confusion_matrix(y_test_ccasses_cc_lm_cv, y_pred_ccasses_cc_lm_cv)
print(cmx_cc_lm_cv,'\n')
TP_cc_lm_cv = cmx_cc_lm_cv[1,1]
TN_cc_lm_cv = cmx_cc_lm_cv[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding ccients well predicted :',cmx_cc_lm_cv[1,1])
print('TN (True Negative)                   Negative responding ccients well predicted :',cmx_cc_lm_cv[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_lm_cv[1,1]+cmx_cc_lm_cv[0,0],'\n')

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_ccasses_cc_lm_cv,y_pred_ccasses_cc_lm_cv)
plt.plot(fpr, tpr)
print('ROC CURVE')
print('Area under the curve - AUC :', round(roc_auc_score(y_test_ccasses_cc_lm_cv,y_pred_ccasses_cc_lm_cv),3))
plt.show()


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[1440]:


#####TEST OPTIMISATION TO PROCEED BUT NO TIME


# ## SVR

# ###### TEST ON THE TRAINING DATASET WITH ALL FEATURES (RUNNING THE DIRTY MODEL NON OPTIMISED)

# In[1441]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score
svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
model_svr_rv_eval_cc = svr.fit(x_train_cc_deploy_rv, y_train_cc_deploy_rv.values.ravel())

#Evaluate the model on the test set
cc_predictions_svr_rv = model_svr_rv_eval_cc.score(x_train_cc_deploy_rv, y_train_cc_deploy_rv)
cc_predictions_svr_rv = model_svr_rv_eval_cc.predict(x_train_cc_deploy_rv)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_ccasses_cc_svr = np.zeros_like(cc_predictions_svr_rv)    # initialise a matrix full with zeros
y_pred_ccasses_cc_svr[cc_predictions_svr_rv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_ccasses_cc_svr = np.zeros_like(y_train_cc_deploy_rv)
y_test_ccasses_cc_svr[y_train_cc_deploy_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_svr = round(metrics.mean_absolute_error(y_test_ccasses_cc_svr, y_pred_ccasses_cc_svr),3)
print('Mean Absolute Error:',mean_abs_error_cc_svr)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_ccasses_cc_svr, y_pred_ccasses_cc_svr),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_ccasses_cc_svr, y_pred_ccasses_cc_svr)),3),'\n')


#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_svr = confusion_matrix(y_test_ccasses_cc_svr,y_test_ccasses_cc_svr)
print(cmx_cc_svr,'\n')
TP_cc_svr = cmx_cc_svr[1,1]
TN_cc_svr = cmx_cc_svr[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding ccients well predicted :',cmx_cc_svr[1,1])
print('TN (True Negative)                   Negative responding ccients well predicted :',cmx_cc_svr[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_svr[1,1]+cmx_cc_svr[0,0],'\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_svr_micro = precision_score(y_test_ccasses_cc_svr, y_test_ccasses_cc_svr, average='micro')
precision_cc_svr_weighted = precision_score(y_test_ccasses_cc_svr, y_test_ccasses_cc_svr, average='weighted')


# ##### TEST OF THE CROSS VALIDATION DATASET

# In[1442]:


from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import accuracy_score
svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
model_svr_rv_eval_cc_cv = svr.fit(x_cv_cc_deploy_rv, y_cv_cc_deploy_rv.values.ravel())

#Evaluate the model on the test set
cc_predictions_svr_rv_cv = model_svr_rv_eval_cc_cv.score(x_cv_cc_deploy_rv, y_cv_cc_deploy_rv)
cc_predictions_svr_rv_cv = model_svr_rv_eval_cc_cv.predict(x_cv_cc_deploy_rv)

#Set off a limit cutoff to transform continuous data into binary data
cutoff = 0.7                                          
y_pred_ccasses_cc_svr_cv = np.zeros_like(cc_predictions_svr_rv_cv)    # initialise a matrix full with zeros
y_pred_ccasses_cc_svr_cv[cc_predictions_svr_rv_cv > cutoff] = 1       # add a 1 if the cutoff was breached
#perform the same for the actual values too:
y_test_ccasses_cc_svr_cv = np.zeros_like(y_cv_cc_deploy_rv)
y_test_ccasses_cc_svr_cv[y_cv_cc_deploy_rv > cutoff] = 1

#Evaluating the results
from sklearn import metrics
mean_abs_error_cc_svr_cv = round(metrics.mean_absolute_error(y_test_ccasses_cc_svr_cv, y_pred_ccasses_cc_svr_cv),3)
print('Mean Absolute Error:',mean_abs_error_cc_svr_cv)
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test_ccasses_cc_svr_cv, y_pred_ccasses_cc_svr_cv),3))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test_ccasses_cc_svr_cv, y_pred_ccasses_cc_svr_cv)),3),'\n')


#confusion matrix
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRIX (Test Set)")
cmx_cc_svr_cv = confusion_matrix(y_test_ccasses_cc_svr_cv,y_test_ccasses_cc_svr_cv)
print(cmx_cc_svr_cv,'\n')
TP_cc_svr_cv = cmx_cc_svr_cv[1,1]
TN_cc_svr_cv = cmx_cc_svr_cv[0,0]
print('TP (True Positive) ---- TARGET ----> POSITIVE responding ccients well predicted :',cmx_cc_svr_cv[1,1])
print('TN (True Negative)                   Negative responding ccients well predicted :',cmx_cc_svr_cv[0,0])
print('                                                                     TOTAL TRUE :',cmx_cc_svr_cv[1,1]+cmx_cc_svr_cv[0,0],'\n')

# precision 
from sklearn.metrics import precision_score
precision_cc_svr_micro_cv = precision_score(y_test_ccasses_cc_svr_cv, y_test_ccasses_cc_svr_cv, average='micro')
precision_cc_svr_weighted_cv = precision_score(y_test_ccasses_cc_svr_cv, y_test_ccasses_cc_svr_cv, average='weighted')


# ##### TEST ON THE TRAINING DATASET WITH IMPROVING PARAMETERS

# In[1443]:


#####TEST OPTIMISATION TO PROCEED BUT NO TIME


# ##### CONCLUSION FOR REVENUE_CC

# In[1444]:


print('MODEL 1 LINEAR REGRESSION -----------------')
#print('Precision Rate - Weighted :',round(precision_cc_lm_weighted_optimized,3))
#print('Recall Rate - Weighted :',round(recall_cc_lm_weighted_optimized,3))
print('Mean Absolute Error :',mean_abs_error_cc_lm)
print('AUC :', round(roc_auc_score(y_test_ccasses_cc_lm,y_pred_ccasses_cc_lm),3))
print('TOTAL TRUE :',cmx_cc_lm[1,1]+cmx_cc_lm[0,0])
print(cmx_cc_lm,'\n')


print('MODEL 2 SVR ---------------------------------')

print('Mean Absolute Error :',mean_abs_error_cc_svr)
print('AUC :', round(roc_auc_score(y_test_ccasses_cc_svr,y_pred_ccasses_cc_svr),3))
print('TOTAL TRUE :',cmx_cc_svr[1,1]+cmx_cc_svr[0,0])
print(cmx_cc_svr,'\n')


# ######## BASED ON THE RESULTS : the most performing model for projecting REVENUE_CC is SVR but suspicion of overfitting
# Linear regression appears to have results with great benchmark to SVR : the surer
# 

# In[ ]:





# ### MODELS ELECTED FOR DEPLOYMENT :
# >  - Revenue_MF :  LINEAR REGRESSION
#  - Revenue_CL :  LINEAR REGRESSION
#  - Revenue_CC :  LINEAR REGRESSION
# 

# # 8. MODELS DEPLOYMENT TO PREDICT 3 REVENUE TARGETS

# ##### REVENUE_MF : DEPLOYING THE MODEL TO CREATE PREDICTIONS

# ##### Definition of the target over the target dataset

# In[1445]:


df_2_40_3 = df_2_40_3.set_value(0, 'Revenue_MF', 1) 
df_2_40_3[['Revenue_MF']].nunique()


# In[1446]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}#,'Revenue_Total', 'Sale_Total', 'Responding_Client'}

features_60 = [a for a in df2 if a not in dropers]
x_mf_df_2_60 = df2[features_60]
y_mf_df_2_60 = df2[['Revenue_MF']]

features_40 = [a for a in df_2_40_3 if a not in dropers]
x_mf_df_2_40 = df_2_40_3[features_40]
y_mf_df_2_40 = df_2_40_3[['Revenue_MF']]

#selector = SelectKBest(chi2, k=21)
#x_mf_df_2_60 = selector.fit_transform(x_mf_df_2_60,y_mf_df_2_60)
#x_mf_df_2_40 = selector.fit_transform(x_mf_df_2_40,y_mf_df_2_40)

print(y_mf_df_2_40.nunique())
print('x 60% :', x_mf_df_2_60.shape,'\ny 60% :', y_mf_df_2_60.shape,'\n\nx 40%:', x_mf_df_2_40.shape,'\ny 40%:', y_mf_df_2_40.shape)


# 
# ###### Model definition : LINEAR REGRESSION

# In[1447]:


#from sklearn import svm
#from sklearn.svm import SVR
#import numpy as np
#from sklearn.metrics import accuracy_score
#svr = SVR(C=1.0, epsilon=0.2)

#Plug the model to the training set
#model_svr_deployed = svr.fit(x_mf_df_2_60, y_mf_df_2_60.values.ravel())
#y_pred_mf_svr_ok = model_svr_deployed.score(x_mf_df_2_60, y_mf_df_2_60)

#y_pred_mf_svr_ok = model_svr_deployed.predict(x_mf_df_2_40)
#y_pred_mf_svr_ok


# In[1448]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

model_lm_rv_eval_mf = lm.fit(x_mf_df_2_60, y_mf_df_2_60.values.ravel())
y_pred_mf_lm_ok = model_lm_rv_eval_mf.predict(x_mf_df_2_40)

pred_labels_mf_rv = {'Client': df_2_40_3['Client'], 'Revenue_MF_Pred': y_pred_mf_lm_ok}
pred_mf_lm = pd.DataFrame(pred_labels_mf_rv,columns= ['Client', 'Revenue_MF_Pred']).sort_values(by=['Client']).set_index('Client')

print('Verification Projection :',pred_mf_lm['Revenue_MF_Pred'].describe,' with nunique() of',pred_mf_lm['Revenue_MF_Pred'].nunique())
print('Number of Sales projected :',pred_mf_lm['Revenue_MF_Pred'].sum(),' / out of ',pred_mf_lm['Revenue_MF_Pred'].count())
pred_mf_lm


# ##### REVENUE_CL : DEPLOYING THE MODEL TO CREATE PREDICTIONS

# ##### Definition of the target over the target dataset

# In[1449]:


df_2_40_3 = df_2_40_3.set_value(0, 'Revenue_CL', 1) 
df_2_40_3[['Revenue_CL']].nunique()


# In[1450]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}

features_60 = [a for a in df2 if a not in dropers]
x_cl_df_2_60 = df2[features_60]
y_cl_df_2_60 = df2[['Revenue_CL']]

features_40 = [a for a in df_2_40_3 if a not in dropers]
x_cl_df_2_40 = df_2_40_3[features_40]
y_cl_df_2_40 = df_2_40_3[['Revenue_CL']]

#selector = SelectKBest(chi2, k=20)
#x_cl_df_2_60 = selector.fit_transform(x_cl_df_2_60,y_cl_df_2_60)
#x_cl_df_2_40 = selector.fit_transform(x_cl_df_2_40,y_cl_df_2_40)

print(y_cl_df_2_40.nunique())
print('x 60% :', x_cl_df_2_60.shape,'\ny 60% :', y_cl_df_2_60.shape,'\n\nx 40%:', x_cl_df_2_40.shape,'\ny 40%:', y_cl_df_2_40.shape)


# 
# ###### Model definition : LINEAR REGRESSION

# In[1451]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

model_lm_rv_eval_cl = lm.fit(x_cl_df_2_60, y_cl_df_2_60.values.ravel())
y_pred_cl_lm_ok = model_lm_rv_eval_cl.predict(x_cl_df_2_40)

pred_labels_cl_rv = {'Client': df_2_40_3['Client'], 'Revenue_CL_Pred': y_pred_cl_lm_ok}
pred_cl_lm = pd.DataFrame(pred_labels_cl_rv,columns= ['Client', 'Revenue_CL_Pred']).sort_values(by=['Client']).set_index('Client')

print('Verification Projection :',pred_cl_lm['Revenue_CL_Pred'].describe,' with nunique() of',pred_cl_lm['Revenue_CL_Pred'].nunique())
print('Number of Sales projected :',pred_cl_lm['Revenue_CL_Pred'].sum(),' / out of ',pred_cl_lm['Revenue_CL_Pred'].count())
pred_cl_lm


# ##### SALE_CC : DEPLOYING THE MODEL TO CREATE PREDICTIONS

# ##### Definition of the target over the target dataset

# In[1452]:


dropers = {'Sale_MF','Revenue_MF','Count_MF','ActBal_MF','Count_CC', 'ActBal_CC', 'Sale_CC', 'Revenue_CC','Count_CL', 'ActBal_CL', 'Sale_CL', 'Revenue_CL'}

features_60 = [a for a in df2 if a not in dropers]
x_cc_df_2_60 = df2[features_60]
y_cc_df_2_60 = df2[['Revenue_CC']]

features_40 = [a for a in df_2_40_3 if a not in dropers]
x_cc_df_2_40 = df_2_40_3[features_40]
y_cc_df_2_40 = df_2_40_3[['Revenue_CC']]

#selector = SelectKBest(chi2, k=21)
#x_cc_df_2_60 = selector.fit_transform(x_cc_df_2_60,y_cc_df_2_60)
#x_cc_df_2_40 = selector.fit_transform(x_cc_df_2_40,y_cc_df_2_40)

print(y_cc_df_2_40.nunique())
print('x 60% :', x_cc_df_2_60.shape,'\ny 60% :', y_cc_df_2_60.shape,'\n\nx 40%:', x_cc_df_2_40.shape,'\ny 40%:', y_cc_df_2_40.shape)


# 
# ###### Model definition : LINEAR REGRESSION

# In[1453]:


from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
lm = linear_model.LinearRegression()

model_lm_rv_eval_cc = lm.fit(x_cc_df_2_60, y_cc_df_2_60.values.ravel())
y_pred_cc_lm_ok = model_lm_rv_eval_cc.predict(x_cc_df_2_40)

pred_labels_cc_rv = {'Client': df_2_40_3['Client'], 'Revenue_CC_Pred': y_pred_cc_lm_ok}
pred_cc_lm = pd.DataFrame(pred_labels_cc_rv,columns= ['Client', 'Revenue_CC_Pred']).sort_values(by=['Client']).set_index('Client')

print('Verification Projection :',pred_cc_lm['Revenue_CC_Pred'].describe,' with nunique() of',pred_cc_lm['Revenue_CC_Pred'].nunique())
print('Number of Sales projected :',pred_cc_lm['Revenue_CC_Pred'].sum(),' / out of ',pred_cc_lm['Revenue_CC_Pred'].count())
pred_cc_lm


# ### FINAL PROJECTIONS FOR REVENUES

# #### DATASET 40%

# In[1454]:


M1_ = pd.merge(pred_mf_lm, pred_cl_lm, on='Client', how='left')
df_forecast_revenues = pd.merge(M1_,pred_cc_lm, on='Client', how='left')
df_forecast_revenues


# # 9. FINAL PROJECTIONS MIXING SALES & REVENUES

# In[1455]:


df_2_40_Pred_Final = pd.merge(df_forecast_revenues,df_2_40_Sales_Pred,on="Client",how='left')
df_2_40_Pred_Final = df_2_40_Pred_Final.drop(['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL'],axis=1)
df_2_40_Pred_Final['Sales_Total_Pred']=df_2_40_Pred_Final['Sale_MF_Pred']+df_2_40_Pred_Final['Sale_CL_Pred']+df_2_40_Pred_Final['Sale_CC_Pred']
df_2_40_Pred_Final['Revenue_Total_Pred']=df_2_40_Pred_Final['Revenue_MF_Pred']+df_2_40_Pred_Final['Revenue_CL_Pred']+df_2_40_Pred_Final['Revenue_CC_Pred']
df_2_40_Pred_Final = df_2_40_Pred_Final.sort_values('Revenue_Total_Pred',ascending=False)
df_2_40_Pred_Final = df_2_40_Pred_Final[df_2_40_Pred_Final['Sales_Total_Pred']!=0]
df_2_40_Pred_Final


# In[1456]:





# In[1472]:


df_2_40_Pred_Final = pd.merge(df_forecast_revenues,df_2_40_Sales_Pred,on="Client",how='left')
df_2_40_Pred_Final = df_2_40_Pred_Final.drop(['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL'],axis=1)
df_2_40_Pred_Final['Sales_Total_Pred']=df_2_40_Pred_Final['Sale_MF_Pred']+df_2_40_Pred_Final['Sale_CL_Pred']+df_2_40_Pred_Final['Sale_CC_Pred']
df_2_40_Pred_Final['Revenue_Total_Pred']=df_2_40_Pred_Final['Revenue_MF_Pred']+df_2_40_Pred_Final['Revenue_CL_Pred']+df_2_40_Pred_Final['Revenue_CC_Pred']
df_2_40_Pred_Final = df_2_40_Pred_Final.sort_values('Revenue_Total_Pred',ascending=False)
df_2_40_Pred_Final = df_2_40_Pred_Final[df_2_40_Pred_Final['Sales_Total_Pred']!=0]
df_2_40_Pred_Final_100clients = df_2_40_Pred_Final_100clients[['Client','Sale_MF_Pred','Revenue_MF_Pred','Revenue_CL_Pred','Sale_CL_Pred','Revenue_CC_Pred','Sale_CC_Pred','Sales_Total_Pred','Revenue_Total_Pred']]
df_2_40_Pred_Final_100clients


# #### LIST OF CLIENTS TO CONTACT FOR PRODUCT MF

# In[1467]:


Clients_To_Contact_MF = df_2_40_Pred_Final_100clients[df_2_40_Pred_Final_100clients['Sale_MF_Pred']!=0]
Clients_To_Contact_MF = Clients_To_Contact_MF[['Client','Revenue_MF_Pred','Sale_MF_Pred']].sort_values('Revenue_MF_Pred',ascending=False)
Clients_To_Contact_MF = Clients_To_Contact_MF[0:100]
print(Clients_To_Contact_MF.nunique())
Clients_To_Contact_MF


# #### LIST OF CLIENTS TO CONTACT FOR PRODUCT CC

# In[1470]:


Clients_To_Contact_CC = df_2_40_Pred_Final_100clients[df_2_40_Pred_Final_100clients['Sale_CC_Pred']!=0]
Clients_To_Contact_CC = Clients_To_Contact_CC[['Client','Revenue_CC_Pred','Sale_CC_Pred']].sort_values('Revenue_CC_Pred',ascending=False)
Clients_To_Contact_CC = Clients_To_Contact_CC[0:100]
print(Clients_To_Contact_CC.nunique())
Clients_To_Contact_CC


# #### LIST OF CLIENTS TO CONTACT FOR PRODUCT CL

# In[1469]:


Clients_To_Contact_CL = df_2_40_Pred_Final_100clients[df_2_40_Pred_Final_100clients['Sale_CL_Pred']!=0]
Clients_To_Contact_CL = Clients_To_Contact_CL[['Client','Revenue_CL_Pred','Sale_CL_Pred']].sort_values('Revenue_CL_Pred',ascending=False)
Clients_To_Contact_CL = Clients_To_Contact_CL[0:100]
print(Clients_To_Contact_CL.nunique())
Clients_To_Contact_CL


# In[1473]:


#Exportation of the dataset
df_2_40_Pred_Final.to_pickle('df_2_40_Pred_Final.dat')
Clients_To_Contact_MF.to_pickle('Clients_To_Contact_MF.dat')
Clients_To_Contact_CL.to_pickle('Clients_To_Contact_CL.dat')
Clients_To_Contact_CC.to_pickle('Clients_To_Contact_CC.dat')

