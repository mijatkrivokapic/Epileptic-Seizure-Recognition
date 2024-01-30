# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from scipy.fft import fft, fftfreq,rfft,fftshift,rfftfreq
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from utilities import *
from sklearn.decomposition import PCA
import pickle

# %% [markdown]
# Skup podataka i pretprocesiranje

# %%
df=pd.read_csv('dataset.csv')
df.drop('Unnamed',axis=1,inplace=True)
df['y'].replace({1:1,2:0,3:0,4:0,5:0},inplace=True)
x=df.drop(columns=['y'])
y=df['y']
df.head()

# %% [markdown]
# Razlika između snimaka tokom i van napada

# %%
_ , axis = plt.subplots(nrows=1,ncols=2)

df2=df[df['y']==0].iloc[:5]
df2.T.iloc[:-1].plot(legend=None,ax=axis[0])
axis[0].title.set_text("Prvih 5 snimaka koji nisu snimljeni tokom epileptičnog napada")

df2=df[df['y']==1].iloc[:5]
df2.T.iloc[:-1].plot(legend=None,ax=axis[1])
axis[1].title.set_text("Prvih 5 snimaka snimljenih tokom epileptičnog napada")
plt.rcParams["figure.figsize"] = (20,20)
plt.show()
plt.style.use('default')

# %% [markdown]
# Izdvajanje obeležja

# %% [markdown]
# Statističke osobine (statistical features)

# %%
normalized_signal=pd.DataFrame(preprocessing.normalize(x),columns=df.columns[:-1])

stat_features=pd.DataFrame()
stat_features['std']=x.std(axis='columns')
stat_features['mean']=x.mean(axis='columns')
stat_features['first difference']=x.diff(axis='columns').abs().T.mean()
stat_features['first difference normalized']=normalized_signal.diff(axis='columns').diff(axis='columns').abs().T.mean()
stat_features['second difference']=x.diff(axis='columns').diff(axis='columns').abs().T.mean()
stat_features['second difference normalized']=normalized_signal.diff(axis='columns').abs().T.mean()
stat_features['y']=y

stat_features

# %% [markdown]
# Autoregresioni model (Autoregressive model)

# %%
#isprobavati sa drugim stepenima regresije

order=10
rows=[]
for i in range(len(x)):
    model=AutoReg(x.iloc[i].values,lags=order).fit()
    row={}
    for j in range(1,order+1):
        row[f"param {j}"]=model.params[j]
    row['y']=y.iloc[i]
    rows.append(row)
ar_model_params=pd.DataFrame(rows)
ar_model_params

# %% [markdown]
# Furieova transformacija

# %%
rows=[]
for i in range(len(x)):
    frequencies=rfft(x.iloc[i])
    row={str(i+1):frequencies[i] for i in range(0,len(frequencies))}
    row['y']=y.iloc[i]
    rows.append(row)
f_transform=pd.DataFrame(rows)

f_transform.head()

# %% [markdown]
# Amplitude frekvencija dobijenih Furieovom transformacijom

# %%
y_plt=f_transform.drop(columns=['y']).iloc[0]
xf = rfftfreq(178, 1 / 178)

plt.plot(xf, np.abs(y_plt))
plt.title("Amplitude frekvencija dobijenih Furieovom transformacijom jednog EEG signala")
plt.xlabel("Frekvencija (Hz)")
plt.ylabel("Amplituda")
plt.show()

# %% [markdown]
# Kako ne utiču sve frekvenicje pojednako, neke se mogu zanemariti. To ćemo postići pimenom PCA algoritma.

# %% [markdown]
# Prvo treba normalizovati i centrirati podatke

# %%
scaler = StandardScaler(with_mean=True, with_std=True)
x_scaled = scaler.fit_transform(abs(f_transform.drop(columns=['y'])))
f_transform_scaled = pd.DataFrame(x_scaled, columns=f_transform.drop(columns=['y']).columns)
f_transform_scaled['y']=f_transform['y'].values

#plot_correlation_for_col(abs(f_transform.drop(columns=[str(i) for i in range(20,91)])), col_name='y')
#plot_correlation_for_col(df_scaled, col_name='y')
f_transform_scaled.head()

# %%
#da li za pca treba da se deli na train i test nakon pravljenja

# pravimo PCA model
f_transform_pca = PCA(n_components=5, random_state=42)
# primenjujemo PCA na originalne atribute
f_transform_components = f_transform_pca.fit_transform(f_transform_scaled.drop(columns=['y']))
# procenat informacija koji smo sačuvali iz originalnih podataka
print(f'ukupna varijansa: {sum(f_transform_pca.explained_variance_ratio_) * 100:.1f}%')

plot_explained_variance(f_transform_pca)

# %% [markdown]
# Podela na trening i test podatke

# %%
stat_x_train, stat_x_test, stat_y_train, stat_y_test = train_test_split(stat_features.drop(columns=['y']), stat_features['y'], train_size=0.8, random_state=42, shuffle=True)
ar_x_train, ar_x_test, ar_y_train, ar_y_test = train_test_split(ar_model_params.drop(columns=['y']), ar_model_params['y'], train_size=0.8, random_state=42, shuffle=True)
ft_x_train, ft_x_test, ft_y_train, ft_y_test = train_test_split(f_transform_components, y, train_size=0.8, random_state=42, shuffle=True)


scaler = StandardScaler()
stat_x_train = scaler.fit_transform(stat_x_train)
stat_x_test = scaler.transform(stat_x_test)

scaler = StandardScaler()
ar_x_train = scaler.fit_transform(ar_x_train)
ar_x_test = scaler.transform(ar_x_test)

scaler = StandardScaler()
ft_x_train = scaler.fit_transform(ft_x_train)
ft_x_test = scaler.transform(ft_x_test) 


# %%

# %%
best_ft=find_best_svc_params(ft_x_train,ft_y_train)
with open('best_ft.pkl', 'wb') as f:
    pickle.dump(best_ft, f)

best_stat=find_best_svc_params(stat_x_train,stat_y_train)
with open('best_stat.pkl', 'wb') as f:
    pickle.dump(best_stat, f)


# %%
best_ar=find_best_svc_params(ar_x_train,ar_y_train)
with open('best_ar.pkl', 'wb') as f:
    pickle.dump(best_ar, f)




