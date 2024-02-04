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
from sklearn.metrics import f1_score

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
""" _ , axis = plt.subplots(nrows=1,ncols=2)

df2=df[df['y']==0].iloc[:5]
df2.T.iloc[:-1].plot(legend=None,ax=axis[0])
axis[0].title.set_text("Prvih 5 snimaka koji nisu snimljeni tokom epileptičnog napada")

df2=df[df['y']==1].iloc[:5]
df2.T.iloc[:-1].plot(legend=None,ax=axis[1])
axis[1].title.set_text("Prvih 5 snimaka snimljenih tokom epileptičnog napada")
plt.rcParams["figure.figsize"] = (20,20)
plt.show()
plt.style.use('default') """

# %% [markdown]
# Poređenje modela dobijenih različitim izdvajanjem osobina

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
# Podela na trening i test skup podataka i skaliranje podataka

# %%
stat_x_train, stat_x_test, stat_y_train, stat_y_test = train_test_split(stat_features.drop(columns=['y']), stat_features['y'], train_size=0.8, random_state=42, shuffle=True)

scaler = StandardScaler()
stat_x_train = scaler.fit_transform(stat_x_train)
stat_x_test = scaler.transform(stat_x_test)

# %% [markdown]
# Pronalaženje najboljih parametara za SVC algoritam

# %%
svc=SVC()
svc.fit(stat_x_train,stat_y_train)
y_pred=svc.predict(stat_x_test)
print(f"Rezultati nad test podacima sa podrazumevanim parametrima: {f1_score(stat_y_test,y_pred)}")

stat_gs_result=find_best_svc_params(stat_x_train,stat_y_train,'stat_gs_result.pkl',True)
print(f"Pronađeni parametri: {stat_gs_result.best_params_}")
print(f"Rezultati nad test podacima sa pronađenim parametrima {stat_gs_result.score(stat_x_test,stat_y_test)}")

# %% [markdown]
# Autoregresioni model (Autoregressive model)

# %%
def find_ar_params(order):
    rows=[]
    for i in range(len(x)):
        model=AutoReg(x.iloc[i].values,lags=order).fit()
        row={}
        for j in range(1,order+1):
            row[f"param {j}"]=model.params[j]
        row['y']=y.iloc[i]
        rows.append(row)
    ar_model_params=pd.DataFrame(rows)
    return ar_model_params

start_order=6
end_order=15

ar_params=[find_ar_params(i) for i in range(start_order,end_order+1)]

# %% [markdown]
# Podela na trening i test skup podataka i skaliranje podataka

# %%
ar_params_split=[]

for order in ar_params:
    row={}
    x_train, x_test, y_train, y_test = train_test_split(order.drop(columns=['y']), order['y'], train_size=0.8, random_state=42, shuffle=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    row['x_train']=x_train
    row['x_test']=x_test
    row['y_train']=y_train
    row['y_test']=y_test
    ar_params_split.append(row)
    

# %% [markdown]
# Pronalaženje najboljeg reda autoregresivnog modela

# %%
i=start_order
for order in ar_params_split:
    print(f"Autoregresivni model {i}. reda:")
    svc=SVC()
    svc.fit(order['x_train'],order['y_train'])
    y_pred=svc.predict(order['x_test'])
    print(f"Rezultati nad test podacima sa podrazumevanim parametrima: {f1_score(order['y_test'],y_pred)}")
    i+=1

# %%
ar_x_train=ar_params_split[3]['x_train']
ar_x_test=ar_params_split[3]['x_test']
ar_y_train=ar_params_split[3]['y_train']
ar_y_test=ar_params_split[3]['y_test']

print(ar_x_train)
print(ar_x_test)
print(ar_y_train)
print(ar_y_test)

ar_gs_result=find_best_svc_params(ar_x_train,ar_y_train,'ar_gs_result.pkl',False)
print(f"Pronađeni parametri: {ar_gs_result.best_params_}")
print(f"Rezultati nad test podacima sa pronađenim parametrima {ar_gs_result.score(ar_x_test,ar_y_test)}")

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
""" y_plt=f_transform.drop(columns=['y']).iloc[0]
xf = rfftfreq(178, 1 / 178)

plt.plot(xf, np.abs(y_plt))
plt.title("Amplitude frekvencija dobijenih Furieovom transformacijom jednog EEG signala")
plt.xlabel("Frekvencija (Hz)")
plt.ylabel("Amplituda")
plt.show()
 """
# %% [markdown]
# Kako ne utiču sve frekvenicje pojednako, neke se mogu zanemariti. To ćemo postići pimenom PCA algoritma.

# %% [markdown]
# Prvo treba normalizovati i centrirati podatke

# %%
scaler = StandardScaler(with_mean=True, with_std=True)
x_scaled = scaler.fit_transform(abs(f_transform.drop(columns=['y'])))
f_transform_scaled = pd.DataFrame(x_scaled, columns=f_transform.drop(columns=['y']).columns)
f_transform_scaled['y']=f_transform['y'].values

f_transform_scaled.head()

# %%
f_transform_pca = PCA(n_components=20, random_state=42)
f_transform_components = f_transform_pca.fit_transform(f_transform_scaled.drop(columns=['y']))
print(f'ukupna varijansa: {sum(f_transform_pca.explained_variance_ratio_) * 100:.1f}%')

#plot_explained_variance(f_transform_pca)

# %%
f_transform_pca = PCA(n_components=5, random_state=42)
f_transform_components = f_transform_pca.fit_transform(f_transform_scaled.drop(columns=['y']))
print(f'ukupna varijansa: {sum(f_transform_pca.explained_variance_ratio_) * 100:.1f}%')

#plot_explained_variance(f_transform_pca)

# %% [markdown]
# Podela na trening i test skup podataka nad različitim brojevima komponenti PCA algoritma

# %%
ft_components_split=[]
for components_count in range(1,6):
  f_transform_pca = PCA(n_components=components_count, random_state=42)
  f_transform_components = f_transform_pca.fit_transform(f_transform_scaled.drop(columns=['y']))

  row={}
  x_train, x_test, y_train, y_test = train_test_split(f_transform_components, y, train_size=0.8, random_state=42, shuffle=True)
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)

  row['x_train']=x_train
  row['x_test']=x_test
  row['y_train']=y_train
  row['y_test']=y_test
  ft_components_split.append(row)

# %% [markdown]
# Pronalaženje najboljeg broja komponenti PCA algotirma za SVC algoritam

# %%
i=1
for components in ft_components_split:
    print(f"Broj PCA komponenti: {i}")
    svc=SVC()
    svc.fit(components['x_train'],components['y_train'])
    y_pred=svc.predict(components['x_test'])
    print(f"Rezultati nad test podacima sa podrazumevanim parametrima: {f1_score(components['y_test'],y_pred)}")
    i+=1

# %% [markdown]
# Pronalaženje najboljih parametara za SVC algoritam

# %%
ft_x_train=ft_components_split[1]['x_train']
ft_x_test=ft_components_split[1]['x_test']
ft_y_train=ft_components_split[1]['y_train']
ft_y_test=ft_components_split[1]['y_test']

ft_gs_result=find_best_svc_params(ft_x_train,ft_y_train,'ft_gs_result.pkl',False)
print(f"Pronađeni parametri: {ft_gs_result.best_params_}")
print(f"Rezultati nad test podacima sa pronađenim parametrima {ft_gs_result.score(ft_x_test,ft_y_test)}")

