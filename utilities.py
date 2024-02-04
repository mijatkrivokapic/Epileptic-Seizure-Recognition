import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle

def plot_correlation_for_col(df, col_name):
    plt.figure(figsize=(12,6)) # podesimo velicinu grafika
    correlation_matrix = df.corr() # racunamo matricu korelacije
    sorted_col_corr = correlation_matrix[col_name].sort_values(ascending=True) # indeksiramo kolonu i soritramo vrednosti
    sorted_col_corr = sorted_col_corr.drop(col_name) # izbacujemo vrednost samu sa sobom
    sb.barplot(x=sorted_col_corr.index, y=sorted_col_corr.values, palette='RdBu')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_explained_variance(pca_model):
    '''Plots the explained variance plot using a trained PCA model.'''
    plt.figure(figsize=(9,3)) # podesimo velicinu grafika
    
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, align='center')
    plt.xlabel('Glavna komponenta')
    plt.ylabel('Objasnjena varijansa')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, '--o')
    plt.xlabel('Broj glavnih komponenti')
    plt.ylabel('Kumulativna varijansa')

    plt.tight_layout()
    plt.show()

def find_best_svc_params(x_train,y_train,filename,load_from_file):
    svc=SVC()

    if load_from_file:
        with open(filename, 'rb') as f:
            grid_search = pickle.load(f)
        return grid_search

    c = [0.01, 0.1, 1.0, 10.0, 90, 100.0, 110]
    gamma = [0.001, 0.001, 0.01, 0.1, 1]
    degree = [2,3,4]
    param_grid=[{'C': c,'kernel': ['linear']},
                {'C': c,'kernel': ['rbf'],'gamma': gamma} ,
                {'C': c,'kernel': ['poly'],'gamma': gamma,'degree': degree}
            ]

    grid_search = GridSearchCV(estimator=svc, 
                            param_grid=param_grid, 
                            scoring='f1', 
                            refit=True, 
                            n_jobs=-1, 
                            verbose=4)
    
    grid_search.fit(x_train,y_train)

    with open(filename, 'wb') as f:
        pickle.dump(grid_search, f)

    return grid_search