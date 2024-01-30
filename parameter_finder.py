from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def find_best_svc_params(x_train,y_train):
    svc=SVC()

    c = [1.0,10.0,100.0,500.0,1000.0]
    gamma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
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
    
    return grid_search.fit(x_train,y_train),grid_search