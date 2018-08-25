def run_GLM(x,y,ElasticNet=False,Lasso=False,Ridge=False):
    """
    Elastic net, Lasso, Ridge. Set True to use one of the functions; with Pandas and numpy imported
    """
    from sklearn.model_selection import GridSearchCV # search for the best lambda
    from sklearn import linear_model
    import pandas as pd
    import numpy as np
    
    if ElasticNet==True:
        EN= linear_model.ElasticNet(normalize=True) # create a ridge regression instance

        # find the best alpha (lambda) for ridge
        grid_param = [{'alpha': np.logspace(-4.5, 2, 50),'l1_ratio':np.linspace(0.1,0.9,20)}]
        para_search_EN = GridSearchCV(estimator=EN, param_grid=grid_param, scoring='neg_mean_squared_error', cv=5, return_train_score=True)
        para_search_EN.fit(x, y)

        print(para_search_EN.best_params_)
        print("Lowest RMSE found: ", np.sqrt(np.abs(para_search_EN.best_score_)))

        # fit best ridge equation to all train data
        best_EN_y = para_search_EN.best_estimator_.predict(x)
        print("RMSE: ", np.sqrt(np.mean((y-best_EN_y)**2)))

######    
    if Lasso == True:
        lasso= linear_model.Lasso(normalize=True) # create a ridge regression instance

        # find the best alpha (lambda) for ridge
        grid_param = [{'alpha': np.logspace(-4.5, 2, 100)}]
        para_search_lasso = GridSearchCV(estimator=lasso, param_grid=grid_param, scoring='neg_mean_squared_error', cv=5, return_train_score=True)
        para_search_lasso.fit(x, y)

        print(para_search_lasso.best_params_)
        print("Lowest RMSE found: ", np.sqrt(np.abs(para_search_lasso.best_score_)))

        # fit best ridge equation to all train data
        best_lasso_y = para_search_lasso.best_estimator_.predict(x)
        print("RMSE: ", np.sqrt(np.mean((y-best_lasso_y)**2)))

##############
    if Ridge==True:

        ridge = linear_model.Ridge(normalize=True) # create a ridge regression instance

        # find the best alpha (lambda) for ridge
        grid_param = [{'alpha': np.logspace(-4, 2, 100)}]
        para_search_ridge = GridSearchCV(estimator=ridge, param_grid=grid_param, scoring='neg_mean_squared_error', cv=5, return_train_score=True)
        para_search_ridge.fit(x, y)

        print(para_search_ridge.best_params_)
        print("Lowest RMSE found: ", np.sqrt(np.abs(para_search_ridge.best_score_)))

        # fit best ridge equation to all train data
        best_ridge_y = para_search_ridge.best_estimator_.predict(x)
        print("RMSE: ", np.sqrt(np.mean((y-best_ridge_y)**2)))

def run_RF(x,y,number_of_trees=1000, min_leaf=4,min_split=4,rs=1):
    """
    RandomForest. Input number of trees. rs is randomstate
    """
    from sklearn import ensemble
    from sklearn import model_selection
    import pandas as pd
    import numpy as np
    randomForest = ensemble.RandomForestRegressor()
    grid_para_forest = [{
        "n_estimators": [number_of_trees],
        "max_features": ["sqrt"],
        "criterion": ["mse"],
        "min_samples_leaf": [min_leaf],
        "min_samples_split": [min_split],
        "oob_score": [True],
        "random_state": [rs]}]
    grid_search_forest = model_selection.GridSearchCV(randomForest, grid_para_forest, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search_forest.fit(x, y)
    print(grid_search_forest.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(grid_search_forest.best_score_)))

     # fit best ridge equation to all train data
    best_rf_y = grid_search_forest.best_estimator_.predict(x)
    print("RMSE: ", np.sqrt(np.mean((y-best_rf_y)**2)))
    
def run_GB(x,y,number_of_trees=10000, min_split=4,learningrate=0.001, maxfeature=18):
    from sklearn import model_selection
    from sklearn.metrics import mean_squared_error
    from sklearn import ensemble
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor
    import pandas as pd
    import numpy as np
    xg=ensemble.GradientBoostingRegressor()

    grid_para_xg =[{'n_estimators': number_of_trees, 'max_depth': [4], 'min_samples_split':min_split, 
                 'learning_rate': learningrate, 'loss':['ls'], 'max_features':maxfeature}]

    #max depth 4 #min_sample:6
    #max_features 18
    grid_search_xg = model_selection.GridSearchCV(xg, grid_para_xg, scoring='neg_mean_squared_error', cv=3, return_train_score=True,  n_jobs=-1)
    #scoring:mse
    grid_search_xg.fit(x,y)
    print(grid_search_xg.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(grid_search_xg.best_score_)))

        # fit best ridge equation to all train data
    best_gb_y = grid_search_xg.best_estimator_.predict(x)
    print("RMSE: ", np.sqrt(np.mean((y-gb_y)**2)))

