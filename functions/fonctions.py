from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
import pandas as pd
def prepare_data(data):
     data_columns= data.columns
     for i in data_columns:
        if data[i].dtype == 'int64' or data[i].dtype == 'float64':
         data[i]=  data[i].fillna(data[i].mean())
        elif data[i].dtype == 'object':
          data[i] = data[i].fillna(data[i].mode()[0])
     prepared_data = data.drop(columns=['Courier_Experience_yrs'], errors='ignore')
     return prepared_data
def encode_data(data):
    columns = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    # Create encoder with dense output
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    for col in columns:
        # Convert column to 2D and handle NaN
        transformed = encoder.fit_transform(data[[col]])
        # Convert to DataFrame with proper column names
        transformed_df = pd.DataFrame(
            transformed,
            columns=encoder.get_feature_names_out([col]),
            index=data.index
        ) 
        # Drop original column and add encoded columns
        data = pd.concat([data.drop(columns=[col]), transformed_df], axis=1)
    return data

def split_data(prepared_data):
  X= prepared_data.drop(columns='Delivery_Time_min', axis=1)
  y= prepared_data['Delivery_Time_min']
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
  return X_train, X_test, y_train, y_test,X,y

def select_k_best(splited_data):
   #f_regression Computes the correlation between each feature and the target.

   X_train, X_test, y_train, y_test,X,y= splited_data
   selector= SelectKBest(score_func=f_regression, k=5)
   selector.fit(X_train, y_train)
   featured_columns= X_train.columns[selector.get_support()]
   return featured_columns.values

param_grid = {
    'n_estimators': [70, 80, 100],
    'max_depth': [5, 10, None]
}

grid_search_random = GridSearchCV(
    RandomForestRegressor(),
    param_grid=param_grid,
    cv=5,             # K-Fold Cross Validation with 5 folds
)
svc_grid= {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [1, 10],
    'gamma': ['scale', 'auto', 0.01],
    'epsilon': [0.1] 
}
grid_search_svr = GridSearchCV(
    SVR(),
    param_grid=svc_grid,
    cv=5,
    verbose=2,
    n_jobs=-1# Use all CPU cores
    # K-Fold Cross Validation with 5 folds
     # use all CPU cores
)

data = pd.read_csv("C:\\Users\\hp\\Documents\\projects\\simplon_projects\\prediction_du_Temps_de_Livraison\\data\\\\dataset.csv")
prepared_data1= prepare_data(data)
encoded_data1=encode_data(prepared_data1)
def get_metrics(encoded_data, gridsearch): 
  X_train, X_test, y_train, y_test,X,y=split_data(encoded_data)
  selector = SelectKBest(score_func=f_regression, k=8)
  X_train_selected = selector.fit_transform(X_train, y_train)
  X_test_selected = selector.transform(X_test)
  print('gread search')
  gridsearch.fit(X_train_selected,y_train)
  y_pred1= gridsearch.best_estimator_.predict(X_test_selected)
  print("MAE:", mean_absolute_error(y_test, y_pred1))
  MAE=mean_absolute_error(y_test, y_pred1)
  print("R2_score:", r2_score(y_test, y_pred1))
  R2_score=r2_score(y_test, y_pred1)
  best_score=gridsearch.best_score_
  print("best_score", gridsearch.best_score_)
  print("best_estimator",gridsearch.best_estimator_)
  best_estimator=gridsearch.best_estimator_
  print("best_params",gridsearch.best_params_)
  best_params=gridsearch.best_params_
  return MAE, R2_score,best_score,best_estimator,best_params

print("Random forest metrics")
# print(get_metrics(encoded_data1,grid_search_random))
# print("SVR metrics",get_metrics(encoded_data1,grid_search_svr))

def normalisation(prepared_data):
     scaler= StandardScaler()
     columns= ['Distance_km', 'Preparation_Time_min', 'Delivery_Time_min', 'Courier_Experience_yrs']
     for i in columns:
        prepared_data[i]= scaler.prepared_data[i]
     return prepared_data


#A pipeline allows us to assemble several steps that can be cross-validated 
# together while setting different parameters. This ensures that all steps are performed sequentially 
# and that the transformations are applied only to the training data within each cross-validation fold.
data = pd.read_csv("C:\\Users\\hp\\Documents\\projects\\simplon_projects\\prediction_du_Temps_de_Livraison\\data\\\\dataset.csv")
prepared_data= prepare_data(data)

encoded_data= encode_data(prepared_data)
splited_data=split_data(encoded_data)
print(select_k_best(splited_data))
#Creation d'un pipeline
#1. ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
def data_preprocessing():
      prepared_data= prepare_data(data)
      X= prepared_data.drop(columns=['Delivery_Time_min','Order_ID'])
      y= prepared_data['Delivery_Time_min']
      X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=42)
      return X_train, X_test, y_train, y_test,X,y

X = prepared_data.drop(columns=['Delivery_Time_min'])  # features only
y = prepared_data['Delivery_Time_min']                 # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_num= ['Distance_km', 'Preparation_Time_min']
X_cat=['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
preprocessor= ColumnTransformer(
   transformers= [
      ('num', StandardScaler(), X_num),
      ('cat', OneHotEncoder(), X_cat)
   ]
)

#2.define the pipeline of RandomForestRegressor
pipeline= Pipeline([
   ('preprocessor', preprocessor),
   ('feature_selection', SelectKBest(score_func=f_regression, k=5)),
   ('regression', RandomForestRegressor())
])
preprocessor.fit_transform(data)
#3.define the parameter grid
grid = { 
    'regression__n_estimators': [70,80,90,100],
    'regression__random_state' : [18]
}
#function
def gridsearch_metrics(pipline_param, gridparam ):
    grid_search_cv=GridSearchCV(pipline_param, gridparam, cv=5,n_jobs=-1, scoring='r2')
    grid_search_cv.fit(X_train, y_train)
    y_pred_rf= grid_search_cv.predict(X_test)
    mae= mean_absolute_error(y_test, y_pred_rf)
    print("MAE:", mean_absolute_error(y_test, y_pred_rf))
    r2=r2_score(y_test, y_pred_rf)
    print("R2_score:", r2_score(y_test, y_pred_rf))
    return  f'MAE:{mae} \n R2_score: {r2}'
#

# grid_search_cv=GridSearchCV(pipeline, grid, cv=5,n_jobs=-1, scoring='r2')
# grid_search_cv.fit(X_train, y_train)
# y_pred_rf= grid_search_cv.predict(X_test)
# print("Random forest regression metrics")
# print("MAE:", mean_absolute_error(y_test, y_pred_rf))
# print("R2_score:", r2_score(y_test, y_pred_rf))

#4.define the pipeline of SVR
pipeline_svr= Pipeline([
   ('preprocessor', preprocessor),
   ('feature_selection', SelectKBest(score_func=f_regression, k=5)),
   ('svr', SVR())
])
#5.define the parameter grid
grid_svr = { 
    'feature_selection__k': [5, 8],          # fewer options
    'svr__kernel': ['linear', 'rbf'],        # remove 'poly' if slow
    'svr__C': [1, 10],
    'svr__gamma': ['scale', 'auto'],
    'svr__epsilon': [0.1]
}
print("Random forest regression metrics :")
gridsearch_metrics(pipeline, grid)
print("SVR metrics :")
gridsearch_metrics(pipeline_svr, grid_svr)

# grid_search_cv=GridSearchCV(pipeline_svr, grid_svr, cv=5,n_jobs=-1,scoring='r2')
# grid_search_cv.fit(X_train, y_train)
# y_pred_svr= grid_search_cv.predict(X_test)
# print('SVR metrics:')
# print("MAE : ", mean_absolute_error(y_test, y_pred_svr))
# print("R2_score: ", r2_score(y_test, y_pred_svr))

# # # #use GridSearchCV to perform hyperparameter tuning on the entire pipeline
# # # # rf_cv = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv= 5)
