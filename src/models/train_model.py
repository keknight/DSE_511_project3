import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier


#TODO: get preprocessed data
# load dataset
data = pd.read_csv("preprocessed_data.csv")


#Test, training, and validation data for just the cleaned data
#need to determine what are relevant features, as well as the target field 

X_train, X_test, y_train, y_test = train_test_split(data[['feature1', 'feature2']], data['target'], 
                                   test_size=0.15, shuffle=True, stratify=data['target'], random_state=42)

#get validation data for hyperparameter setting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                 test_size=0.15/0.85, shuffle=True, stratify=y_train, random_state=42)

print('original:', data[['feature1', 'feature2']].shape, data['target'].shape)
print('train:', X_train.shape, y_train.shape)
print('val:', X_val.shape, y_val.shape)
print('test:', X_test.shape, y_test.shape)

#Decision tree 

#Decision tree parameters
#['ccp_alpha', 'class_weight', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 
#	'min_impurity_split', 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'random_state', 'splitter']

#assign classifier 
clf_dt = DecisionTreeClassifier()


#create a grid of parameters to test 
parameters_dt = {
    'ccp_alpha': (1.0, 1.5, 2.0),
    'class_weight': (None, 5000, 10000, 50000),
    'criterion': ((1, 1), (1, 2)),  # unigrams or bigrams
    'max_depth': (True, False),
    'max_features': (0.5, 1.0, 1.5, 2.0),
    'max_leaf_nodes': (True, False),
}

#GridSearchCV (cross validation + hyperparameter tuning)
dt_gridsearch = GridSearchCV(clf_dt, parameters_dt, cv=10, scoring='accuracy')




clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#XGBoost 


#Use best model to predict test data


