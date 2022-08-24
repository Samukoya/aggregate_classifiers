import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn. model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def load_data(path_or_link)->pd.DataFrame:
    """
    Loads the data from source
    
    Parameters:
    -----------
    path_or_link : `str`, the path or link to the data
    
    Returns:
    --------
    pd.DataFrame : The loaded data"""
    return pd.read_excel(path_or_link, sheet_name='alle')

def clean_data(data):
    """
    Performs initial cleaning of the data, in this case adds the target labels
    And removes the 'wavelength(nm) column.
    """
    data['categories'].loc[0:299] = "harmless component"
    data['categories'].loc[300:466] = "water absorbent, part"
    data['categories'].loc[467:626] = "harmful component"
    data['categories'].loc[627:646] = "sensitive"
    data.drop('wavelength (nm)', axis=1, inplace=True)

def synthgen(data, N, k=5):
    """
    Uses the SMOTE oversampling technique to generate synthetic data from an existing dataset.
    
    Adapted from: https://stackoverflow.com/questions/55027404/generate-larger-synthetic-dataset-in-python
    
    Parameters:
    -----------
    data : array-like, shape = (n_samples,n_features)
           Data to be enlarged
    N    : number of times by which to enlarge the data
    k    : int, default=5
           number of nearest neighbours
    
    Returns:
    --------
    S    : array, shape=(N*n_samples, n_features)
           Enlarged data
        """
    n_samples, n_features = data.shape

    n_synthetic_sammples = N * n_samples
    S = np.zeros(shape=(n_synthetic_sammples, n_features))

    # Learn nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    neigh.fit(data)

    # Calculate synthetic samples
    for i in range(n_samples):
        nn = neigh.kneighbors(data[i].reshape(1,-1), return_distance=False)
        for n in range (N):
            index = choice(nn[0])
            # to avoid selecting T[i] from nn
            while index == i:
                index = choice(nn[0])
            
            diff = data[index] - data[i]
            gap = np.random.random()
            S[n+i *N, :] = data[i,:] + gap * diff[:]
    return S

def extend_dataset(data:pd.DataFrame, n_times:int, target_col:str):
    """
    Increases an existing dataset's size by n+1 times.
    
    Only works with continous data.
    
    Includes the data from which samples are generated

    Parameters:
    -----------
    data : `pd.DataFrame`, data from which samples are to be generated.
    n_times : int , number of times the data should be increased
    target_col : str , name of the column to be used as the Y column
    """
    # Dropping categorical colums other than the target.
    columns_to_drop = [column for column in data.columns if (data[column].dtype=='object') and column != target_col]
    data_copy = data.copy(deep=True)
    data_copy.drop(columns_to_drop, axis=1, inplace=True)

    # Dividing the dataframe into several dataframes depending on the categories
    harmless_df = data_copy[data_copy[target_col]=='harmless component']
    harmful_df = data_copy[data_copy[target_col]=='harmful component']
    waterad_df = data_copy[data_copy[target_col]=='water absorbent, part']
    sensitive_df = data_copy[data_copy[target_col]=='sensitive']

    df_list = [harmful_df, harmless_df, sensitive_df, waterad_df]
    extended_df = data_copy.copy(deep=True)

    # For each dataframe, enlarge by the number of times then add to the original dataset
    for df in df_list:
        X = df.drop(target_col, axis=1)
        component = df[target_col].unique()[0]
        # Convert the X to an array 
        X_np = X.to_numpy()
        X_ex = synthgen(X_np, n_times)
        df_ex = pd.DataFrame(X_ex, columns=X.columns)
        # Add the target column to the enlarged data
        df_ex[target_col] = [component for i in range(X_ex.shape[0])]
        extended_df = extended_df.append(df_ex, ignore_index=True)
    
    return extended_df

def split(data):
    """
    Splits the data into train, validation and test sets
    
    Parameters:
    -----------
    data : DataFrame or ndarray, the data to be split

    Returns: 
    --------
    split_data : list, a list containing the split data
    """

    # Splitting the original data to remain with samples for use in tests later on
    # 30% of the data is held out for final testing
    df_train, df_test = train_test_split(data, test_size=0.3, random_state=42)
    # increase the training Dataset by 40 times to get a significant number of samples for the models
    data_extended = extend_dataset(df_train, 40, 'categories')
    # dividing the data to X, and Y, and training and validation sets
    X = data_extended.drop('categories', axis=1)
    y = data_extended[['categories']]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)

    # Test data
    X_test = df_test.drop('categories', axis=1)
    y_test = df_test['categories']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

class MLClassifiers():
    """
    This compares the multiclass classification accuracies of several Machine learning algorithms.
    The algorithms compared are Random Forests, Support vector machine, and Multi Layer Perceptron.

    Parameters:
    -----------
    data_path : str, the path to the data.
                could be a directory or a link

    n_times : The number of times to enlarge the data, defalt=40
              The training data will be increased by n_times + 1
    
    """
    def __init__(self, data_path:str, n_times=40, grid_search=False) -> None:
        self.data_path = data_path
        self.n_times = n_times
        self.grid_search = grid_search
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.preprocess()
        self.rfc = RandomForestClassifier(n_jobs=-1)
        self.svc = SVC()
        self.mlp = MLPClassifier() 

    def preprocess(self):
        # Load data
        df = load_data(self.data_path)

        #clean the data
        clean_data(df)

        # Extends and splits the data, then returns the split data
        return split(df)

    def baseline_fit(self):
        """
        Compares the baseline scores using the training and validation data
        """
        algs = [self.svc, self.rfc, self.mlp]
        for alg in algs:
            alg.fit(self.X_train, self.y_train)
            print(f"Model : {alg} Validation accuracy: {alg.score(self.X_valid, self.y_valid)}")
        return algs
    
    def grid_search(self):
        """
        Using GridSearchCV to obtain the optimal hyperparameters
        May take a really long while
        """
        rfc_params = {
            'n_estimators':(50, 100, 150),
            'criterion':('gini', 'entropy'),
            'class_weight':(None, 'balanced')
        }
        svc_params = {
            'C':(0.1, 0.5, 1, 10),
            'kernel':('rbf', 'linear'),
            'class_weight':(None, 'balanced')
        }
        mlp_params = {
            'hidden_layer_sizes':((100,), (50,), (150,)),
            'activation': ('relu', 'tanh', 'logistic'),
            'solver': ('adam', 'sgd')

        }

        algs = [self.svc, self.rfc, self.mlp]
        params = [svc_params, rfc_params, mlp_params]
        best_estimators = [] # To store the best estimators

        for alg, param in zip(algs, params):
            grid_search = GridSearchCV(alg, param, cv=5, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            best_estimators.append(grid_search.best_estimator_)
        return best_estimators

    def predict(self, data):
        """
        Uses the fitted algorithms to predict the classes of the given data
        """
        # Convert the data to a dataframe if it isn't 
        if data.dtype() != pd.DataFrame:
            data = pd.DataFrame(data)
            columns_to_drop = [column for column in data.columns if (data[column].dtype=='object')]
            data.drop(columns_to_drop, axis=1, inplace=True)
            if len(data.columns)!= self.X_train.columns:
                raise ValueError(f"Invalid data, expected {self.X_train.columns} continous columns")
            
        if not self.grid_search:
            for alg in self.baseline_fit():
                print(f"{alg}'s prediction: {alg.predict(data)}")
        else:
            # Alternatively, uncomment to use the best estimators from GridsearchCV
            for alg in self.grid_search():
                print(f"{alg}'s prediction: {alg.predict(data)}")
            
data_path = "C:\\Users\\pc\\Downloads\\dataset(1).xlsx" # Change it to the directory of the data 
model = MLClassifiers(data_path) # Leaving all other parameters to default
model.baseline_fit()