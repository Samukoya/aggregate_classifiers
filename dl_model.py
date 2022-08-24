from imp import load_module
from tensorflow import keras 
import pandas as pd
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from keras.utils.np_utils import to_categorical
import warnings
import matplotlib.pyplot as plt
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

def dim_reduction_pls(data, n_comp):
    xtr = data.drop('categories', axis=1)
    ytr = data[['categories']]
    # One hot encoding the target
    encoded_y = pd.get_dummies(ytr)
    pls = PLSRegression(scale=False, n_components=n_comp)
    # Fitting the data
    X_train = pls.fit_transform(xtr, encoded_y)[0]
    return X_train, ytr, pls

def load_and_preprocess(data_path, dim_reduce=True, balance=False, n_comp=180, extend=False, n_times=5, scale=False): 
    
    # Load the raw data 
    df = load_data(data_path)

    # Preliminary cleaning
    clean_data(df)
    # Separation of test and train sizes
    df_train, df_test = train_test_split(df, train_size=0.7)

    if extend:
        df_train = extend_dataset(df_train, n_times-1, 'categories')

    y_test = df_test['categories']
    X_test = df_test.drop('categories', axis=1)
    
    X_train = df_train.drop('categories', axis=1)
    y_train = df_train[['categories']]
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    if balance:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(n_jobs=-1)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        #df_train2 = pd.concat([xtr_sm, ytr_sm])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    pls = None
    
    if dim_reduce:
        X_train, y_train, pls = dim_reduction_pls(df_train, n_comp)
        X_valid = pls.transform(X_valid)
        X_test = pls.transform(X_test)
    
    

    if scale:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
    
    enc = LabelEncoder()    
    y_train = enc.fit_transform(y_train)
    y_valid = enc.transform(y_valid)
    y_test = enc.transform(y_test)
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, pls
    

class DLClassifier():
    """
    Uses a feedfoward neural net from Keras to classify the data.

    Parameters:
    -----------
    data_path : str, the path to the data.
                could be a directory or a link

    n_times : The number of times to enlarge the data, defalt=5
              The training data will be increased by n_times
    
    dim_reduce : bool, default=True
                 whether or not to perform dimensionality reduction
    
    balance : bool, default=False
              Whether to perform class balancing on the data
    """
    
    def __init__(self, data_path:str, n_times=5, dim_reduce=True, balance=False, n_comp=180, extend=True, scale=False) -> None:
        self.data_path = data_path
        self.n_times = n_times
        self.n_comp =n_comp
        self.balance = balance
        self.extend=extend
        self.dim_reduce= dim_reduce
        self.scale = scale
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.pls = self.preprocess()
        self.model = self.create_model()
        self.scaler = MinMaxScaler()
        

    def preprocess(self):
        """
        Loads, cleans, transforms and splits the data into training, validation and test sets.
        """
        return load_and_preprocess(self.data_path, dim_reduce=self.dim_reduce, balance=self.balance, n_comp=self.n_comp, extend=self.extend, n_times=self.n_times, scale=self.scale)


    def create_model(self):
        """Creates a base model"""
        model = keras.Sequential([
            layers.Dense(1024, activation='relu', input_shape=[self.X_train.shape[1]]),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'), 
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'), 
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(4, activation='softmax'),
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def train(self):
        """Train the model"""
        # Define early stopping specifications for the data
        early_stopping = EarlyStopping(
            min_delta=0.001, # minimium amount of change to count as an improvement
            patience=20, # how many epochs to wait before stopping
            restore_best_weights=True,
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            batch_size=128,
            epochs=100,
            callbacks=[early_stopping],
            verbose=1 # Set to 1 to see the training log
        )

        # Plots of the learning curves
        history_df = pd.DataFrame(history.history)
        history_df.loc[:, ['loss', 'val_loss']].plot();
        history_df.loc[:, ['accuracy', 'val_accuracy']].plot();

        # Save the model. Saves it to the same directory as the python file
        self.model.save('model.h5')

    def predict(self, data):
        """Uses the model to predict the new data."""
        # Convert the data to a dataframe if it isn't 
        if data.dtype() != pd.DataFrame:
            data = pd.DataFrame(data)
            columns_to_drop = [column for column in data.columns if (data[column].dtype=='object')]
            data.drop(columns_to_drop, axis=1, inplace=True)
            if len(data.columns)!= self.X_train.columns:
                raise ValueError(f"Invalid data, expected {self.X_train.columns} continous columns")
        if self.pls != None:
            data = self.pls.transform(data)
        if self.scale:
            data = self.scaler.fit_transform(data) # To be rectified
        model = load_model('model.h5')

        predictions = model.predict(data)
        print(f"Predictions: {predictions}")

data_path = "C:\\Users\\pc\\Downloads\\dataset(1).xlsx" # Change it to the directory of the data 
model = DLClassifier(data_path) # Leaving all other parameters to default
model.train() 