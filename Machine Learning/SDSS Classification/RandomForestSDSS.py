"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Phase:      Feature engineering and ML classification

Algorithm:  Random Forest

Steps:      1) Import libraries
            2) Read, shuffle, and partition data
            3) Restructure data as inputs for DNN
            4) Feature Engineering
            5) Create and train DNN
            6) Make predictions on validation sets
            7) Fine-tune models for highest performance on validation set
            8) Make predictions on test set
            9) Evaluate model with confusion matrix

See full commentary at:
    https://www.kaggle.com/code/ktrinh/sdss-classification-with-random-forests-99-2/notebook

@author:    Kevin Trinh
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


# read in and shuffle SDSS data
filename = 'SDSS_data.csv'
sdss_df = pd.read_csv(filename, encoding='utf-8')
sdss_df = sdss_df.sample(frac=1)

# drop physically insignificant columns
sdss_df = sdss_df.drop(['objid', 'specobjid', 'run', 'rerun', 'camcol',
                        'field'], axis=1)


# partition SDSS data (60% train, 20% validation, 20% test)
train_count = 60000
val_count = 20000
test_count = 20000

train_df = sdss_df.iloc[:train_count]
validation_df = sdss_df.iloc[train_count:train_count+val_count]
test_df = sdss_df.iloc[-test_count:]


# obtain feature dataframes
X_train = train_df.drop(['class'], axis=1)
X_validation = validation_df.drop(['class'], axis=1)
X_test = test_df.drop(['class'], axis=1)


# encode labels as arbitrary integer classes
le = LabelEncoder()
labels = le.fit_transform(sdss_df['class'])

y_train = labels[:train_count]
y_validation = labels[train_count:train_count+val_count]
y_test = labels[-test_count:]

# scale features
scaler = StandardScaler()
scaler.fit(X_train) # fit scaler to training data only
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_validation.columns)

# apply principal component analysis to wavelength intensities
pca = PCA(n_components=3)
dfs = [X_train, X_validation, X_test]
for i in range(len(dfs)):
    df = dfs[i]
    ugriz = pca.fit_transform(df[['u', 'g', 'r', 'i', 'z']])
    df = pd.concat((df, pd.DataFrame(ugriz)), axis=1)
    df.rename({0: 'PCA1', 1: 'PCA2', 2: 'PCA3'}, axis=1, inplace=True)
    df.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)
    dfs[i] = df
X_train, X_validation, X_test = dfs

# create a random forest model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
preds = rfc.predict(X_validation)
model_acc = (preds == y_validation).sum().astype(float) / len(preds) * 100

print('Validation Accuracy: %3.5f' % (model_acc))