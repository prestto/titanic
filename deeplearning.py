import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set(style="darkgrid")
from sklearn.model_selection import train_test_split

# Read in data
train = pd.read_csv('./train.csv')
train = train.set_index('PassengerId')
test = pd.read_csv('./test.csv')
test = test.set_index('PassengerId')

# Data processing functions to prep for deep learning
def add_mean_ages(input_df):
    avg_male = input_df['Age'][input_df['Sex'] == 'male'].mean()
    avg_female = input_df['Age'][input_df['Sex'] == 'female'].mean()
    
    input_df['Age'][(input_df['Age'].isnull()) & (input_df['Sex'] == 'male')] = avg_male
    input_df['Age'][(input_df['Age'].isnull()) & (input_df['Sex'] == 'female')] = avg_female
    
    return input_df

def add_mean_fare(input_df):
    avg_fare = int(input_df['Fare'].mean())
    input_df['Fare'].fillna(avg_fare, inplace=True)
    
    return input_df

def bin_nulls(input_df):
    df = input_df[input_df['Age'].notnull()]
    df = df.drop(['Cabin'], axis = 1)
    
    return df
    
def assign_port(port_code):
    """
    Add port names in place of ports.
    These port names will become column headers once the variable
    is converted to dummies
    """
    if port_code == 'S':
        return 'Southampton'
    elif port_code == 'C':
        return 'Cherbourg'
    elif port_code == 'Q':
        return 'Queenstown'
    else:
        return np.nan

def add_embark(input_df):
    port_embarked = input_df['Embarked'].apply(assign_port)
    port_embarkation = pd.get_dummies(port_embarked, prefix = 'Embarked', drop_first = True)
    
    df = pd.merge(input_df, port_embarkation, left_index=True, right_index=True)
    
    return df

def transform_dataframe(input_df):
    df = add_mean_ages(input_df)
    df = bin_nulls(df)
    df = add_embark(df)
    df = add_mean_fare(df)
    
    sex = pd.get_dummies(df['Sex'], drop_first = True)
    df = pd.merge(df, sex, left_index=True, right_index=True)
    
    classes = pd.get_dummies(df['Pclass'], prefix = 'class', drop_first = True)
    df = pd.merge(df, classes, left_index=True, right_index=True)
    
    df = df.drop(['Pclass', 'Embarked', 'Name', 'Sex', 'Ticket'], axis = 1)
    
    return df

processed_train = transform_dataframe(train)
X = processed_train.drop(['Survived'], axis = 1)
y = processed_train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state = 42)


X_unseen = transform_dataframe(test)


# Declare and fit the model using deep learning
# Deep learning approach 
# This only acheived a 72% accuracy score on kaggel :(

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import numpy as np


y_train_target = pd.get_dummies(y_train)

n_cols = X_train.shape[1]

model = Sequential()

model.add(Dense(100, activation = 'relu', input_shape=(n_cols,)))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

early_stop = EarlyStopping(patience=3)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# model_training = model.fit(X_train, y_train_target, validation_split=0.2, epochs = 50, callbacks=[early_stop])

# acc = np.array(model_training.history['acc'])
# loss = np.array(model_training.history['loss'])
# val_acc = np.array(model_training.history['val_acc'])
# val_loss = np.array(model_training.history['val_loss'])
# epochs = np.arange(1, len(model_training.history['acc']) + 1)

# plt.subplot(221)
# plt.title("Accuracy")
# plt.plot(epochs, acc)

# plt.subplot(222)
# plt.title("Loss")
# plt.plot(epochs, loss)

# plt.subplot(223)
# plt.title("Value Accuracy")
# plt.plot(epochs, val_acc)

# plt.subplot(224)
# plt.title("Value Loss")
# plt.plot(epochs, val_loss)

# plt.show()

model.fit(X_train, y_train_target, validation_split=0.2, epochs = 50, callbacks=[early_stop])


preds = model.predict(X_unseen)

# Add the predictions to the dataframe
X_unseen['Survived'] = preds.round()[:,1]
kaggle_output = X_unseen['Survived'].astype(int).rename('Survived')

# Output the file to csv: gender_submission.csv
kaggle_output.to_csv('gender_submission.csv', header=True)