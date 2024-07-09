import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the data
df = pd.read_csv('cleaned_dataset.csv')

# Create the feature matrix and target variable
X = df.drop("Churn", axis=1) # Features
y = df["Churn"].map({'No': 0, 'Yes': 1}) # Target variable

from sklearn.preprocessing import OneHotEncoder

#  Define the columns to be encoded
columns_to_encode = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Select the columns to be encoded from the DataFrame
X_to_encode = X[columns_to_encode]

# Initialize the OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)  # Use sparse=False to get a dense matrix

# Fit and transform the selected columns
encoded_columns = cat_encoder.fit_transform(X_to_encode)

# Create new column names for the encoded values
new_columns = cat_encoder.get_feature_names_out(columns_to_encode)

# Create a DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_columns, columns=new_columns)

# Drop the original columns from X
X = X.drop(columns=columns_to_encode)

# Concatenate the original DataFrame with the encoded DataFrame
X = pd.concat([X, encoded_df], axis=1)

# install tensorflow



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ... (Your code for data loading, preprocessing, and scaling is perfect!)

# Splitting the data (make sure you have scaled data here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

# Define the FNN model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],))) 
model.add(Dropout(0.2))  # Regularization
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Generate predictions on the test set
y_pred = model.predict(X_test) 

# Convert probabilities to class labels (assuming threshold of 0.5) 
y_pred = np.where(y_pred > 0.5, 1, 0) 

# Calculate additional metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred,zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)   

print('Model accuracy:', accuracy)
print('Model precision:', precision)
print('Model recall:', recall)
print('Model F1 score:', f1)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# calculate the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# calculate the metrics for calss 0 and class 1

# Model precision for class 1 which is 'Yes'
precision = precision_score(y_test, y_pred, pos_label=1)
print('Model precision for class 1:', precision)

# Model recall for class 1 which is 'Yes'
recall = recall_score(y_test, y_pred, pos_label=1)
print('Model recall for class 1:', recall)
      
# Model F1 score for class 1 which is 'Yes'
f1 = f1_score(y_test, y_pred, pos_label=1)
print('Model F1 score for class 1:', f1)

# Model precision for class 0 which is 'No'
precision = precision_score(y_test, y_pred, pos_label=0)
print('Model precision for class 0:', precision)

# Model recall for class 0 which is 'No'
recall = recall_score(y_test, y_pred, pos_label=0)

print('Model recall for class 0:', recall)
# Model F1 score for class 0 which is 'No'
f1 = f1_score(y_test, y_pred, pos_label=0)
print('Model F1 score for class 0:', f1)

