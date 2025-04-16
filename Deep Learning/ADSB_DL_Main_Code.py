######################
##  Imports         ##
######################

# Python
import time
import os

# Clear Terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Data 
import numpy as np
import pandas as pd

# Scikit-Learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt

# TensorFlow
import keras

# Models
import tab_to_img
import models


######################
##  Definitions     ##
######################

# Tabular to Image Conversion Methods
NO_CONV = "no-conv"
DWTM = "dwtm"
BIE = "bie"

# Models
ALEX_NET = "AlexNet"
RES_NET_50 = "ResNet50"
VGG_16 = "vgg16"
EFFICIENT_NET_B0 = "EfficientNetB0"
TAB_NET = "TabNet"
SIMPLE_3_LAYER_1D_CNN = "Simple-3-Layer-1D-CNN"
SIMPLE_6_LAYER_1D_CNN = "Simple-6-Layer-1D-CNN"


######################
##  Dataset         ##
######################

# Load the data set
data = pd.read_csv("10-airports-red.csv")

# Convert To Image
conv_method: str = NO_CONV

if conv_method == DWTM:
    X, y = tab_to_img.dwtm(data)
elif conv_method == BIE:
    X, y = tab_to_img.bie(data, repeat=6)
elif conv_method == NO_CONV:
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
else:
    raise Exception("Invalid Conversion Method Chosen.")

print(f"{conv_method} Data Shape: {X[0].shape}")

# Convert labels to one-hot encoding
y = keras.layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")(y).numpy()



######################
##  Model           ##
######################

# Select the model
model_name = TAB_NET
input_shape = X[0].shape
nb_classes = 4

if model_name == ALEX_NET:
    model = models.alex_net(input_shape, nb_classes)
elif model_name == VGG_16:
    model = models.vgg_16(input_shape, nb_classes)
elif model_name == RES_NET_50:
    model = models.res_net_50(input_shape, nb_classes)
elif model_name == EFFICIENT_NET_B0:
    model = models.efficient_net_b0(input_shape, nb_classes)
elif model_name == TAB_NET:
    model = models.tab_net(input_shape, nb_classes)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
elif model_name == SIMPLE_3_LAYER_1D_CNN or model_name == SIMPLE_6_LAYER_1D_CNN:
    if conv_method is not NO_CONV:
        raise Exception(f"Invalid Conversion Method for 1D CNN: {conv_method}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(-1, X.shape[1], 1)
    input_shape = X[0].shape
    if model_name == SIMPLE_3_LAYER_1D_CNN:
        model = models.simple_3_layer_1d_cnn(input_shape, nb_classes)
    else:
        model = models.simple_6_layer_1d_cnn(input_shape, nb_classes)
else:
    raise Exception("Invalid Model Name.")

# Split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=345)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model
model.summary()


######################
##  Training        ##
######################

# Enable Early Stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, start_from_epoch=8, restore_best_weights=True)

train_start = time.time()

history = model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=64, 
    epochs=100,
    verbose=1,
    validation_split=0.1,
    callbacks=[early_stopping_cb]
)

train_stop = time.time()



######################
##  Validation      ##
######################

val_start = time.time()

y_pred = model.predict(X_test)

val_stop = time.time()



######################
##  Evaluate        ##
######################

# Reformat y
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate TP, TN, FP, FN
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (tp + fp + fn)

# Calcuate Average Metrics
acc = metrics.accuracy_score(y_test, y_pred) * 100
pr = metrics.precision_score(y_test, y_pred, average='weighted')
rc = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

# Calculate Class Specific Metrics
dr = (tp / (tp + fn)) * 100
mdr = (fn / (tp + fn)) * 100
far = (fp / (fp + tn)) * 100

# Calculate Training and Validation Times
train_time = (train_stop - train_start) * 1000
val_time = (val_stop - val_start) * 1000


######################
##  Save Results    ##
######################

# Create Dataframe of Metrics
m = {
    "ACC":f'{acc:.2f}',
    "PR":f'{pr:.2f}',
    "RC":f'{rc:.2f}',
    "F1":f'{f1:.2f}',
    "DR-AU":f'{dr[0]:.2f}',
    "DR-PM":f'{dr[1]:.2f}',
    "DR-GI":f'{dr[2]:.2f}',
    "DR-VD":f'{dr[3]:.2f}',
    "MDR-AU":f'{mdr[0]:.2f}',
    "MDR-PM":f'{mdr[1]:.2f}' ,
    "MDR-GI":f'{mdr[2]:.2f}',
    "MDR-VD":f'{mdr[3]:.2f}',
    "FAR-AU":f'{far[0]:.2f}',
    "FAR-PM":f'{far[1]:.2f}',
    "FAR-GI":f'{far[2]:.2f}',
    "FAR-VD":f'{far[3]:.2f}',
    "TT":f'{train_time:.2f}',
    "VT":f'{val_time:.2f}' 
}

m = pd.DataFrame(m, index=[0])
print("Results:")
print(m)
print("\n")

# Define Result Paths
sep = '\\' if os.name == 'nt' else '/'
RESULTS_FOLDER = f"results{sep}"
METRICS_FOLDER = RESULTS_FOLDER + f"metrics{sep}"
MODELS_FOLDER = RESULTS_FOLDER + f"models{sep}"
MATRIX_FOLDER = RESULTS_FOLDER + f"confusion-matrix{sep}"

os.makedirs(METRICS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(MATRIX_FOLDER, exist_ok=True)

prefix = f"{model_name}_{conv_method}"

metrics_file = METRICS_FOLDER + f'{prefix}_metrics.csv'
model_file =  MODELS_FOLDER + f"{prefix}_model.keras"
matrix_file = MATRIX_FOLDER + f"{prefix}_confusion_matrix"

# Write Model Summary
from contextlib import redirect_stdout
with open(metrics_file, 'w') as file:
    with redirect_stdout(file):
        print(f"Conversion Method: {conv_method}")
        print(f"Input Shape: {input_shape}\n")
        model.summary()
        
        print("\nEpoch:\tTraining Loss  Validation Loss")
        for e, (tl, vl) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            print(f"{e}:\t{tl:.4f}\t{vl:.4f}")

        print(f"\tConfusion Matrix:")
        print(cm)

        print(f"\nMetrics:")
        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"TN: {tn}")
        print(f"FN: {fn}")


# Write Model Results
m.to_csv(metrics_file, index=False, mode='a')

# Save Model
model.save(model_file)

# Save Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Auth', 'PM', 'GI', 'VD'])
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
disp.plot(cmap=plt.cm.GnBu, colorbar=False)
plt.savefig(matrix_file, dpi=300, bbox_inches='tight')