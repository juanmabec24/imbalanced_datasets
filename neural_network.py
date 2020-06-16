import keras
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import f1_score, log_loss, roc_auc_score

import itertools

# Define the model architecture
model = Sequential()

# Calculate the positive to negative ratio to take advantage of the bias_initializer hyperparameter
pos = 1000
neg = 99000
initial_bias = np.log([pos/neg])
output_bias = keras.initializers.Constant(initial_bias)

model.add(Dense(32, activation='relu',
          input_shape=(X_train.shape[-1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='sigmoid', bias_initializer=output_bias))
# print the model architecture
model.summary()

# Define the set of metrics we want to use
metrics = [ 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(curve='PR', name='pr')
]

# Compile the model using the most appropriate loss for this imbalanced binary classification problem
model.compile(loss='binary_crossentropy', optimizer= Adam(lr=3e-3), metrics=metrics)

# Fit the model using the hyperparameter "class_weight" to give more weight to the default samples (loan_status = 1)
history = model.fit(
    X_train, y_train,
    batch_size=256,
    epochs=5,
    validation_data=(X_test, y_test), 
    verbose=2,
    class_weight={0: 1, 1: 110})

y_pred_nn = model.predict(X_test)

# Define the threshold used to classify a sample as positive or negative
threshold = 0.5

# Compute confusion matrix
cnf_matrix_nn = confusion_matrix(y_test, y_pred_nn>threshold)

#Compute f1 score
f1_score_nn = f1_score(y_pred_nn>threshold, y_test)
print("F1 score tree: {}".format(f1_score_nn))

# Plot non-normalized confusion matrix
plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix_nn, classes=["Non-defalut", "Default"])
plt.show()
