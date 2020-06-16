from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Divide our dataset between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Load the model and fit it to the training data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict the category for the test data based on the trained model 
y_pred = model.predict(X_test)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = model.score(X_test,y_test)
print("Accuracy Linear: {}%".format(accuracy*100))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
# Plot confusion matrix using the plot_confusion_matrix function created in a separated file
plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix, classes=["Non-defalut", "Default"])
plt.savefig('confusion_matrix_raw.png', dpi=150)
plt.show()
