import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("HW05_Abominable_Training____Data_v2205_n500.csv")

# replaces Assam with 0 and Bhutan with 1
class_id = []
for c in data["Class"].tolist():
    if c == "Assam":
        class_id.append(0)
    else:
        class_id.append(1)

del data["Class"]


mean_digit = np.mean(data.T, axis=1)            # mean of columns
centered_data = data - mean_digit
covariance = np.cov(centered_data.T)            # covariance
(eigenvalues, eigenvectors) = eig(covariance)
print(eigenvalues)
xy = np.round(eigenvectors, decimals=2)
print(xy.T)

# idx = np.argsort(eigenvalues)[::-1]
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:, idx]
# print("eigen values", eigenvalues)
# print("eigen vectors", eigenvectors)

y = []
ev_cum = 0
for ev in eigenvalues:
    ev_cum += ev
    y.append(ev_cum/np.sum(eigenvalues))

x = [1, 2, 3, 4, 5, 6, 7]
plt.plot(x, y)
plt.xlabel("Number of eigen vectors")
plt.ylabel("Cumulative amount of variance captured")
plt.show()


#  plotting of all possible combinations of eigen vectors
for i in range(0, len(eigenvectors) - 1):
    for j in range(i+1, len(eigenvectors)):
        vector_A = eigenvectors[:, i]
        vector_B = eigenvectors[:, j]

        x_values = np.dot(centered_data, vector_A)
        y_values = np.dot(centered_data, vector_B)
        k = 0
        for c in class_id:
            if c == 0:
                plt.plot(x_values[k], y_values[k], '.r')
            else:
                plt.plot(x_values[k], y_values[k], '.b')
            k += 1

        ndata = pd.DataFrame(columns=['one', 'two', 'class'])
        ndata['one'] = x_values
        ndata['two'] = y_values
        ndata['class'] = class_id

        # print(ndata)
        attributes = ndata[['one', 'two']]
        class_label = ndata[['class']]

        X_train, X_test, y_train, y_test = train_test_split(attributes, class_label, random_state=42)

        neigh = KNeighborsClassifier()
        # rf_clf = RandomForestClassifier(random_state=42)
        # rf_clf.fit(X_train, y_train)
        neigh.fit(X_train, y_train)

        # y_pred = rf_clf.predict(X_test)
        # s = rf_clf.score(X_test, y_test)
        s = neigh.score(X_test, y_test)
        print(str(s) + " for " + str(i+1) + " and " + str(j+1))

        plt.title("Data projected onto eigenvectors " + str(i+1) + " and " + str(j+1))

        plt.savefig("Figure_" + str(i+1) + "_" + str(j+1) + ".png")
        plt.clf()
