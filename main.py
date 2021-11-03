from utils import *
from numpy import loadtxt

glcm_normal = loadtxt('Array/glcm_normal.txt', delimiter=',')
glcm_glaucoma = loadtxt('Array/glcm_glaucoma.txt', delimiter=',')
moment_invariant_normal = loadtxt('Array/moment_invariant_normal.txt', delimiter=',')
moment_invariant_glaucoma = loadtxt('Array/moment_invariant_glaucoma.txt', delimiter=',')
phog_normal = loadtxt('Array/phog_normal.txt', delimiter=',')
phog_glaucoma = loadtxt('Array/phog_glaucoma.txt', delimiter=',')

label_0 = []
label_1 = []

for i in range(len(glcm_normal)):
    label_0.append(0)
for i in range(len(glcm_glaucoma)):
    label_1.append(1)

X_normal = np.concatenate([glcm_normal,moment_invariant_normal,phog_normal],axis=1)
print('Concatenating array...(1/2)')
X_glaucoma = np.concatenate([glcm_glaucoma,moment_invariant_glaucoma,phog_glaucoma],axis=1)
print('Concatenating array...(2/2)')

# Biar gk ada angka infinity
X_normal[X_normal >= 1E308] = 0
X_glaucoma[X_glaucoma >= 1E308] = 0
X_normal[~np.isfinite(X_normal)] = 0
X_glaucoma[~np.isfinite(X_glaucoma)] = 0

# X = Gabungin normal & glaucoma, y = label nya (normal=0,glaucoma=1)
X = np.vstack((X_normal,X_glaucoma))
y = np.append(label_0,label_1)

print('Splitting train test...')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print('Training KNN...')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

print('Training Random Forest...')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

print('Training SVM...')
from sklearn.svm import SVC
svc=SVC() 
svc.fit(X_train,y_train)


print('====================================================')
print('Predicting...')
predictions_knn = knn.predict(X_test)
predictions_rfc = rfc.predict(X_test)
predictions_svm=svc.predict(X_test)

from sklearn.metrics import accuracy_score
print('KNN Accuracy: %.3f' % accuracy_score(y_test,predictions_knn))
print('Random Forest Accuracy: %.3f' % accuracy_score(y_test,predictions_rfc))
print('SVM Accuracy: %.3f' % accuracy_score(y_test,predictions_svm))