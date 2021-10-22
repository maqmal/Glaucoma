import glob
from utils import *
import time
from numpy import savetxt
from numpy import loadtxt

print('Counting GLCM.')
glcm_normal = [glcm_blood_vessel(cv2.imread(file)) for file in glob.glob("Dataset/normal/*")]
print('Processing GLCM...(1/2)')
print('Saving array into .txt file.')
savetxt('glcm_normal.txt', glcm_normal, delimiter=',')

glcm_glaucoma = [glcm_blood_vessel(cv2.imread(file)) for file in glob.glob("Dataset/glaucoma/*")]
print('Processing GLCM...(2/2)')
savetxt('glcm_glaucoma.txt', glcm_glaucoma, delimiter=',')
print('Saving array into .txt file.')

# # Di tahap ini lama banget
print('Counting Moment Invariant.')
moment_invariant_normal = [count_moment_invariant(cv2.imread(file)) for file in glob.glob("Dataset/normal/*")]
savetxt('moment_invariant_normal.txt', moment_invariant_normal, delimiter=',')
print('Processing Invariant Moment...(1/2)')

moment_invariant_glaucoma = [count_moment_invariant(cv2.imread(file)) for file in glob.glob("Dataset/glaucoma/*")]
savetxt('moment_invariant_glaucoma.txt', moment_invariant_glaucoma, delimiter=',')
print('Processing Invariant Moment...(2/2)')

# # Kalau mau load array dari .txt:
# glcm_normal = loadtxt('glcm_normal.txt', delimiter=',')
# glcm_glaucoma = loadtxt('glcm_glaucoma.txt', delimiter=',')
# moment_invariant_normal = loadtxt('moment_invariant_normal.txt', delimiter=',')
# moment_invariant_glaucoma = loadtxt('moment_invariant_glaucoma.txt', delimiter=',')

label_0 = []
label_1 = []

for i in range(len(glcm_normal)):
    label_0.append(0)
for i in range(len(glcm_glaucoma)):
    label_1.append(1)

X_normal = np.concatenate([glcm_normal,moment_invariant_normal],axis=1)
print('Concatenating array...(1/2)')
X_glaucoma = np.concatenate([glcm_glaucoma,moment_invariant_glaucoma],axis=1)
print('Concatenating array...(2/2)')

X_normal[X_normal >= 1E308] = 0
X_glaucoma[X_glaucoma >= 1E308] = 0

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

print('Training SVM')
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