from utils import *
from numpy import loadtxt
from tpot import TPOTClassifier
from sklearn.model_selection import cross_val_score

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
from sklearn.preprocessing import MinMaxScaler
X = np.vstack((X_normal,X_glaucoma))
y = np.append(label_0,label_1)

from imblearn import over_sampling

X_over_smote, y_over_smote = over_sampling.SMOTE().fit_resample(X, y)

print('Splitting train test...')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_over_smote, y_over_smote, test_size=0.25, random_state=0)
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

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

print('\n')
print('Jumlah data (sebelum SMOTE) = ', X.shape[0])
print('Jumlah data (setelah SMOTE) = ', X_over_smote.shape[0])
print('=================PREDICTING========================')
print('Predicting...')
predictions_knn = knn.predict(X_test)
predictions_rfc = rfc.predict(X_test)
predictions_svm=svc.predict(X_test)

print('======================ACCURACY========================')
from sklearn.metrics import accuracy_score
print('KNN Accuracy = %.3f' % accuracy_score(y_test,predictions_knn))
print('Random Forest Accuracy = %.3f' % accuracy_score(y_test,predictions_rfc))
print('SVM Accuracy = %.3f' % accuracy_score(y_test,predictions_svm))


print('=================CROSS VALIDATION========================')
scores_knn = cross_val_score(KNeighborsClassifier(), X_over_smote, y_over_smote, cv=5)
scores_rf = cross_val_score(RandomForestClassifier(random_state=0), X_over_smote, y_over_smote, cv=5)
scores_svm = cross_val_score(SVC(), X_over_smote, y_over_smote, cv=5)
print("KNN = %0.2f accuracy with a standard deviation of %0.2f" % (scores_knn.mean(), scores_knn.std()))
print("RF = %0.2f accuracy with a standard deviation of %0.2f" % (scores_rf.mean(), scores_rf.std()))
print("SVM = %0.2f accuracy with a standard deviation of %0.2f" % (scores_svm.mean(), scores_svm.std()))

print('======================F1 SCORE========================')
from sklearn.metrics import f1_score
print("KNN = %.3f" % f1_score(y_test, predictions_knn, average='macro'))
print("RF = %.3f" % f1_score(y_test, predictions_rfc, average='macro'))
print("SVM = %.3f" % f1_score(y_test, predictions_svm, average='macro'))

# print("=================TPOT=======================")
# tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
# tpot.fit(X_train, y_train)
# print(tpot.score(X_test, y_test))
# tpot.export('hasil_tpot.py')

# # =====================Ini hasil pipeline dari TPOT==========================
# from sklearn.neural_network import MLPClassifier
# exported_pipeline = MLPClassifier(alpha=0.01, learning_rate_init=0.01)
# if hasattr(exported_pipeline, 'random_state'):
#     setattr(exported_pipeline, 'random_state', 42)

# exported_pipeline.fit(X_train, y_train)
# results = exported_pipeline.predict(X_test)
# print('MLP Accuracy = %.3f' % accuracy_score(y_test,results))
# scores_mlp = cross_val_score(exported_pipeline, X_over_smote, y_over_smote, cv=5)
# print("Cross Val = %0.2f accuracy with a standard deviation of %0.2f" % (scores_mlp.mean(), scores_mlp.std()))
# print("F1 Score = %.3f" % f1_score(y_test, results, average='macro'))