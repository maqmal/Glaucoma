import glob
from utils import *
from numpy import savetxt

print('Counting GLCM.')
glcm_normal = [glcm_blood_vessel(cv2.imread(file)) for file in glob.glob("Dataset/normal/*")]
savetxt('glcm_normal.txt', glcm_normal, delimiter=',')
print('Processing GLCM...(1/2)')

glcm_glaucoma = [glcm_blood_vessel(cv2.imread(file)) for file in glob.glob("Dataset/glaucoma/*")]
savetxt('glcm_glaucoma.txt', glcm_glaucoma, delimiter=',')
print('Processing GLCM...(2/2)')

print('Counting Moment Invariant.')
moment_invariant_normal = [count_moment_invariant(cv2.imread(file)) for file in glob.glob("Dataset/normal/*")]
savetxt('moment_invariant_normal.txt', moment_invariant_normal, delimiter=',')
print('Processing Invariant Moment...(1/2)')

moment_invariant_glaucoma = [count_moment_invariant(cv2.imread(file)) for file in glob.glob("Dataset/glaucoma/*")]
savetxt('moment_invariant_glaucoma.txt', moment_invariant_glaucoma, delimiter=',')
print('Processing Invariant Moment...(2/2)')

print('Counting PHOG.')
phog_normal = [count_phog(cv2.imread(file), max_level=5) for file in glob.glob("Dataset/normal/*")]
savetxt('phog_normal.txt', phog_normal, delimiter=',')
print('Processing PHOG...(1/2)')

phog_glaucoma = [count_phog(cv2.imread(file), max_level=5) for file in glob.glob("Dataset/glaucoma/*")]
savetxt('phog_glaucoma.txt', phog_glaucoma, delimiter=',')
print('Processing PHOG...(2/2)')
