import glob
from utils import *
from numpy import savetxt

normal_path= glob.glob("Dataset/normal/*")
glaucoma_path= glob.glob("Dataset/glaucoma/*")
 
print('Counting GLCM.')
glcm_normal = [glcm_blood_vessel(cv2.imread(file)) for file in normal_path]
savetxt('Array/glcm_normal.txt', glcm_normal, delimiter=',')
print('Processing GLCM...(1/2)')

glcm_glaucoma = [glcm_blood_vessel(cv2.imread(file)) for file in glaucoma_path]
savetxt('Array/glcm_glaucoma.txt', glcm_glaucoma, delimiter=',')
print('Processing GLCM...(2/2)')

print('Counting Moment Invariant.')
moment_invariant_normal = [count_moment_invariant(cv2.imread(file)) for file in normal_path]
savetxt('Array/moment_invariant_normal.txt', moment_invariant_normal, delimiter=',')
print('Processing Invariant Moment...(1/2)')

moment_invariant_glaucoma = [count_moment_invariant(cv2.imread(file)) for file in glaucoma_path]
savetxt('Array/moment_invariant_glaucoma.txt', moment_invariant_glaucoma, delimiter=',')
print('Processing Invariant Moment...(2/2)')

print('Counting PHOG.')
phog_normal = [count_phog(cv2.imread(file), max_level=5) for file in normal_path]
savetxt('Array/phog_normal.txt', phog_normal, delimiter=',')
print('Processing PHOG...(1/2)')

phog_glaucoma = [count_phog(cv2.imread(file), max_level=5) for file in glaucoma_path]
savetxt('Array/phog_glaucoma.txt', phog_glaucoma, delimiter=',')
print('Processing PHOG...(2/2)')
