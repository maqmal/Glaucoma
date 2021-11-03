import numpy as np
import cv2 
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy

from scipy import *
from scipy import copysign, log10

from skimage.feature import hog

def doGLCM(img):
    g = greycomatrix(img, [1], [0], levels=img.max()+1, symmetric=False, normed=True)
    glcm_energy = greycoprops(g, 'energy')[0][0] 
    glcm_contrast = greycoprops(g, 'contrast')[0][0]
    glcm_correlation = greycoprops(g, 'correlation')[0][0]
    glcm_homogeneity = greycoprops(g, 'homogeneity')[0][0]
    glcm_entropy = shannon_entropy(img)
    glcm_dissimilarity = greycoprops(g, 'dissimilarity')[0][0]
    
    return [glcm_energy,glcm_contrast,glcm_correlation,glcm_homogeneity,glcm_entropy,glcm_dissimilarity]

def adaptiveThreshold(imgSrc, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=None, constant=None):
    if adaptiveMethod == 'mean':
        adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
    if thresholdType == 'binary':
        thresholdType =  cv2.THRESH_BINARY

    return cv2.adaptiveThreshold(imgSrc, maxValue, adaptiveMethod, thresholdType, blockSize, constant)

# Function untuk menghitung GLCM dari vessel
def glcm_blood_vessel(image):
    b,green_fundus,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)		

    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"	
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)	

    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv2.bitwise_not(finimage)
    blood_vessels = cv2.subtract(255, blood_vessels)

    # Calculate GLCM
    arr_glcm = doGLCM(blood_vessels)
    arr_glcm = np.array(arr_glcm)

    return arr_glcm

# Function untuk segmentasi OD dan OC (return 2 output, OD dan OC)
def od_oc_segmentation(image):
    Abo,Ago,Aro = cv2.split(image)  #splitting into 3 channels

    Ar = Aro - Aro.mean()           #Preprocessing Red
    Ar = Ar - Ar.mean() - Aro.std() #Preprocessing Red
    Ar = Ar - Ar.mean() - Aro.std() #Preprocessing Red

    Mr = Ar.mean()                           #Mean of preprocessed red
    SDr = Ar.std()                           #SD of preprocessed red
    # Thr = 49.5 - 12 - Ar.std()               #OD Threshold
    Thr = Ar.std()

    Ag = Ago - Ago.mean()           #Preprocessing Green
    Ag = Ag - Ag.mean() - Ago.std() #Preprocessing Green

    Mg = Ag.mean()                           #Mean of preprocessed green
    SDg = Ag.std()                           #SD of preprocessed green
    Thg = Ag.mean() + 2*Ag.std() + 49.5 + 12 #OC Threshold

    r,c = Ag.shape
    Dd = np.zeros(shape=(r,c))
    Dc = np.zeros(shape=(r,c))

    for i in range(1,r):
        for j in range(1,c):
            if Ar[i,j]>Thr:
                Dd[i,j]=255
            else:
                Dd[i,j]=0

    for i in range(1,r):
        for j in range(1,c):
            if Ag[i,j]>Thg:
                Dc[i,j]=1
            else:
                Dc[i,j]=0

    optic_cup = Dc
    optic_disk = Dd
    
    return optic_disk,optic_cup

# Function untuk menghitung moment invariant
def count_moment_invariant(image):
    optic_disk, optic_cup = od_oc_segmentation(image)
    
    moments = cv2.moments(optic_disk)
    huMoments = cv2.HuMoments(moments)
    for i in range(0,7):
        huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    huMoments = huMoments.ravel()
    return huMoments

def vectorHOG(image, level):
    x = 128 / (2**level)
    y = 64 / (2**level)

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(y, x),
                    cells_per_block=(1, 1), visualize=True)

    return fd

def count_phog(image, max_level):
    phog_feature = []
    # convertRGB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # enhancedImage = cv2.convertScaleAbs(convertRGB, alpha=2, beta=22)

    resized_img = cv2.resize(image, (128,64))
    for level in range(max_level):
        vectorCiri = vectorHOG(resized_img, level)
        for i in range(len(vectorCiri)):
            phog_feature.append(vectorCiri[i])
    return phog_feature