import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube
import pdb

# Code block to parse string / SpectralCube into PCA
#     if isinstance(input,SpectralCube):
#         cube = input
#     elif isinstance(input,str):
#         try:
#             cube = SpectralCube.read(input)
#         except:
#             raise
#     else:
#         raise NotImplementedError
    
def EigenImages(evec,cube,nImg = 10):
    imageList = []
    for idx in range(nImg):
        thisImage = np.zeros((cube.shape[1],cube.shape[2]))
        for channel in range(cube.shape[0]):
            thisImage +=np.nan_to_num(cube[channel,:,:].value*evec[channel,idx])
        imageList.append(thisImage)
    return imageList

def pca(cube):
    PCAMatrix = np.zeros((cube.shape[0],cube.shape[0]))
    MeanOfData = cube.mean()
    for i in range(cube.shape[0]):
        for j in range(i):
            PCAMatrix[i,j] = np.nanmean(((cube[i,:,:]-MeanOfData)*(cube[j,:,:]-MeanOfData)).value)
        PCAMatrix[i,i] = np.nanmean(((cube[i,:,:]-MeanOfData)*(cube[i,:,:]-MeanOfData)).value)
    PCAMatrix = PCAMatrix + np.transpose(PCAMatrix)
    # Correct elements on the diagonal for the doubling in the transpose-and-add
    PCAMatrix[range(cube.shape[0]),range(cube.shape[0])] = \
        PCAMatrix[range(cube.shape[0]),range(cube.shape[0])]/2
    evals,evec = np.linalg.eig(PCAMatrix)
    order = (np.argsort(evals))[::-1]
    evals = evals[order]
    evec = evec[:,order]
    return evals,evec,PCAMatrix
