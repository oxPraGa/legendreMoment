import numpy as np
from scipy.special import legendre
from skimage.color import rgb2gray

def legenderMomentFeautre(img , p , q):
    # convert image to gray
    grayscale = rgb2gray(img)
    # get image size
    [n , m] = grayscale.shape

    features = []
    for i in range(1,p +1 ): 
        for j in range(1,q+1):
            # get legendre polynoms
            lp_p = legendre(i)
            lp_q = legendre(j)
            # normalization
            norm = ((2*i+1)*(2*j+1))/((n-1)*(m-1))
            # get moment  
            mom = legenderMoment2D(grayscale,lp_p,lp_q,n,m,norm)
            features.append(mom)
    return features

def legenderMoment2D(img_gray,lp_p,lp_q,N,M,norm):
    mom = 0
    for i in range(0,N):
        for j in range(0,M):
            xi = 2 * i / (N-1) - 1
            yj = 2 * j / (M-1) - 1
            mom += lp_p(xi) * lp_q(yj) * img_gray[i,j]
    return mom * norm 
