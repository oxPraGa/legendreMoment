from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean
from legendre_moment import legenderMomentFeautre



img = data.astronaut();
img = resize(img, (img.shape[0] // 100, img.shape[1] // 100),
                       anti_aliasing=True)
ft = legenderMomentFeautre(img,32,128)
print(ft)