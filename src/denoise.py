import os
import numpy as np
from scipy.ndimage import imread
import fca
import ica
import pca
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


output_dir = '../results/denoise/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

image1 = "../images/locust.jpg"
I1 = imread(image1, flatten=True).astype(np.float64) / 255
print I1.shape

I2 = np.random.randn(*I1.shape) * np.std(I1)
print I2.shape


# plot the original images
plt.figure()
plt.imshow(I1, aspect='auto', cmap='gray')
plt.savefig(output_dir + 'I1.png')
plt.close()

plt.figure()
plt.imshow(I2, aspect='auto', cmap='gray')
plt.savefig(output_dir + 'I2.png')
plt.close()


# stack and then mix the images
X = np.dstack([I1, I2]).transpose(2, 0, 1)
print X.shape

A = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
Z = np.tensordot(A, X, axes=(1, 0))
print Z.shape


# plot mixed images
plt.figure()
plt.imshow(Z[0], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'Z1.png')
plt.close()

plt.figure()
plt.imshow(Z[1], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'Z2.png')
plt.close()


# unmix the images by using free kurtosis based FCA
Aest, Xest, Fvs = fca.fcf(fca.Fhat_free_kurtosis, Z, return_Fhat=True)
print '-|k4|: ', Fvs


# plot the unmixed images
plt.figure()
plt.imshow(Xest[0], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X1_free_kurtosis.png')
plt.close()

plt.figure()
plt.imshow(Xest[1], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X2_free_kurtosis.png')
plt.close()


# unmix the images by using free entropy based FCA
Aest, Xest, Fvs = fca.fcf(fca.Fhat_free_entropy, Z, return_Fhat=True)
print 'free entropy: ', Fvs

plt.figure()
plt.imshow(Xest[0], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X1_free_entropy.png')
plt.close()

plt.figure()
plt.imshow(Xest[1], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X2_free_entropy.png')
plt.close()



# unmix the images by using kurtosis based ICA
Aest, Xest, Fvs = ica.icf(ica.Fhat_kurtosis, Z, return_Fhat=True)
print '-|c4|: ', Fvs


# plot the unmixed images
plt.figure()
plt.imshow(Xest[0], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X1_kurtosis.png')
plt.close()

plt.figure()
plt.imshow(Xest[1], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X2_kurtosis.png')
plt.close()


# unmix the images by using entropy based ICA
Aest, Xest, Fvs = ica.icf(ica.Fhat_negentropy, Z, return_Fhat=True)
print '- negentropy: ', Fvs

plt.figure()
plt.imshow(Xest[0], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X1_entropy.png')
plt.close()

plt.figure()
plt.imshow(Xest[1], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X2_entropy.png')
plt.close()


# unmix the images by using variance based PCA
Aest, Xest, Fvs = pca.pcf(pca.Fhat_variance, Z, return_Fhat=True)
print '-|c2|: ', Fvs


# plot the unmixed images
plt.figure()
plt.imshow(Xest[0], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X1_variance.png')
plt.close()

plt.figure()
plt.imshow(Xest[1], aspect='auto', cmap='gray')
plt.savefig(output_dir + 'X2_variance.png')
plt.close()
