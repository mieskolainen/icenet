import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#from mpl_toolkits.mplot3d import Axes3D
#from tqdm import tqdm

N = 5000 # Events

Nbins = 40

for K in [2,3,4,5,6,7,8]:    # Number of classes
	
	# Naive flat sampling
	a = np.random.rand(N,K)
	row_sums = a.sum(axis=1)
	n = a / row_sums[:,np.newaxis]

	# Dirichlet (proper flat simplex) sampling
	alpha  = np.ones(K) # parameter. case == 1 is flat
	d      = np.random.dirichlet(alpha=alpha, size=(N)) #.transpose()

	# --------------------------------------------------------------------
	
	fig,ax = plt.subplots(2,2, figsize=(6,6))
	plt.subplots_adjust(hspace=0.3)
	fig.suptitle(f'$D={K}$', fontsize=16)

	# Visualize in xy-plane
	#ax[0,0].scatter(n[:,0], n[:,1], color='black')
	ax[0,0].hist2d(n[:,0], n[:,1], bins=Nbins, range=[[0,1], [0,1]], norm=mcolors.PowerNorm(0.5), linewidth=0.0)

	ax[0,0].set(xlim=(0,1), ylim=(0,1))
	ax[0,0].set_aspect('equal', adjustable='box')
	ax[0,0].set_xlabel('$x_1$')
	ax[0,0].set_ylabel('$x_2$')
	ax[0,0].set_title('$x_i \\sim$U$(0,1)$ & 1/sum$\\{x_i\\}$')

	counts, bins = np.histogram(n[:,0], Nbins)
	ax[1,0].hist(bins[:-1], bins, weights=counts, histtype='step', label='$x_i$')
	counts, bins = np.histogram(np.max(n, axis=1), Nbins)
	ax[1,0].hist(bins[:-1], bins, weights=counts, histtype='step', label='max$\\{x_i\\}$')


	ax[1,0].set(xlim=(0,1), ylim=(0,None))
	ax[1,0].set_xlabel('$x$')
	ax[1,0].legend()
	#ax[1,0].set_aspect('equal', adjustable='box')


	#ax[0,1].scatter(d[:,0], d[:,1], color='black')
	ax[0,1].hist2d(d[:,0], d[:,1], bins=Nbins, range=[[0,1], [0,1]], norm=mcolors.PowerNorm(0.5), linewidth=0.0)
	ax[0,1].set(xlim=(0,1), ylim=(0,1))
	ax[0,1].set_aspect('equal', adjustable='box')
	ax[0,1].set_xlabel('$x_1$')
	ax[0,1].set_ylabel('$x_2$')
	ax[0,1].set_title('$\\vec{x} \\sim$Dirichlet($\\vec{\\alpha}\\equiv 1$)')


	counts, bins = np.histogram(d[:,0], Nbins)
	ax[1,1].hist(bins[:-1], bins, weights=counts, histtype='step', label='$x_i$')
	counts, bins = np.histogram(np.max(d, axis=1), Nbins)
	ax[1,1].hist(bins[:-1], bins, weights=counts, histtype='step', label='max$\\{x_i\\}$')


	ax[1,1].set(xlim=(0,1), ylim=(0,None))
	ax[1,1].set_xlabel('$x$')
	ax[1,1].legend()
	#ax[1,1].set_aspect('equal', adjustable='box')

	plt.savefig(f'dir_{K}.pdf')
	#plt.show()


