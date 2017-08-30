import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy.optimize as op
from scipy import stats

def KS_test(alpha, h, b):
	
	pl = b**alpha / b[0]**alpha

	return stats.ks_2samp(h/h[0], pl)[0]


def lnlike(alpha, h, b):

	b_m = b[0]

	a1 = np.sum(h) * (alpha - 1) * np.log(b_m)
	a2 = np.dot(h, np.log(
		(b**(1 - alpha) - np.roll(b, -1)**(1 - alpha))[:-1]))

	return a1 + a2


def lnprior(alpha):

	if 1 < alpha < 5:
		return 0.0
	
	return -np.inf


def lnprob(alpha, h, b):

	lp = lnprior(alpha)
	if not np.isfinite(lp):
		return -np.inf

	return lp + lnlike(alpha, h, b)


def emcee_fit(h, b):

	initial = opt_fit(h, b)

	ndim, nwalkers = 1, 100
	pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(h, b))

	print("Running MCMC...")
	sampler.run_mcmc(pos, 500, rstate0=np.random.get_state())
	print("Done.")

	burn_in = 50
	samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
	return samples
	

def opt_fit(h, b):

	neg_like = lambda *args: -lnlike(*args)
	result = op.minimize(neg_like, [1.2], args=(h, b))

	return result['x']


def plot_trace(samples):

	plt.figure()
	plt.hist(samples)
	plt.show()


def plot_alpha(alpha, h, b):

	pl = lambda x: b**x / b[0]**x

	plt.figure()
	plt.loglog(b[:-1], h / h[0])
	plt.plot(b, pl(alpha))

	plt.show()


if __name__ == '__main__':

	
	filename = "flares.txt"
	data = np.loadtxt(filename)

	ks = np.inf
	h_out, b_out = None, None
	for i in range(1, 15):

		min_threshold = i * 1e1
		h, b = np.histogram(data, bins=np.linspace(min_threshold, 1e3, 100))
		alpha = -1 * opt_fit(h, b)

		ks_n = KS_test(alpha, h, b)
		if ks_n < ks:
			ks = ks_n
			h_out = h
			b_out = b


	samples = emcee_fit(h_out, b_out)
	plot_trace(samples)

	alpha = -np.mean(samples)
	iqr = np.subtract(*np.percentile(samples, [75, 25]))

	print('Best fit alpha: {0:.3f} +- {1:.5f}'.format(alpha, iqr))

	plot_alpha(alpha, h_out, b_out)
