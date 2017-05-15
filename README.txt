Written by Ben Autrey: https://github.com/bsautrey

---Overview---

EM.py is the Expectation-Maximization algorithm, written from Andrew Ng's notes: http://cs229.stanford.edu/notes/cs229-notes7b.pdf. Also see http://cs229.stanford.edu/notes/cs229-notes8.pdf for mathematical derivations and http://cs229.stanford.edu/section/gaussians.pdf for a description of Gaussian isocontours.

tol - Stopping criteria.
k - The number of gaussian densities.
multinomial - The current estimates for the latent multinomial random variable.
gaussians - The current estimates for the k gaussian densities.

---Requirements---

* numpy: https://docs.scipy.org/doc/numpy/user/install.html
* matplotlib: https://matplotlib.org/users/installing.html

---Example---

1) Change dir to where EM.py is.

2) Run this in a python terminal:

from EM import EM
em = EM()
em.generate_example()

OR

See the function generate_example() in EM.py.