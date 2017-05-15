# EM.py is the Expectation-Maximization algorithm, written from Andrew Ng's notes: http://cs229.stanford.edu/notes/cs229-notes7b.pdf. Also see http://cs229.stanford.edu/notes/cs229-notes8.pdf for mathematical derivations and http://cs229.stanford.edu/section/gaussians.pdf for a description of Gaussian isocontours.

import random,pickle
from copy import copy
from math import fmod

import numpy as np
import matplotlib.pyplot as plot

# tol - Stopping criteria.
# k - The number of gaussian densities.
# multinomial - The latent random variable.
# gaussians - The current estimates for the k gaussian densities.


class EM():
    
    def __init__(self):
        self.X = None
        self.tol = None
        self.k = None
        self.multinomial = None
        self.gaussians = None
        
    def set_X(self,X):
        self.X = X
        
    def set_tolerance(self,tol=0.1):
        self.tol = tol
        
    def initialize_EM(self,k):
        self.k = k
        
        # initialize multinomial
        self.multinomial = {}
        prob = 1.0/self.k
        for j in xrange(self.k):
            self.multinomial[j] = prob
            
        # initialize gaussians
        # build hypercube from min/max of dataset
        number_of_rows = self.X.shape[0]
        number_of_columns = self.X.shape[1]
        mins = [float('inf'),]*number_of_columns
        maxes = [float('-inf'),]*number_of_columns
        for i in xrange(number_of_rows):
            x = self.X[i,:]
            for j in xrange(number_of_columns):
                val = x[j]
                current_min = mins[j]
                current_max = maxes[j]
                
                if val < current_min:
                    mins[j] = val
                
                if val > current_max:
                    maxes[j] = val 
        
        # select a random mean from the min/max hypercube. also, initialize covariance, inverse, and determinant.
        self.gaussians = {}
        for j in xrange(self.k):
            self.gaussians[j] = {}
            
            # mean
            mean = []
            for inx in xrange(number_of_columns):
                min_j = mins[inx]
                max_j = maxes[inx]
                rand = random.uniform(min_j,max_j)
                mean.append(rand)
                
            mean = np.array(mean)
            mean.resize((number_of_columns,1))
            self.gaussians[j]['mean'] = mean
            
            # cov
            cov = []
            for inx in xrange(number_of_columns):
                vals = [0.0,]*number_of_columns
                vals[inx] = 1.0
                cov.append(vals)
            cov = np.array(cov)
            self.gaussians[j]['cov'] = cov
            self.gaussians[j]['inv'] = np.linalg.inv(cov)
            self.gaussians[j]['det'] = np.linalg.det(cov)
            
    def run_EM(self):
        current_likelihood = self._calculate_likelihood()
        while True:
            weights = self._E_step()
            self._M_step(weights)
            
            new_likelihood = self._calculate_likelihood()
            if np.isnan(new_likelihood):
                print 'POOR INITIALIZATION. REINITIALIZING...'
                self.initialize_EM(self.k)
                current_likelihood = self._calculate_likelihood()
                continue
                
            diff = new_likelihood - current_likelihood
            print 'DIFF LIKELIHOOD:',diff,new_likelihood
            if diff < self.tol:
                return new_likelihood
                
            current_likelihood = new_likelihood
        
    def _E_step(self):
        weights = {}
        for j in xrange(self.k):
            weights[j] = []
            
        number_of_rows = self.X.shape[0]
        number_of_columns = self.X.shape[1]
        for j in weights:
            weights_j = []
            for i in xrange(number_of_rows):
                x = self.X[i]
                x.resize((number_of_columns,1))
                prob_z_given_x = self._prob_z_given_x(x,j)
                weights_j.append((prob_z_given_x,i))
            
            weights[j] = weights_j
            
        return weights
                
    def _M_step(self,weights):
        self._update_phi(weights)
        self._update_mean(weights)
        self._update_cov(weights)
        
    def _calculate_likelihood(self):
        likelihood = 0.0
        
        number_of_rows = self.X.shape[0]
        number_of_columns = self.X.shape[1]
        for i in xrange(number_of_rows):
            x = self.X[i]
            x.resize((number_of_columns,1))
            for j in xrange(self.k):
                prob_x_given_z = self._prob_x_given_z(x,j)
                prob_z_given_x = self._prob_z_given_x(x,j)
                prob_z = self._prob_z(j)
                
                likelihood = likelihood + prob_z_given_x*np.log(prob_x_given_z*prob_z/prob_z_given_x)
                
        return likelihood
        
    # use Bayes' rule to get prob_z_given_x
    def _prob_z_given_x(self,x,j):
        prob_x_given_z = self._prob_x_given_z(x,j)
        prob_z = self._prob_z(j)
        
        sum_probs = 0.0
        for l in xrange(self.k):
            sum_probs = sum_probs + self._prob_x_given_z(x,l)*self._prob_z(l)
            
        prob_z_given_x = prob_x_given_z*prob_z/sum_probs
        
        return prob_z_given_x
        
    # I had to write the gaussian pdf. I could not get the gaussians pdf in scipy to work.
    def _prob_x_given_z(self,x,j):
        gaussian = self.gaussians[j]
        mean = gaussian['mean']
        inv = gaussian['inv']
        det = gaussian['det']
        number_of_columns = self.X.shape[1]
        
        diff = x - mean
        quadratic = np.dot(np.dot(diff.T,inv),diff)
        e = np.exp(-0.5*quadratic[0][0])
        c = 1.0/((2*np.pi)**(number_of_columns/2.0)*det**0.5)
        prob_x_given_z = c*e
        
        return prob_x_given_z
        
    def _prob_z(self,j):
        prob_z = self.multinomial[j]
        
        return prob_z
        
    def _update_phi(self,weights):
        for j in weights:
            phi_j = 0.0
            for prob_z_given_x,_ in weights[j]:
                phi_j = phi_j + prob_z_given_x
            
            number_of_rows = self.X.shape[0]
            phi_j = phi_j/number_of_rows
            self.multinomial[j] = phi_j
                
    def _update_mean(self,weights):
        number_of_columns = self.X.shape[1]
        for j in weights:
            sum_probs = 0.0
            mean_j = np.zeros((number_of_columns,1))
            for prob_z_given_x,i in weights[j]:
                x = self.X[i]
                x.resize((number_of_columns,1))
                mean_j = mean_j + prob_z_given_x*x
                sum_probs = sum_probs + prob_z_given_x
            
            mean_j = mean_j/sum_probs
            self.gaussians[j]['mean'] = mean_j
                
    def _update_cov(self,weights):
        number_of_columns = self.X.shape[1]
        for j in weights:
            sum_probs = 0.0
            cov_j = np.zeros((number_of_columns,number_of_columns))
            mean_j = self.gaussians[j]['mean']
            for prob_z_given_x,i in weights[j]:
                x = self.X[i]
                x.resize((number_of_columns,1))
                diff = x - mean_j
                product = prob_z_given_x*diff*diff.T
                cov_j = cov_j + product
                sum_probs = sum_probs + prob_z_given_x
            
            cov_j = cov_j/sum_probs
            inv_j = np.linalg.inv(cov_j)
            det_j = np.linalg.det(cov_j)
            self.gaussians[j]['cov'] = cov_j
            self.gaussians[j]['inv'] = inv_j
            self.gaussians[j]['det'] = det_j
                
    def generate_example(self,number_of_clusters=3,sample_size_per_cluster=500,number_of_runs=3):
        # assemble data
        X = np.array([])
        X.resize((0,2))
        for inx in xrange(number_of_clusters):
            x_1 = random.sample(range(0,15),1)[0]
            x_2 = random.sample(range(0,15),1)[0]
            cov = random.uniform(-0.75,0.75)
            mean = np.array([x_1,x_2])
            cov = np.array([[1,cov],[cov,1]])
            res = np.random.multivariate_normal(mean,cov,sample_size_per_cluster)
            X = np.row_stack((X,res))
            
        # run EM a few times
        likelihoods = []
        for i in xrange(number_of_runs):
            print 'RUN:',i
            self.set_X(X)
            self.set_tolerance()
            self.initialize_EM(number_of_clusters)
            likelihood = self.run_EM()
            s_gaussians = pickle.dumps(self.gaussians)
            pair = (likelihood,s_gaussians)
            likelihoods.append(pair)
            print '---\n'
            
        # select run with highest likelihood
        likelihoods.sort(reverse=True)
        s_gaussians = likelihoods[0][1]
        gaussians = pickle.loads(s_gaussians)
        
        # plot the original data, X, and the estimated means from EM.
        plot.scatter(self.X[:,0],self.X[:,1],s=0.5,color='grey')
        for j in gaussians:
            mean_j = gaussians[j]['mean']
            plot.scatter(mean_j[0][0],mean_j[1][0])
            
        plot.show()
        
        # plot random samples from the densities estimated by EM and the means estimated by EM.
        colors = ['red','orange','green','purple','grey']
        X_colors = []
        X = np.array([])
        X.resize((0,2))
        for j in gaussians:
            color = colors[int(fmod(j,len(colors)))]
            mean_j = gaussians[j]['mean']
            cov_j = gaussians[j]['cov']
            res = np.random.multivariate_normal(mean_j.T[0],cov_j,sample_size_per_cluster)
            X = np.row_stack((X,res))
            for inx in xrange(sample_size_per_cluster):
                X_colors.append(color)
                
        X_colors = np.array(X_colors)
        X_colors.resize((number_of_clusters*sample_size_per_cluster,1))
        
        plot.scatter(X[:,0],X[:,1],color=X_colors[:,0],s=0.5)
        for j in gaussians:
            mean_j = gaussians[j]['mean']
            plot.scatter(mean_j[0][0],mean_j[1][0])
            
        plot.show()
            
