"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data1, make_data2
from plot import plot_boundary


class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):

	def fit(self, X, y):
		"""Fit a Gaussian navie Bayes model using the training set (X, y).
			self.
		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			The training input samples.

		y : array-like, shape = [n_samples]
			The target values.

		Returns
		-------
		self : object
			Returns self.
		"""
		# Input validation
		X = np.asarray(X, dtype=np.float)
		if X.ndim != 2:
			raise ValueError("X must be 2 dimensional")

		y = np.asarray(y)
		if y.shape[0] != X.shape[0]:
			raise ValueError("The number of samples differs between X and y")

		# ====================
		# Get classes and number of occurence
		self.classes_, classes_count = np.unique(y, return_counts=True)

		# Get prior for each class
		self.classes_prior_ = np.divide(classes_count, y.shape[0])

		# Get means and std of all features for each classes
		self.likelihood_means_ = np.empty((self.classes_.shape[0], X.shape[1]))
		self.likelihood_stds_ = np.empty((self.classes_.shape[0], X.shape[1]))
		for class_id in range(self.classes_.shape[0]):
			self.likelihood_means_[class_id] = np.mean(X[y==self.classes_[class_id],:], axis=0)
			self.likelihood_stds_[class_id] = np.std(X[y==self.classes_[class_id],:], axis=0)

		# ====================

		return self

	def predict(self, X):
		"""Predict class for X.

		Parameters
		----------
		X : array-like of shape = [n_samples, n_features]
			The input samples.

		Returns
		-------
		y : array of shape = [n_samples]
			The predicted classes, or the predict values.
		"""

		# ====================
		y = np.argmax(self.predict_proba(X, normalize = False), axis=1)
		for class_id in range(self.classes_.shape[0]):
			y[y==class_id] = self.classes_[class_id]

		return y
		# ====================

	def predict_proba(self, X, normalize = True):
		"""Return probability estimates for the test data X.

		Parameters
		----------
		X : array-like of shape = [n_samples, n_features]
			The input samples.

		Returns
		-------
		p : array of shape = [n_samples, n_classes]
			The class probabilities of the input samples. Classes are ordered
			by lexicographic order.
		"""

		# ====================
		if not hasattr(self, "classes_"):
			raise AttributeError("Classifier not fitted")

		# Compute probability for each sample and each classes
		p = np.empty([X.shape[0], self.classes_.shape[0]])
		for sample_id in range(X.shape[0]):
			for class_id in range(self.classes_.shape[0]):
				p[sample_id, class_id] = self.classes_prior_[class_id] * self.simplified_gaussian(self.likelihood_means_[class_id],
																								 self.likelihood_stds_[class_id],
																								 X[sample_id,:])

		# Normalization
		if normalize:
			for sample_id in range(X.shape[0]):
				p[sample_id,:] = np.divide(p[sample_id,:], np.sum(p[sample_id,:]))

		return p
		# ====================

	def simplified_gaussian(self, means_val, stds_val, features_val):
		"""Return the gaussian probability product for all features (not normalized by sqrt(2*pi)).

		Parameters
		----------
		means_val	: 	array of shape = [n_features,]
						The means for all features having the same class value.

		stds_val	: 	array of shape = [n_features,]
						The standard deviations for all features having the same class value.

		features_val:	array of shape = [n_features,]
						The features' values for all features having the same class value.

		Returns
		-------
		p 			:	 gaussian probability product for all features (not normalized by sqrt(2*pi)).

		"""


		exp_inner_sum = 0.
		cumulated_std = 1.
		for feature_id in range(features_val.shape[0]):
			numerator = (features_val[feature_id] - means_val[feature_id])**2
			denominator = 2*(stds_val[feature_id])**2
			exp_inner_sum -= numerator/denominator

			cumulated_std *= stds_val[feature_id]

		p = np.exp(exp_inner_sum)/cumulated_std

		return p





if __name__ == "__main__":
	# Calculating accuracies on test set
    X,y = make_data1(2000, random_state=0)
    my_classifier = GaussianNaiveBayes()
    my_classifier.fit(X[0:150], y[0:150])
    X2,y2 = make_data2(2000, random_state=0)
    my_classifier2 = GaussianNaiveBayes()
    my_classifier2.fit(X2[0:150], y2[0:150])
    print("Testing set accuracy for dataset 1 = "  + str(my_classifier.score(X[150:2000], y[150:2000])))
    print("Testing set accuracy for dataset 2 = "  + str(my_classifier2.score(X2[150:2000], y2[150:2000])))
