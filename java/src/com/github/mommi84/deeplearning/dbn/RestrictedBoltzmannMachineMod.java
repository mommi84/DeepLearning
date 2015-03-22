package com.github.mommi84.deeplearning.dbn;

import java.util.Random;

public class RestrictedBoltzmannMachineMod {
	public int N;
	public int n_visible;
	public int n_hidden;
	public double[][] W;
	public double[] hbias;
	public double[] vbias;
	public Random rng;
	
	public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}
	
	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;
		
		int c = 0;
		double r;
		
		for(int i=0; i<n; i++) {
			r = rng.nextDouble();
			if (r < p) c++;
		}
		
		return c;
	}
	
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}
	
	
	public RestrictedBoltzmannMachineMod(int N, int n_visible, int n_hidden, 
			double[][] W, double[] hbias, double[] vbias, Random rng) {
		this.N = N;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;
	
		if(rng == null)	this.rng = new Random(1234);
		else this.rng = rng;
		
		if(W == null) {
			this.W = new double[this.n_hidden][this.n_visible];
			double a = 1.0 / this.n_visible;
			
			for(int i=0; i<this.n_hidden; i++) {
				for(int j=0; j<this.n_visible; j++) {
					this.W[i][j] = uniform(-a, a); 
				}
			}	
		} else {
			this.W = W;
		}
		
		if(hbias == null) {
			this.hbias = new double[this.n_hidden];
			for(int i=0; i<this.n_hidden; i++) this.hbias[i] = 0;
		} else {
			this.hbias = hbias;
		}
		
		if(vbias == null) {
			this.vbias = new double[this.n_visible];
			for(int i=0; i<this.n_visible; i++) this.vbias[i] = 0;
		} else {
			this.vbias = vbias;
		}
	}
	
	
	public void contrastive_divergence(double[] layer_input, double lr, int k) {
		double[] ph_mean = new double[n_hidden];
		double[] ph_sample = new double[n_hidden];
		double[] nv_means = new double[n_visible];
		double[] nv_samples = new double[n_visible];
		double[] nh_means = new double[n_hidden];
		double[] nh_samples = new double[n_hidden];
		
		/* CD-k */
		sample_h_given_v(layer_input, ph_mean, ph_sample);
		
		for(int step=0; step<k; step++) {
			if(step == 0) {
				gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
			} else {
				gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
			}
		}
		
		for(int i=0; i<n_hidden; i++) {
			for(int j=0; j<n_visible; j++) {
				// W[i][j] += lr *(ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
				W[i][j] += lr *(ph_mean[i] * layer_input[j] - nh_means[i] * nv_samples[j]) / N;
			}
			hbias[i] += lr * (ph_sample[i] - nh_means[i]) / N;
		}
		

		for(int i=0; i<n_visible; i++) {
			vbias[i] += lr * (layer_input[i] - nv_samples[i]) / N;
		}

	}
	
	
	public void sample_h_given_v(double[] layer_input, double[] mean, double[] ph_sample) {
		for(int i=0; i<n_hidden; i++) {
			mean[i] = propup(layer_input, W[i], hbias[i]);
			ph_sample[i] = binomial(1, mean[i]);
		}
	}

	public void sample_v_given_h(double[] h0_sample, double[] mean, double[] nv_samples) {
		for(int i=0; i<n_visible; i++) {
			mean[i] = propdown(h0_sample, i, vbias[i]);
			nv_samples[i] = binomial(1, mean[i]);
		}
	}
	
	public double propup(double[] layer_input, double[] w, double b) {
		double pre_sigmoid_activation = 0.0;
		for(int j=0; j<n_visible; j++) {
			pre_sigmoid_activation += w[j] * layer_input[j];
		}
		pre_sigmoid_activation += b;
		return sigmoid(pre_sigmoid_activation);
	}
	
	public double propdown(double[] h0_sample, int i, double b) {
	  double pre_sigmoid_activation = 0.0;
	  for(int j=0; j<n_hidden; j++) {
	    pre_sigmoid_activation += W[j][i] * h0_sample[j];
	  }
	  pre_sigmoid_activation += b;
	  return sigmoid(pre_sigmoid_activation);
	}
	
	public void gibbs_hvh(double[] h0_sample, double[] nv_means, double[] nv_samples, double[] nh_means, double[] nh_samples) {
	  sample_v_given_h(h0_sample, nv_means, nv_samples);
	  sample_h_given_v(nv_samples, nh_means, nh_samples);
	}


	public void reconstruct(double[] v, double[] reconstructed_v) {
	  double[] h = new double[n_hidden];
	  double pre_sigmoid_activation;
	
	  for(int i=0; i<n_hidden; i++) {
	    h[i] = propup(v, W[i], hbias[i]);
	  }
	
	  for(int i=0; i<n_visible; i++) {
	    pre_sigmoid_activation = 0.0;
	    for(int j=0; j<n_hidden; j++) {
	      pre_sigmoid_activation += W[j][i] * h[j];
	    }
	    pre_sigmoid_activation += vbias[i];
	
	    reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
	  }	
	}	
}
