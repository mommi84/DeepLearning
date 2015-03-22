package com.github.mommi84.deeplearning.dbn;

import java.util.Random;

public class HiddenLayerMod {
	public int N;
	public int n_in;
	public int n_out;
	public double[][] W;
	public double[] b;
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
	
	
	
	public HiddenLayerMod(int N, int n_in, int n_out, double[][] W, double[] b, Random rng) {
		this.N = N;
		this.n_in = n_in;
		this.n_out = n_out;
		
		if(rng == null)	this.rng = new Random(1234);
		else this.rng = rng;
	
		if(W == null) {
			this.W = new double[n_out][n_in];
			double a = 1.0 / this.n_in;
			
			for(int i=0; i<n_out; i++) {
				for(int j=0; j<n_in; j++) {
					this.W[i][j] = uniform(-a, a);
				}
			}
		} else {
			this.W = W;
		}
		
		if(b == null) this.b = new double[n_out];
		else this.b = b;
	}
	
	public double output(double[] prev_layer_input, double[] w, double b) {
		double linear_output = 0.0;
		for(int j=0; j<n_in; j++) {
			linear_output += w[j] * prev_layer_input[j];
		}
		linear_output += b;
		return sigmoid(linear_output);
	}
	
	public void sample_h_given_v(double[] prev_layer_input, double[] layer_input) {
		for(int i=0; i<n_out; i++) {
			layer_input[i] = binomial(1, output(prev_layer_input, W[i], b[i]));
		}
	}
}
