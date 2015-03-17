package org.aksw.tsoru.softeng;

import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import com.github.mommi84.deeplearning.dbn.DeepBeliefNets;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

/**
 * @author Tommaso Soru <t.soru@informatik.uni-leipzig.de>
 *
 */
public class PredictValues {

	public static void main(String[] args) throws IOException {
				
		CSVReader reader = new CSVReader(new FileReader("data/throughput-4.csv"), ',', '"', CSVWriter.NO_ESCAPE_CHARACTER);
		reader.readNext(); // skip header
		String[] nextLine;
		
		final int TRAIN_ROWS = 178, IN = 24, OUT = 15;
		int[][] train_X = new int[TRAIN_ROWS][IN];
		int[][] train_Y = new int[TRAIN_ROWS][OUT];

		final int TEST_ROWS = 20;
		int[][] test_X = new int[TEST_ROWS][IN];
		double[][] test_Y = new double[TEST_ROWS][OUT];

		System.out.println("== TRAIN ==");

		for (int i=0; i<TRAIN_ROWS && (nextLine = reader.readNext()) != null; i++) {
			// training data
			train_X[i] = inputBits(nextLine);
			print(train_X[i]);
			train_Y[i] = outputBits(nextLine[5]);
			print(train_Y[i]);
		}
		
		System.out.println("== TEST ==");
		
		for (int i=0; i<TEST_ROWS && (nextLine = reader.readNext()) != null; i++) {
			// test data
			test_X[i] = inputBits(nextLine);
			print(test_X[i]);
			test_Y[i] = outputDoubleBits(nextLine[5]);
			print(test_Y[i]);
		}

		reader.close();
		
		Random rng = new Random(123);
		
		double pretrain_lr = 1.0;
		int pretraining_epochs = 5000;
		int k = 1;
		double finetune_lr = 0.1;
		int finetune_epochs = 2000;
		
		int train_N = TRAIN_ROWS;
		int test_N = TEST_ROWS;
		int n_ins = IN;
		int n_outs = OUT;
		int[] hidden_layer_sizes = {3};
		int n_layers = hidden_layer_sizes.length;
		
		
		// construct DBN
		DeepBeliefNets dbn = new DeepBeliefNets(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers, rng);
		
		// pretrain
		dbn.pretrain(train_X, pretrain_lr, k, pretraining_epochs);
		
		// finetune
		dbn.finetune(train_X, train_Y, finetune_lr, finetune_epochs);
		
		double[][] predict_Y = new double[TEST_ROWS][OUT];
		
		// test
		for(int i=0; i<test_N; i++) {
			dbn.predict(test_X[i], predict_Y[i]);
			for(int j=0; j<n_outs; j++) {
				System.out.print( predict_Y[i][j] + "\t");
			}
			System.out.println();
		}

				
	}
	
	private static void print(double[] arr) {
		for(double x : arr)
			System.out.print(x+" ");
		System.out.println();
	}

	private static void print(int[] arr) {
		for(int x : arr)
			System.out.print(x);
		System.out.println();
	}

	private static double[] outputDoubleBits(String string) {
		/*
		 * output bits: 15
		 */
		double[] bits = new double[15];
		
		int x = Integer.parseInt(string);
		String xbit = Integer.toBinaryString(x);
		for(int j=0; j<xbit.length(); j++)
			bits[14-j] = Integer.parseInt( "" + xbit.charAt(xbit.length()-j-1) );
		
		return bits;
	}

	/**
	 * Concat of input bits.
	 * Test:
		System.out.print(Integer.toBinaryString(692)+"---");
		System.out.print(Integer.toBinaryString(19)+"---");
		System.out.print(Integer.toBinaryString(205)+"---");
		int[] b = inputBits(new String[]{"31","692","19","205"});
		for(int b1 : b)
			System.out.print(b1);

	 * @param decimal
	 * @return
	 */
	private static int[] inputBits(String[] decimal) {
		/*
		 * input bits: 24 (10,5,9)
		 */
		int[] bits = new int[24];
		int[] indexes = {9, 14, 23};
		
		for(int i=0; i<indexes.length; i++) {
			int x = Integer.parseInt(decimal[i+1]);
			String xbit = Integer.toBinaryString(x);
			for(int j=0; j<xbit.length(); j++)
				bits[indexes[i]-j] = Integer.parseInt( "" + xbit.charAt(xbit.length()-j-1) );
		}
		
		return bits;
	}

	private static int[] outputBits(String decimal) {
		/*
		 * output bits: 15
		 */
		int[] bits = new int[15];
		
		int x = Integer.parseInt(decimal);
		String xbit = Integer.toBinaryString(x);
		for(int j=0; j<xbit.length(); j++)
			bits[14-j] = Integer.parseInt( "" + xbit.charAt(xbit.length()-j-1) );
		
		return bits;
	}


}
