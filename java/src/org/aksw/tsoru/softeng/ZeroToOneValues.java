package org.aksw.tsoru.softeng;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

import com.github.mommi84.deeplearning.dbn.DeepBeliefNetsMod;

/**
 * @author Tommaso Soru <t.soru@informatik.uni-leipzig.de>
 *
 */
public class ZeroToOneValues {
	
	static PrintWriter pw = null;

	static final int TRAIN_ROWS = 178, IN = 4, OUT = 1, TEST_ROWS = 20;

	public static void main(String[] args) throws IOException {
			
		pw = new PrintWriter(new File("dbn-test-"+System.currentTimeMillis()+".log"));

		print("DBN Started.");
		
		CSVReader reader = new CSVReader(new FileReader("data/throughput-0to1.csv"), ',', CSVWriter.NO_QUOTE_CHARACTER, CSVWriter.NO_ESCAPE_CHARACTER);
		reader.readNext(); // skip header
		String[] nextLine;
		
		double[][] train_X = new double[TRAIN_ROWS][IN];
		double[][] train_Y = new double[TRAIN_ROWS][OUT];

		double[][] test_X = new double[TEST_ROWS][IN];
		double[][] test_Y = new double[TEST_ROWS][OUT];

		print("== TRAINING SET ==");

		for (int i=0; i<TRAIN_ROWS && (nextLine = reader.readNext()) != null; i++) {
			// training data
			for(int j=0; j<IN; j++)
				train_X[i][j] = Double.parseDouble(nextLine[j]);
			train_Y[i][0] = Double.parseDouble(nextLine[4]);
			print(train_X[i], train_Y[i]);
		}
		
		print("\n== TEST SET ==");
		
		for (int i=0; i<TEST_ROWS && (nextLine = reader.readNext()) != null; i++) {
			// test data
			for(int j=0; j<IN; j++)
				test_X[i][j] = Double.parseDouble(nextLine[j]);
			test_Y[i][0] = Double.parseDouble(nextLine[4]);
			print(test_X[i], test_Y[i]);
		}

		reader.close();
		
		Random rng = new Random(456);
		int k = 1;
		
//		for(int exp=-1; exp<0; exp++) {
//			double lr = Math.pow(10, exp);
//			for(int pre=100; pre<5000; pre*=2) {
//				int tune = pre / 2;
//				for(int hid=4; hid<=128; hid*=2) {
//					int[] hidden_layer_sizes = {hid, hid};
//					int n_layers = hidden_layer_sizes.length;
//
//					System.out.println("Running DBN(lr="+lr+",pre="+pre+",tune="+tune+",k=1,hid="+hid+",hlsize=2)...");
//
//					// construct DBN
//					DeepBeliefNetsMod dbn = new DeepBeliefNetsMod(TRAIN_ROWS, IN, hidden_layer_sizes, OUT, n_layers, rng);
//
//					// pretrain
//					dbn.pretrain(train_X, lr, k, pre);
//					
//					// finetune
//					dbn.finetune(train_X, train_Y, lr, tune);
//					
//					double[][] predict_Y = new double[TEST_ROWS][OUT];
//					
//					// test
//					for(int i=0; i<TEST_ROWS; i++) {
//						dbn.predict(test_X[i], predict_Y[i]);
//						print(predict_Y[i][0]+"\t"+test_Y[i][0]);
//						for(double p : predict_Y[i])
//						if(p != 1.0) {
//							print("DBN(lr="+lr+",pre="+pre+",tune="+tune+",k=1,hid="+hid+",hlsize=2) has found a value = "+p);
//							System.out.println("DBN(lr="+lr+",pre="+pre+",tune="+tune+",k=1,hid="+hid+",hlsize=2) has found a value = "+p);
//						}
//					}
//					
//				}
//					
//			}
//		}
		
		singleExec(rng, k, train_X, train_Y, test_X, test_Y);
		
		pw.close();
	}
	

//	@SuppressWarnings("unused")
	private static void singleExec(Random rng, int k, double[][] train_X, double[][] train_Y, double[][] test_X, double[][] test_Y) {
		double pretrain_lr = 0.1;
		int pretraining_epochs = 200;
		double finetune_lr = 0.1;
		int finetune_epochs = 100;
		int[] hidden_layer_sizes = {200, 500};
		
		int n_layers = hidden_layer_sizes.length;
		
		// construct DBN
		DeepBeliefNetsMod dbn = new DeepBeliefNetsMod(TRAIN_ROWS, IN, hidden_layer_sizes, OUT, n_layers, rng);
		
		// pretrain
		dbn.pretrain(train_X, pretrain_lr, k, pretraining_epochs);
		
		// finetune
		dbn.finetune(train_X, train_Y, finetune_lr, finetune_epochs);
		
		double[][] predict_Y = new double[TEST_ROWS][OUT];
		
		// test
		print("\n== RESULTS ==");
		for(int i=0; i<TEST_ROWS; i++) {
			dbn.predict(test_X[i], predict_Y[i]);
			System.out.println(predict_Y[i][0]+"\t"+test_Y[i][0]);
		}
	}


	private static void print(String string) {
		pw.write(string+"\n");
	}


	private static void print(double[] x, double[] y) {
		for(double a : x)
			pw.write(a+"\t");
		pw.write("| "+y[0]+"\n");
	}



}
