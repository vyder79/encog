package pl.vyder;

import org.encog.Encog;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.som.SOM;
import org.encog.neural.som.training.basic.BasicTrainSOM;
import org.encog.neural.som.training.basic.neighborhood.NeighborhoodSingle;

public class SOMTest {
	
	public static double SOM_INPUT[][] = { 
		{ -1.0, -1.0, 1.0, 1.0 }, 
		{ 1.0, 1.0, -1.0, -1.0 },
		{ -1.0, -1.0, 1.0, -1.0 }, 
		{ 1.0, 1.0, -1.0, 1.0 },
		{ -1.0, -1.0, -1.0, -1.0 },
		{ -1.2, -0.67, 0.9, 0.8 },
		{ 1.0, -1.0, -1.0, 1.0 }
		};
	
	public static void main(String args[])
	{	
		// create the training set
		MLDataSet training = new BasicMLDataSet(SOM_INPUT,null);
		
		// Create the neural network.
		SOM network = new SOM(4,7);
		network.reset();
		
		BasicTrainSOM train = new BasicTrainSOM(
				network,
				0.7,
				training,
				new NeighborhoodSingle());
				
		int iteration = 0;
		
		for(iteration = 0;iteration<=100;iteration++)
		{
			train.iteration();
			System.out.println("Iteration: " + iteration + ", Error:" + train.getError());
		}
		
		MLData data1 = new BasicMLData(SOM_INPUT[0]);
		MLData data2 = new BasicMLData(SOM_INPUT[1]);
		MLData data3 = new BasicMLData(SOM_INPUT[2]);
		MLData data4 = new BasicMLData(SOM_INPUT[3]);
		MLData data5 = new BasicMLData(SOM_INPUT[4]);
		MLData data6 = new BasicMLData(SOM_INPUT[5]);
		MLData data7 = new BasicMLData(SOM_INPUT[6]);
		System.out.println("Pattern 1 winner: " + network.classify(data1));
		System.out.println("Pattern 2 winner: " + network.classify(data2));
		System.out.println("Pattern 3 winner: " + network.classify(data3));
		System.out.println("Pattern 4 winner: " + network.classify(data4));
		System.out.println("Pattern 5 winner: " + network.classify(data5));
		System.out.println("Pattern 6 winner: " + network.classify(data6));
		System.out.println("Pattern 6 winner: " + network.classify(data7));
		Encog.getInstance().shutdown();
	}
}
