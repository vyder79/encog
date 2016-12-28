/*
 * Encog(tm) Java Examples v3.3
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-examples
 *
 * Copyright 2008-2014 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package pl.vyder;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import org.encog.mathutil.randomize.RangeRandomizer;
import org.encog.mathutil.rbf.RBFEnum;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.som.SOM;
import org.encog.neural.som.training.basic.BasicTrainSOM;
import org.encog.neural.som.training.basic.neighborhood.NeighborhoodRBF;

/**
 * A classic SOM example that shows how the SOM groups similar color shades.
 *
 */
public class SomColors extends JFrame implements Runnable {
	
	public static double SOM_INPUT[][] = { 
//			{ -1.0, -0.5, 0.5, 1.0 }, 
//			{ 0.5, 1.0, -1.0, -0.5 },
//			{ -1.0, 1.0, 0.5, -0.5 }, 
//			{ 1.0, 1.0, -1.0, 1.0 },
//			{ -1.0, -1.0, -1.0, -1.0 },
//			{ -1.2, -0.67, 0.9, 0.8 },
			{ 1.0, 1.0, -1.0, 1.0 },
			{ -1.0, -1.0, -1.0, -0.5 },
			{ -1.2, -0.67, 0.9, 0.8 },
			{ 0.0, 0.0, 0.0, 0.0},
			{ 1.0, -1.0, -1.0, 1.0 }
//			randomInputData(),
//			randomInputData(),
//			randomInputData(),
//			randomInputData(),
//			randomInputData()
			};
	
	public static double SOM_VERIFY[][] = { 
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData(),
			randomInputData()
			};

	/**
	 * 
	 */
	private static final long serialVersionUID = -6762179069967224817L;
	private MapPanel map;
	private SOM network;
	private Thread thread;
	private BasicTrainSOM train;
	private NeighborhoodRBF gaussian;

	public SomColors() {
		this.setSize(1040, 1060);
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.network = createNetwork();
		this.getContentPane().add(map = new MapPanel(this));
		this.gaussian = new NeighborhoodRBF(RBFEnum.Gaussian, MapPanel.WIDTH, MapPanel.HEIGHT);
		this.train = new BasicTrainSOM(this.network, 0.01, null, gaussian);
		train.setForceWinner(false);
		this.thread = new Thread(this);
		thread.start();
	}

	public SOM getNetwork() {
		return this.network;
	}

	private SOM createNetwork() {
		SOM result = new SOM(SOM_INPUT[0].length, MapPanel.WIDTH * MapPanel.HEIGHT);
		result.reset();
		return result;
	}

	public static void main(String[] args) {
		SomColors frame = new SomColors();
		frame.setVisible(true);
	}

	public void run() {

		List<MLData> samples = new ArrayList<MLData>();
		for (int i = 0; i < SOM_INPUT.length; i++) {
//			MLData data = new BasicMLData(4);
//			data.setData(0, RangeRandomizer.randomize(-1, 1));
//			data.setData(1, RangeRandomizer.randomize(-1, 1));
//			data.setData(2, RangeRandomizer.randomize(-1, 1));
//			data.setData(3, RangeRandomizer.randomize(-1, 1));
			MLData data = new BasicMLData(SOM_INPUT[i]);

			samples.add(data);
		}

		this.train.setAutoDecay(2000, 0.9, 0.002, 40, 2);

		for (int i = 0; i < 2000; i++) {
			int idx = (int) (Math.random() * samples.size());
			MLData c = samples.get(idx);

			this.train.trainPattern(c);
			this.train.autoDecay();
			this.map.repaint();
			System.out.println("Iteration " + i + "," + this.train.toString());
		}
		
		/* show centers of winners neurons */
		for (int i = 0; i < SOM_INPUT.length; i++) {
			int winnerNumber = network.classify(new BasicMLData(SOM_INPUT[i]));
			System.out.println("Pattern "+ i +" winner: " + winnerNumber);
			this.map.paintSelected(this.map.getGraphics(), winnerNumber, new Color(0, 111, 0));
		}
		
		/* show where are randomized vectors at SOM trained map */
		for (int i = 0; i < SOM_VERIFY.length; i++) {
			int winnerNumber = network.classify(new BasicMLData(SOM_VERIFY[i]));
			System.out.println("Verifying Pattern "+ i +" winner: " + winnerNumber);
			this.map.paintSelected(this.map.getGraphics(), winnerNumber, new Color(0, 255, 0));
		}
	}
	
	private static double[] randomInputData() {
		double[] data = new double[4];
		data[0] = RangeRandomizer.randomize(-1, 1);
		data[1] = RangeRandomizer.randomize(-1, 1);
		data[2] = RangeRandomizer.randomize(-1, 1);
		data[3] = RangeRandomizer.randomize(-1, 1);
		return data;
	}
}
