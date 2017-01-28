package anizotrophy;

import org.encog.neural.som.SOM;

/**
 * @author Tomek
 *
 */
public class AnizotrophySOM {
	
	private SOM som_1;
	
	private SOM som_2;
	
	private int version;
	
	
	/**
	 * @param version of network, 1 or 2
	 * @return network by given number, first if '1' and second if anything else as parameter
	 */
	public SOM getNetwork(int version) {
		return version == 2 ? getSom_2() : getSom_1();
	}
	
	
	/**
	 * @return always first network
	 */
	public SOM getNetwork() {
		return getSom_1();
	}	

	public AnizotrophySOM(SOM som_1, SOM som_2) {
		super();
		this.som_1 = som_1;
		this.som_2 = som_2;
		this.version = 1;
	}

	public SOM getSom_1() {
		return som_1;
	}

	public void setSom_1(SOM som_1) {
		this.som_1 = som_1;
	}

	public SOM getSom_2() {
		return som_2;
	}

	public void setSom_2(SOM som_2) {
		this.som_2 = som_2;
	}

	public int getVersion() {
		return version;
	}

	public void setVersion(int version) {
		this.version = version;
	}

}
