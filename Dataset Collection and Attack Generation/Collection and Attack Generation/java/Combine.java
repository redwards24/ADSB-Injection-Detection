import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import opensky.StateVectorIO;
import opensky.StateVectorList;


/**
 * Combines the randomly selected authentic samples by airport.
 */
public class Combine {

	
	static final String COMBINED_SETS_PATH = Main.DATA_DIR + File.separator + "combined_sets";
	
	public static void main(String[] args) {
		
		try {
			
			final StateVectorList combinedData = new StateVectorList();
			
			// For each file path in data/random_sets
			Arrays.asList(new File(Main.RANDOM_SETS_PATH).list()).forEach(path -> {
				
				// If data comes from selected airport, add to combined set
				if(path.contains(Main.AIRPORT.toString())) {
					StateVectorIO.read(Main.RANDOM_SETS_PATH + File.separator + path).ifPresent(combinedData::addAll);
				}
			});
			
			Files.createDirectories(Path.of(COMBINED_SETS_PATH));
			
			StateVectorIO.write(combinedData, String.format(
					"%s%s%s_combined_rand.csv", 
					COMBINED_SETS_PATH, File.separator, Main.AIRPORT));
			
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
}
