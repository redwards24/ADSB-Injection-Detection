
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Script to write specific columns to a csv file.
 */
public class FormatSets {
	
	public static void main(String[] args) {
		
		final List<Integer> cols = 
				Arrays.asList(
						StateVector.LAT,
						StateVector.LON,
						StateVector.VELOCITY,
						StateVector.HEADING,
						StateVector.VERTRATE,
						StateVector.BAROALTITUDE,
						StateVector.GEOALTITUDE,
						StateVector.CLASSIFICATION
				);
		
		final StateVectorList all = new StateVectorList();
		
		Arrays.asList(Path.of(Combine.COMBINED_SETS_PATH).toFile().listFiles()).forEach(file -> {
			System.out.println("Formatting: " + file.getName());
			final StateVectorList data = StateVectorIO.read(file.getAbsolutePath()).get();
			StateVectorIO.writeColumns(data, file.getAbsolutePath(), cols);
			all.addAll(data);
		});
		
		StateVectorIO.writeColumns(all, "training_set.csv", cols);
	}
	
}
