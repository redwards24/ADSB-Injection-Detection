import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import opensky.StateVectorIO;
import opensky.StateVectorList;
import util.Airport;


/**
 * Combines authentic and attack samples into a single file by airport.
 */
public class CreateAirportSets {

	public static void main(String[] args) {
		
		
		// Rename Combined Sets
		Arrays.asList(Path.of(Combine.COMBINED_SETS_PATH).toFile().listFiles()).forEach(file -> {
			try {
				Files.move(Path.of(file.getAbsolutePath()), Path.of(file.getAbsolutePath().replace("combined_rand", "authentic")));
			} catch (IOException e) {
				e.printStackTrace();
			}
		});
		
		final List<String> files = Arrays.asList(
				Path.of(Main.RANDOM_SETS_PATH).toFile().listFiles())
				.stream()
				.map(file -> file.getAbsolutePath())
				.toList();
		
		Arrays.asList(Airport.values()).forEach(airport -> {
			System.out.println("Creating set for: " + airport.toString());
			final StateVectorList data = new StateVectorList();
			files.stream().filter(str -> str.contains(airport.toString())).forEach(file -> {
				data.addAll(StateVectorIO.read(file).get());
			});
			if(data.size() != 0)
				StateVectorIO.write(data, Combine.COMBINED_SETS_PATH + "\\" + airport.toString() +"_authentic_attack.csv");
		});
	}
}
