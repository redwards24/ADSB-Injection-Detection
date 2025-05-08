package opensky;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Read and Write StateVectors.
 */
public class StateVectorIO {

	final static String HEADER = "time,icao24,lat,lon,velocity,heading,vertrate,callsign,onground,alert,spi,squawk,baroaltitude,geoaltitude,lastposupdate,lastcontact,hour,class";
	
	
	public static boolean write(final StateVectorList data, final String filename) {
		return writeColumns(data, filename, IntStream.range(0, 18).boxed().toList());
	}
	
	public static boolean writeColumns(final StateVectorList data, final String filename, final List<Integer> columns) {
		try (final BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
			final String[] split = HEADER.split(",");
			final String newHeader = columns.stream().map(i -> split[i]).collect(Collectors.joining(","));
			bw.write(newHeader);
			bw.newLine();
			for(final StateVector entry: data) {
				bw.write(entry.toString(columns));
				bw.newLine();
			}
			return true;
		} catch (IOException e)	{
			e.printStackTrace();
			return false;
		}
	}
	
	public static Optional<StateVectorList> read(final String file) {
		try{
			return Optional.of(
				Files.lines(new File(file).toPath())
				.skip(1)
				.map(StateVector::fromString)
				.collect(Collectors.toCollection(StateVectorList::new))
			);
		} catch (IOException e) {
			e.printStackTrace();
			return Optional.empty();
		}
	}	
}