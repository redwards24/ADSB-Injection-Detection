import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Combines authentic and attack airport sets into a single training set.
 */
public class CreateTrainingSet {
	
	public static void main(String[] args) {
		
		try {
			System.out.print("Creating training set...");
			final List<String> data = new ArrayList<>();
			final List<File> files = Arrays.asList(Path.of(Combine.COMBINED_SETS_PATH).toFile().listFiles());
			for(final File file: files) {
				if(file.getName().contains("attack"))
					data.addAll(Files.lines(file.toPath()).skip(1).toList());
			}
			
			final BufferedWriter bw = new BufferedWriter(new FileWriter(Main.DATA_DIR + "\\training-set.csv"));
			bw.write("lat,lon,velocity,heading,vertrate,baroaltitude,geoaltitude,class");
			bw.newLine();
			for(final String str: data) {
				bw.write(str);
				bw.newLine();
			}
			bw.close();
			System.out.println(" complete.");
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
