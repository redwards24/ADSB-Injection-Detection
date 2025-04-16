import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class GenerateRandom {

	public static void main(String[] args) {
		
		Arrays.asList(Path.of(Main.COMPLETE_SETS_PATH).toFile().listFiles()).forEach(file -> {
			final StateVectorList data = StateVectorIO.read(file.getAbsolutePath()).get();
			
			// Generate a random set
			final List<StateVectorList> randLists = data.groupBy(sv -> new HourAirlineKey(sv.hour(), sv.getAirline()))
				.values().stream()
				.map(list -> list.getRandomList(list.size()))
				.toList();
			final StateVectorList rand = new StateVectorList(Main.TOTAL_TARGET);
			boolean run = true;
			int index = 0;
			while(run) {
				for(var list: randLists) {
					if(rand.size() == Main.TOTAL_TARGET) {
						run = false;
						break; 
					}
					if(index < list.size()) {
						rand.add(list.get(index)); 
					}
				}
				++index;
			}
			
			// Write random set to a file
			StateVectorIO.write(rand,String.format(
							"%s%s%s_rand.csv", 
							Main.RANDOM_SETS_PATH, File.separator,
							file.getName().replace(".csv", "")));
		});
		
		

	}
	
}
