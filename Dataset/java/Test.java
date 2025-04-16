import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;


public class Test {
	

	
	public static void main(String[] args) {
		
		try{
			
			final List<List<String>> data =
				Files.lines(new File("OHARE_authentic_attack.csv").toPath())
				.skip(1)
				.map(str -> Arrays.asList(str.split(",")))
				.filter(list -> list.get(list.size()-1).equals("0"))
				.toList();
			
			final BufferedWriter bw = new BufferedWriter(new FileWriter("OHARE_authentic.csv"));
			
			for(List<String> list: data) {
				bw.write(list.stream().collect(Collectors.joining(",")));
				bw.newLine();
			}
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
	}
	
	
}
