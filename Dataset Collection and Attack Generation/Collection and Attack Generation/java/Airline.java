import java.util.Arrays;
import java.util.List;

public enum Airline {
	
	DELTA("DAL"),
	AMERICAN("AAL"),
	SOUTHWEST("SWA"),
	ALASKA("ASA"),
	UNITED("UAL");
	
	public final String code;
	
	private Airline(final String code) {
		this.code = code;
	}
	
	public static Airline toAirline(final String code) {
		switch (code) {
		case "DAL": return DELTA;
		case "AAL": return AMERICAN;
		case "SWA": return SOUTHWEST;
		case "ASA": return ALASKA;
		case "UAL": return UNITED;
		default:
			throw new IllegalArgumentException("Unexpected value: " + code);
		}
	}
	
	public static List<Airline> toList() {
		return Arrays.asList(values());
	}
}