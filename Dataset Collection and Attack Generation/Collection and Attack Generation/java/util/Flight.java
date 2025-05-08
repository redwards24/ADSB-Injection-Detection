package util;
import opensky.StateVector;

/**
 * Represents a specific flight flown by a specific aircraft.
 */
public record Flight(
		String icao24, 
		String callsign, 
		int start, 
		int stop, 
		int messageCount,
		StateVector.Type type) {

	
	public static Flight.Key createKey(final StateVector sv) {
		return createKey(sv.icao24(), sv.callsign(), sv.classification());
	}
	
	public Key getKey() {
		return createKey(this.icao24, this.callsign, this.type);
	}
	
	private static Key createKey(final String icao24, final String callsign, final StateVector.Type classification) {
		return new Key(icao24, callsign, classification);
	}
	
	public record Key(
		String icao24,
		String callsign,
		StateVector.Type classification
	) {}
}