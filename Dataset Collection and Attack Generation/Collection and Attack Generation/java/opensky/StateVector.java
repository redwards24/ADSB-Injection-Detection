package opensky;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import util.Airline;
import util.Flight;


/**
 * Represents a state vector obtained from the OpenSky-Network.
 */
public record StateVector(	
		Integer time,
		String  icao24,
		Double  lat,
		Double  lon,
		Double  velocity,
		Double  heading,
		Double  vertrate,
		String  callsign,
		Boolean onground,
		Boolean alert,
		Boolean spi,
		String  squawk,
		Double  baroaltitude,
		Double  geoaltitude,
		Double  lastposupdate,
		Double  lastcontact,
		Integer hour,
		Type    classification
) {
	
	public enum Type {
		AUTHENTIC(0),
		PATH_MODIFICATION(1),
		GHOST_INJECTION(2),
		VELOCITY_DRIFT(3);
		
		public final int id;
		
		private Type(final int id) {
			this.id = id;
		}
		
		public static Type toType(final int id) {
			switch (id) {
			case 0: return AUTHENTIC;
			case 1: return PATH_MODIFICATION;
			case 2: return GHOST_INJECTION;
			case 3: return VELOCITY_DRIFT;
			default:
				throw new IllegalArgumentException("Unexpected value: " + id);
			}
		}
		
		@Override
		public String toString() {
			return id + "";
		}
	}
	
	public Airline getAirline() {
		return Airline.toAirline(callsign.substring(0, 3));
	}

	
	public Flight.Key getFlightKey() {
		return Flight.createKey(this);
	}

	public List<Object> asList() {
		return Arrays.asList(
				time, icao24, lat, lon, velocity, heading, vertrate,
				callsign, onground, alert, spi, squawk, baroaltitude, geoaltitude,
				lastposupdate, lastcontact, hour, classification);
	}
	
	public boolean containsNulls() {
		for(final Object obj: asList()) 
			if(obj == null) 
				return true;
		return false;
	}
	
	@Override
	public String toString() {
		return asList().stream()
				.map(StateVector::checkNullString)
				.collect(Collectors.joining(","));
	}
	
	private static String checkNullString(final Object object) {
		if(object == null) 
			return "NULL";
		return object.toString();
	}
	
	public boolean containsNulls(final List<Integer> columns) {
		final List<Object> list = asList();
		for(final Integer i: columns)
			if(list.get(i) == null)
				return true;
		return false;
	}
	
	public String toString(final List<Integer> columns) {
		final List<Object> list = new ArrayList<Object>(columns.size());
		columns.forEach(i -> list.add(asList().get(i)));
		return list.stream()
				.map(StateVector::checkNullString)
				.collect(Collectors.joining(","));
	}

	
	public final static int TIME 			=  0;
	public final static int ICAO24 			=  1;
	public final static int LAT 			=  2;
	public final static int LON				=  3;
	public final static int VELOCITY 		=  4;
	public final static int HEADING 		=  5;
	public final static int VERTRATE 		=  6;
	public final static int CALLSIGN 		=  7;
	public final static int ONGROUND 		=  8;
	public final static int ALERT 			=  9;
	public final static int SPI 			= 10;
	public final static int SQUAWK 			= 11;
	public final static int BAROALTITUDE 	= 12;
	public final static int GEOALTITUDE 	= 13;
	public final static int LASTPOSUPDATE 	= 14;
	public final static int LASTCONTACT 	= 15;
	public final static int HOUR 			= 16;
	public final static int CLASSIFICATION  = 17;
	
	
	public static StateVector fromString(final String string) {
		final String[] values = string.split(",");
		return new StateVector(
					toInt(values[0]),
					values[1],
					toDouble(values[2]),
					toDouble(values[3]),
					toDouble(values[4]),
					toDouble(values[5]),
					toDouble(values[6]),
					values[7],
					toBoolean(values[8]),
					toBoolean(values[9]),
					toBoolean(values[10]),
					values[11],
					toDouble(values[12]),
					toDouble(values[13]),
					toDouble(values[14]),
					toDouble(values[15]),
					toInt(values[16]),
					values.length == 18
						? Type.toType(toInt(values[17]))
						: Type.AUTHENTIC
				);
	}
	
	private static Integer toInt(final String string) {
		try {
			return Integer.parseInt(string);
		} catch (Exception e) {
			return null;
		}
	}
	
	private static Boolean toBoolean(final String string) {
		try {
			return Boolean.parseBoolean(string);
		} catch (Exception e) {
			return null;
		}
	}
	
	private static Double toDouble(final String string) {
		try {
			return Double.parseDouble(string);
		} catch (Exception e) {
			return null;
		}
	}

}