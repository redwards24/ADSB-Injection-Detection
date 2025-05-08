import java.io.File;
import java.time.LocalDateTime;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import opensky.StateVector;
import opensky.StateVectorIO;
import opensky.StateVectorList;
import opensky.Trino;
import util.Airline;
import util.Airport;

/**
 * Start of Collection Process
 * 
 * This file will:
 *     1. Query OpenSky
 *     2. Filter Airport, 
 *     3. 
 */
public class Main {
     
	//
	// Script Variables
	//
	
	// Trino Username
	static final String USERNAME = "";
	
	// List of airlines to be filtered
	static final List<Airline> AIRLINES = Airline.toList();
	
	// The airport that data will come from
	static final Airport AIRPORT = Airport.OHARE;
	
	// The boundary box around the airport
	static final int RADIUS_MILES = 15;
	static final double RADIUS_KM = RADIUS_MILES * 1.609;
	static final Boundary BOX = toBounds(AIRPORT.lat, AIRPORT.lon, RADIUS_KM);
	
	
	// Start and Stop time in local time zone
	static final ZonedDateTime START = LocalDateTime.of(
			2024,	// year 
			4,		// month
			5,		// day
			0,		// hour
			0 		// minute
		).atZone(AIRPORT.zoneId);
	static final int TOTAL_DAYS = 10;
	
	// Maximum age of message.
	// If MAX_DELAY = 3, then no messages with a position older than 3 seconds will be collected
	static final int MAX_DELAY = 3;
	
	// List of features to remove duplicates
	static final List<Integer> COLUMNS = 
			Arrays.asList(
					StateVector.ICAO24,
					StateVector.LAT,
					StateVector.LON,
					StateVector.VELOCITY,
					StateVector.HEADING,
					StateVector.VERTRATE,
					StateVector.CALLSIGN,
					StateVector.BAROALTITUDE,
					StateVector.GEOALTITUDE
			);
	
	
	static final int TOTAL_TARGET = 2100;

	static final Trino TRINO = new Trino(USERNAME);
	
	static final String DIR = System.getProperty("user.dir");
	static final String DATA_DIR = DIR + File.separator + "data";
	static final String COMPLETE_SETS_PATH = DATA_DIR + File.separator + "complete_sets";
	static final String RANDOM_SETS_PATH   = DATA_DIR + File.separator + "random_sets";
	
	
	public static void main(String[] args) {

		IntStream.range(0, TOTAL_DAYS).forEach(day -> { // for each day
			
			// Create new date
			final ZonedDateTime date = START.plusDays(day);
			
			// Build query
			final String query = new StringBuilder()
					.append("select ")
					.append(	"time, icao24, lat, lon, velocity, heading, vertrate, callsign, onground, ")
					.append(	"alert, spi, squawk, baroaltitude, geoaltitude, lastposupdate, lastcontact, hour ")
					.append("from ")
					.append(	"minio.osky.state_vectors_data4 ")
					.append("where ")
					.append(String.format(
										"hour in (%s) and ", 
										IntStream.range(0, 24).boxed()
										.map(i -> toEpoch(date) + i*3600)
										.map(Object::toString)
										.collect(Collectors.joining(","))))
					.append(String.format(
										"substring(callsign, 1, 3) in (%s) and ", 
										AIRLINES.stream()
											.map(a -> "\'" + a.code.toUpperCase() + "\'")
											.collect(Collectors.joining(",")))) 
					.append(String.format(
										"lat < %f and lat > %f and lon < %f and lon > %f and ", 
										BOX.north(),
										BOX.south(), 
										BOX.east(), 
										BOX.west()))
					.append(String.format(
										"time - lastposupdate < %d ", 
										MAX_DELAY)) 
					.append("\r\n")
					.toString();
			
			// Execute query
			TRINO.executeQuery(query).ifPresent(res -> {
				
				// Apply more filters locally
				final StateVectorList data = res
						.filterNulls()
						.filterDuplicates(COLUMNS)
						.filterBoundaryRadius(AIRPORT.lat, AIRPORT.lon, RADIUS_KM);
				
				// Write complete set to a file
				StateVectorIO.write(data,String.format(
								"%s%s%s_%s.csv", 
								COMPLETE_SETS_PATH, File.separator,
								AIRPORT, formatDate(date)));
				
				// Generate a random set
				final List<StateVectorList> randLists = data.groupBy(sv -> new HourAirlineKey(sv.hour(), sv.getAirline()))
				.values().stream()
				.map(list -> list.getRandomList(list.size()))
				.toList();
				final StateVectorList rand = new StateVectorList(TOTAL_TARGET);
				boolean run = true;
				int index = 0;
				while(run) {
					for(var list: randLists) {
						if(rand.size() == TOTAL_TARGET) {
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
								"%s%s%s_%s_rand.csv", 
								RANDOM_SETS_PATH, File.separator,
								AIRPORT, formatDate(date)));
			});
		});
	}



	private static long toEpoch(final ZonedDateTime time) {
		final long epoch = time.toEpochSecond();
		return epoch - (epoch % 3600);
	}
	
	private static Boundary toBounds(final double clat, final double clon, final double distance)	{
		final double[] bounds = new double[4];
		final double degToRad = Math.PI / 180;
		final double radToDeg = 180 / Math.PI;		
		final double lat1 = clat * degToRad;
		final double lon1 = clon * degToRad;		
		final double earthPolarRadiusKM = 6371.0;		
		final double angDist = distance / earthPolarRadiusKM;	
		final double sinLat1 = Math.sin(lat1);
		final double cosLat1 = Math.cos(lat1);		
		final double sinAngDist = Math.sin(angDist);
		final double cosAngDist = Math.cos(angDist);		
		bounds[0] = (Math.asin( sinLat1 * cosAngDist + cosLat1 * sinAngDist *  1)) * radToDeg;
		bounds[1] = (Math.asin( sinLat1 * cosAngDist + cosLat1 * sinAngDist * -1)) * radToDeg;
		bounds[2] = (lon1 + Math.atan2( 1 * sinAngDist * cosLat1 , cosAngDist - sinLat1 * sinLat1)) * radToDeg;
		bounds[3] = (lon1 + Math.atan2(-1 * sinAngDist * cosLat1 , cosAngDist - sinLat1 * sinLat1)) * radToDeg;		
		if(bounds[0] > 90.0) bounds[0] = 90.0;		
		if(bounds[1] < -90.0) bounds[1] = -90.0;		
		if(bounds[2] >= 180.0) bounds[2] = -180.0 + (bounds[2] - 180.0);		
		if(bounds[3] < -180.0) bounds[3] = 180.0 - (-1 * bounds[3] - 180);
		return new Boundary(bounds[0], bounds[1], bounds[2], bounds[3]);
	}
	
	static record Boundary(
			double north,
			double south,
			double east,
			double west
		) {
	}
	
	static String formatDate(final ZonedDateTime date) {
		return DateTimeFormatter.ofPattern("MMM-d-yyyy").format(date);
	}
}
record HourAirlineKey(int hour, Airline airline){}






