import java.time.ZoneId;

public enum Airport {

	OHARE		( "ORD", 41.9803, - 87.9047, "US/Central" ),
	KENNEDY		( "JFK", 40.6418, - 73.7810, "US/Eastern" ),
	LA			( "LAX", 33.9428, -118.4100, "US/Pacific" ),
	DENVER		( "DEN", 39.8493, -104.6738, "US/Mountain" ),
	MIAMI		( "MIA", 25.7923, - 80.2823, "US/Eastern" ),
	SEATTLE		( "SEA", 47.4435, -122.3017, "US/Pacific" ),
	ANCHORAGE	( "ANC", 61.1769, -149.9906, "US/Alaska" ),
	ATLANTA		( "ATL", 33.6324, - 84.4333, "US/Eastern" ),
	DALLAS		( "DFW", 32.8990, - 97.0036, "US/Central" ),
	ORLANDO		( "MCO", 28.4230, - 81.3115, "US/Eastern" ),
	VEGAS		( "LAS", 36.0931, -115.1482, "US/Pacific" ),
	CHARLOTTE	( "CLT", 35.2163, - 80.9539, "US/Eastern" );
	
	public final String code;
	public final Double lat;
	public final Double lon;
	public final ZoneId zoneId;
	
	private Airport(final String code, final Double lat, final Double lon, final String id) {
		this.code = code;
		this.lat = lat;
		this.lon = lon;
		this.zoneId = ZoneId.of(id);
	}
}