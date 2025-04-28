import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Optional;
import java.util.Properties;

public class Trino {
	
	
	private static final String DATABASE_URL = "jdbc:trino://trino.opensky-network.org:443";

	private final String username;
	
	public Trino(final String username) {
		this.username = username;
	}

	private Connection getConnection() throws SQLException {
		final Properties properties = new Properties();
		properties.setProperty("user", username);
		properties.setProperty("externalAuthentication", "true");
		properties.setProperty("externalAuthenticationTokenCache", "MEMORY");
		return DriverManager.getConnection(DATABASE_URL, properties);	
	}
	
	public Optional<StateVectorList> executeQuery(final String query) {
		try {
			// Create connection and query for data.
			final Connection conn = getConnection();
			
			// Execute Query
			final ResultSet rs = conn.createStatement().executeQuery(query);
			
			// Create StateVectorList
			final StateVectorList result = new StateVectorList();
			
			while(rs.next()) {
				result.add(new StateVector(		
					rs.getInt("time"),
					rs.getString("icao24"),
					rs.getDouble("lat"),
					rs.getDouble("lon"),
					rs.getDouble("velocity"),
					rs.getDouble("heading"),
					rs.getDouble("vertrate"),
					rs.getString("callsign"),
					rs.getBoolean("onground"),
					rs.getBoolean("alert"),
					rs.getBoolean("spi"),
					rs.getString("squawk"),
					rs.getDouble("baroaltitude"),
					rs.getDouble("geoaltitude"),
					rs.getDouble("lastposupdate"),
					rs.getDouble("lastcontact"),
					rs.getInt("hour"),
					StateVector.Type.AUTHENTIC
				));
			}
			
			// Close connection
			conn.close();

			// Operation was successful
			return Optional.of(result);
		}
		catch(SQLException e) {
			e.printStackTrace();
			return Optional.empty();
		}
	}
}