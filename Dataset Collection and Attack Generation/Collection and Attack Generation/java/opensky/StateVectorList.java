package opensky;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import util.Airline;
import util.Airport;
import util.Flight;
import util.FlightList;
import util.Flight.Key;

public class StateVectorList extends ArrayList<StateVector>{
	
	public StateVectorList() {
	}
	
	public StateVectorList(final int size) {
		super(size);
	}
	
	public FlightList getFlights() {
		return stream()
				.collect(
						Collectors.groupingBy(
								StateVector::getFlightKey, 
								Collectors.summarizingInt(StateVector::time)))
				.entrySet().stream()
				.map(e -> {
					final Flight.Key key = e.getKey();
					final IntSummaryStatistics val = e.getValue();
					return new Flight(
							key.icao24(), 
							key.callsign(), 
							val.getMin(), 
							val.getMax(), 
							(int)val.getCount(), 
							key.classification());
				})	
				.collect(Collectors.toCollection(FlightList::new));
	}
	
	public List<StateVector.Type> getUniqueClassifications() {
		final Set<Integer> set = new HashSet<>();
		return stream()
				.filter( sv -> set.add( sv.classification().id ) )
				.map( sv -> sv.classification() )
				.collect( Collectors.toList() );
	}
	
	public List<String> getUniqueICAO() {
		final Set<String> set = new HashSet<String>();		
		return stream()
				.filter(sv -> set.add(sv.icao24()))
				.map(sv -> sv.icao24())
				.toList();
	}
	
	public void sortByTimeAscending() {
		Collections.sort(this, new Comparator<StateVector>() {
			@Override public int compare(StateVector o1, StateVector o2) {
				final int t1 = o1.time();
				final int t2 = o2.time();				
				     if(t1 <  t2) return -1;
				else if(t1 == t2) return  0;
				else              return  1; }});
	}


	public StateVectorList filterDuplicates(final List<Integer> columns) {
		final Set<String> set = new HashSet<>();
		return stream()
				.filter(sv -> set.add(sv.toString(columns)))
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}

	public StateVectorList filterNulls() {
		return stream()
				.filter(sv -> ! sv.containsNulls())
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	
	public StateVectorList filterNulls(final List<Integer> columns) {
		return stream()
				.filter(sv -> ! sv.containsNulls(columns))
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	
	

	
	public StateVectorList filterBoundaryBox(final double n, final double s, final double e, final double w) {
		return stream()
				.filter(sv -> sv.lat() > n || sv.lat() < s || sv.lon() > e || sv.lon() < w)
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	
	public StateVectorList filterAirport(final Airport airport, final double radius) {
		return filterBoundaryRadius(airport.lat, airport.lon, radius);
	}
	
	public StateVectorList filterBoundaryRadius(final double clat, final double clon, final double radius) {
		final double c = Math.PI / 180.0;
		final double r = 6371.0;		
		final double lat2 = clat * c;
		final double lon2 = clon * c;		
		return stream()
				.filter(sv -> {
					final double lat1 = sv.lat() * c;
					final double lon1 = sv.lon() * c;					
					final double dist = Math.acos( Math.sin(lat1) * Math.sin(lat2) + Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon2 - lon1) ) * r;					
					return dist <= radius; })
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	
	public StateVectorList filterDateTime(final long start, final long stop) {
		return stream()
				.filter(sv -> sv.time() >= start && sv.time() < stop)
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	
	public StateVectorList filterLastContact(final double interval) {
		return stream()
				.filter(sv -> sv.time() - sv.lastcontact() > interval)
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	
	public StateVectorList filterLastPosUpdate(final double interval) {
		return stream()
				.filter(sv -> sv.time() - sv.lastposupdate() > interval)
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	

	public StateVectorList filterAirlines(final List<Airline> airlines)	{
		final List<String> codes = airlines.stream()
									.map(a -> a.code)
									.collect(Collectors.toList());		
		return stream()
				.filter(sv -> codes.contains(sv.callsign().substring(0, 3)))
				.collect(Collectors.toCollection(() -> new StateVectorList()));
	}
	

	public StateVectorList getRandomList(final int count) {
		final List<Integer> list = new ArrayList<>(this.size());		
		IntStream.range(0, size()).forEach(list::add);
		Collections.shuffle(list);		
		final StateVectorList result = new StateVectorList();
		IntStream.range(0, count).forEach(i -> result.add(this.get(list.get(i))));
		return result;
	}
	
	public <T extends Record> Map<T, StateVectorList> groupBy(final Function<StateVector, T> func) {
		return stream()
				.collect(Collectors.groupingBy(
						func::apply, 
						Collectors.mapping(
								Function.identity(),
								Collectors.toCollection(StateVectorList::new))));		
	}
	
	
}










