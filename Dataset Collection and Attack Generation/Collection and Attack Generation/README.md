# Dataset Collection and Attack Generation
This folder contains the code for collecting authentic ADS-B samples from the [OpenSky Network](https://opensky-network.org/) as well as generating attack samples using Python scripts.
The "java" folder contains the scripts to collect the data and the "python" folder contains the attack scripts.
## Dataset Collection
Java 21 was used for the collection of authentic data. 
The project is setup as a series of scripts, which should be ran in the order of:
1. Main
2. Combine
3. Attack
4. CreateAirportSets
5. CreateTrainingSet

Due to query limitations and hardware limitations, the process of querying, filtering, attacking, and creating the final dataset is split into multiple scripts.

Main.java contains global script variables that are used to control how the data, from OpenSky, is queried, filtered, and stored. The data is first fetched from OpenSky, then filtered locally. The entire filtered set is saved along with subsets of randomly selected samples. The randomly selected samples are used later for machine learning and deep learning.

Since Main.java queries one day of one airport at a time, Combine.java is used to combine the files for each day from a single airport into a single file.

Attack.java is used to perform PM, GI, and VD attacks on the complete sets. Similarly to the authentic samples, the attack samples are randomly selected from the larger pool of attack samples and saved to a file.

CreateAirportSets.java is used to combine the authentic and attack samples into a singular, airport specific file.

CreateTrainingSet.java is used to combine the authentic and attack samples from every airport into a single file.

## Dataset Attacks
Python 3.13 was used when generating the attacks on the authentic samples. The three attack files can be found in the python folder. PM.py is for path modification, GI.py for ghost injection, and VD.py for velocity drift. The java scripts are setup to call these files to automatically generate the attacks.
