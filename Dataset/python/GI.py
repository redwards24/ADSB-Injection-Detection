from haversine import haversine, Unit
import pandas as pd
import csv
import math
import random
import sys

# Data set for O'Hare
file = sys.argv[1]
df = pd.read_csv(file, encoding='latin-1', header=0)

# Dataset file after gradual drift simulation - this is writing a new file
dest = sys.argv[2]
f = open(dest, 'w', encoding='latin-1', newline='')
writer = csv.writer(f)

# write the header
cl = list(df.columns)
writer.writerow(cl)

r_earth = 6356752.3142  # in m

# Attack Parameters
drift = 20  # degrees

# m
minRangeAttack = 6000
maxRangeAttack = 10000

# Balt
minBalt = -22.86
MaxBalt = 13106.4

# Alt
minAlt = -30.48
MaxAlt = 13533.12

# Alt dist range
minDist = 500
maxDist = 3000

# m
geoAltitudeDrift = 50
altitudeRange = 50

# heading
headingRadius = 10

# velocity
minVelocity = 1
maxVelocity = 190

# vertrate
minVertrate = -12.5171
maxVertrate = 16.5405

# Setup Message Targets
nbTotalAttack = 150000 # Could be 0.4

# For each Aircraft in Full Authentic Data Set
# 	icao24 = unique aircraft identifier
# 	data = list of data for aircraft with icao24
for icao24, data in df.groupby('icao24'):
    allMsgIcao= data
    nbMsgIcao = len(allMsgIcao)
    attack_ratio = nbMsgIcao*4//10
    middleIndex= nbMsgIcao//3
    
    if(middleIndex <=1):
        attackStart =1
    else:
        attackStart = random.randint(1,middleIndex)

    index=0
    nbAttack=0
    wasAttacked = True

    # For each entry in data
    #	i = row number
    #	d = row data
    for i, d in data.iterrows():
        dl=d.tolist()

        if (index < attackStart or nbAttack>=attack_ratio or nbTotalAttack <=0):
            wasAttacked = False
        else:
            nbAttack+=1
            nbTotalAttack-=1
            # Lat/Lon
            dist = (maxRangeAttack - minRangeAttack) * random.random() + minRangeAttack
            drift = 360 * random.random()
            if (random.random() < 0.5):
                drift = -1 * drift
            driftAngle = (drift + dl[5]) % 360

            dl[2] = math.degrees(
                math.asin(math.sin(math.radians(dl[2]))) * math.cos(dist / r_earth) + math.cos(
                    math.radians(dl[2])) * math.sin(dist / r_earth) * math.cos(math.radians(driftAngle)))
            dl[3] = math.degrees(math.radians(dl[3]) + math.atan2((math.sin(
                math.radians(driftAngle)) * math.sin(dist / r_earth) * math.cos(math.radians(dl[2]))), (
                                                                                        math.cos(
                                                                                            dist / r_earth) - math.sin(
                                                                                    math.radians(
                                                                                        dl[2])) * math.sin(
                                                                                    math.radians(dl[2])))))

            # velocity
            dl[4] = (maxVelocity - minVelocity) * random.random() + minVelocity
            # heading
            dl[5] = driftAngle
            # vertrate
            dl[6] = (maxVertrate - minVertrate) * random.random() + minVertrate

            # baroaltitude
            altitudeDrift = (maxDist - minDist) * random.random() + minDist
            if (random.random() < 0.5):
                altitudeDrift = -1 * altitudeDrift

            dl[12] = dl[12] + altitudeDrift

            # geoaltitude
            dl[13] = dl[13] + altitudeDrift + geoAltitudeDrift * random.random()

            # lastposupdate
            dl[14] = dl[14] + (0.999 + 0.999) * random.random() - 0.999

            # lastcontact
            dl[15] = dl[15] + (0.999 + 0.999) * random.random() - 0.999

            # class
            dl[17] = 2
            wasAttacked=True
        index+=1

        if wasAttacked:
             writer.writerow(dl)
