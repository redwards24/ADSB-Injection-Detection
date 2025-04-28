from haversine import haversine, Unit
import pandas as pd
import csv  
import math
import random
import sys

#Original Dataset file - NY radius 50 Km
file=sys.argv[1]
df = pd.read_csv(file, header=0)

#Dataset file after gradual drift simulation
dest = sys.argv[2]
f=open(dest, 'w', encoding='latin-1', newline='')
writer = csv.writer(f)
# write the header
cl= list(df.columns)
cl.append('class')
writer.writerow(cl)


r_earth =6356752.3142# in m

#Attack Parameters
drift=20 # degrees
nbTotalAttack= 150000 #2790 #5579


# Group data by icao to simulate attack
for icao24, data in df.groupby('icao24'):
    allMsgIcao= data
    nbMsgIcao = len(allMsgIcao)
    attack_ratio = nbMsgIcao*4//10
    middleIndex= nbMsgIcao//3
    
    if(middleIndex <=1):
        attackStart =1
    else:
        attackStart = random.randint(1,middleIndex)
    
    
    prevOrigData=list()
    index=0
    nbAttack=0
    wasAttacked = True

    #print(nbTotalAttack)
    #print(nbAttack)
    for i,d in allMsgIcao.iterrows():
        dl=d.tolist()
        if (index < attackStart or nbAttack>=attack_ratio or nbTotalAttack <=0):
            wasAttacked = False
        else:
            #dl[2] = Lat, dl[3] = Long, dl[5] = heading

            nbAttack+=1
            nbTotalAttack-=1
            location1 =(prevOrigData[2], prevOrigData[3])
            location2 =(dl[2], dl[3])
            
            #Calculate the distance between the two coordinates
            dist=haversine(location1, location2, unit= Unit.METERS)
            driftAngle=(drift+dl[5])%360

            dl[2] = math.degrees(math.asin(math.sin(math.radians(prevAttackData[2]))*math.cos(dist/r_earth) + math.cos(math.radians(prevAttackData[2]))*math.sin(dist/r_earth)*math.cos(math.radians(driftAngle))))
            dl[3] = math.degrees(math.radians(prevAttackData[3]) + math.atan2((math.sin(math.radians(driftAngle)) * math.sin(dist/r_earth)*math.cos(math.radians(prevAttackData[2]))), (math.cos(dist/r_earth)-math.sin(math.radians(prevAttackData[2]))*math.sin(math.radians(dl[2])))))

            dl[5] = driftAngle
            dl[-1] = 1

            wasAttacked = True
        
        prevOrigData=d.tolist()
        prevAttackData=dl
        index+=1

        if wasAttacked:
            writer.writerow(dl)
