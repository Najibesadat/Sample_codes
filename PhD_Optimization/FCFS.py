##############################################################################       
### Import packages
import numpy as np 
import numpy 
import pandas as pd 
from numpy import linalg
import math
import random,string   

###############################################################################
######Define FCFS function

def FCFS(P, Beds, Rooms, EVS, Tran, Time, IU_original, IU_D, useful_Beds):    
    S_be = 50  # cleaning time
    Npq = len(P)
    LOS = 48*60         ## Lenght of stay of each patient is 2 days
    EVS1 = EVS.copy(deep=True)
    TH = 300
    T_E = Time + TH
    Tran1 = Tran.copy(deep=True)
    P1 = P.copy(deep=True)
    useful_Beds1 = useful_Beds.copy(deep=True)
    Rooms1 = Rooms.copy(deep=True)
    
    ##############################################################################       
    #### assignment result table  
    P1 = P1.sort_values('Admission time',ascending=True)
    P1 = P1.reset_index()
    P1 = P1.drop(['index'], axis=1)

    FCFS_P = pd.DataFrame(index=range(Npq), columns=['#.Patient','Patient.ID','Gender','Admission time','Priority','P_IU','#.Bed','Bed.ID','Room','Room_Gender','IU','IU_D','Transport.ID','tc','Tran.Availability.T','Transporting.S.T','T_P','Waiting.T', 'LOS'])
    for i in range(Npq):
        FCFS_P.loc[i,['Gender','Priority','LOS']] = P1.loc[i,['Gender','Priority', 'LOS']]
        FCFS_P.loc[i,'Patient.ID'] = P1.loc[i,'ID']
        FCFS_P.loc[i,'#.Patient'] = int(np.where(IU_original['ID'] == str(FCFS_P.loc[i,'Patient.ID']))[0])
        FCFS_P.loc[i,'Admission time'] = P1.loc[i,'Admission time']
        FCFS_P.loc[i,'P_IU'] = P1.loc[i,'IU']
        
    FCFS_P = FCFS_P.sort_values('Admission time',ascending=True)
    FCFS_P = FCFS_P.reset_index()
    FCFS_P = FCFS_P.drop(['index'], axis=1)
    
    FCFS_Beds = pd.DataFrame(index=range(len(useful_Beds1)), columns=['#.Bed','Bed.ID','Status','Bed.Availability.T','P_Gender','Room','IU','D','EVS','EVS_location','EVS.Availability.T','Traveling.S.T','Traveling.E.T','Cleaning.S.T','Cleaning.E.T'])  # E.Availability.T=Earlieast.Availability.T
    for j in range(len(useful_Beds1)):
        FCFS_Beds.loc[j,['Bed.ID','Status','P_Gender','Room','IU','D']] = useful_Beds1.loc[j,['Bed.ID','Status','P_Gender','Room','IU','D']]
        FCFS_Beds.loc[j,'Bed.Availability.T'] = useful_Beds1.loc[j,'Availability time']
        FCFS_Beds.loc[j,'#.Bed'] = int(np.where(Beds['Bed.ID'] == FCFS_Beds.loc[j,'Bed.ID'])[0])   
        if FCFS_Beds.loc[j,'Status'] =='clean': FCFS_Beds.loc[j,'Cleaning.E.T'] = FCFS_Beds.loc[j,'Bed.Availability.T']
    FCFS_Beds = FCFS_Beds.sort_values('Bed.Availability.T',ascending=True)
    FCFS_Beds = FCFS_Beds.reset_index()
    FCFS_Beds = FCFS_Beds.drop(['index'], axis=1)
    
    EVS_journey = pd.DataFrame(index=range(0), columns=['EVS.ID','Availability time','Location','#.Bed','Bed.ID','IU','Idle.S.T', 'Idle.E.T','Traveling.S.T','Traveling.E.T', 'Cleaning.S.T', 'Cleaning.E.T'])

    ##############################################################################       
    ### Bed-Patient Assignment        
    for p in range(Npq):        
        modified_B = useful_Beds1.copy(deep=True)
        B_IU = []
        R_IUG = []  ## Rooms number that have different gender of p              
        for j in range(len(modified_B)):
            if modified_B.loc[j,'IU'] == FCFS_P.loc[p,'P_IU'] and (sum(FCFS_P['Bed.ID'].isin([str(modified_B.loc[j,'Bed.ID'])]))==0) : B_IU.append(j)    ## candidate beds are the beds that are in the preffered IU and above that
        for i in range(len(B_IU)):
            for k in B_IU:
                if (Rooms1.loc[modified_B.loc[k,'Room'],'Gender'] != FCFS_P.loc[p,'Gender'] and Rooms1.loc[modified_B.loc[k,'Room'],'Gender'] != 0):                    
                    R_IUG.append(modified_B.loc[k,'Room'])
                    B_IU.remove(k)                                                ## Gender constraint: if bed is in a room with different gender of patient , the bed will bed removed from candidate beds                        

        for r in range(len(R_IUG)):
            for i in R_IUG:
                 BR_IUG = np.where(modified_B['Room'] == i)[0].tolist()
                 modified_B['Availability time'] = modified_B['Availability time'].astype(int)
                 B_IU.append(modified_B.loc[modified_B.iloc[BR_IUG]['Availability time'].idxmax()].name)
                     
        if len(B_IU)==0 :
            for j in range(len(modified_B)):
                if modified_B.loc[j,'IU'] == FCFS_P.loc[p,'P_IU']+1 : B_IU.append(j)
            for i in range(len(B_IU)):
                for k in B_IU:
                    if Rooms1.loc[modified_B.loc[k,'Room'],'Gender'] != FCFS_P.loc[p,'Gender'] and Rooms1.loc[modified_B.loc[k,'Room'],'Capacity']>1: 
                        B_IU.remove(k)                                                ## Gender constraint: if bed is in a room with different gender of patient , the bed will bed removed from candidate beds                                                                                    
        if len(B_IU)==0 :
            for j in range(len(modified_B)):
                if modified_B.loc[j,'IU'] >= FCFS_P.loc[p,'P_IU'] : B_IU.append(j)
            for i in range(len(B_IU)):
                for k in B_IU:
                    if Rooms1.loc[modified_B.loc[k,'Room'],'Gender'] != FCFS_P.loc[p,'Gender'] and Rooms1.loc[modified_B.loc[k,'Room'],'Capacity']>1: 
                        B_IU.remove(k)                                                ## Gender constraint: if bed is in a room with different gender of patient , the bed will bed removed from candidate beds                                                                                    
                
        modified_B['Availability time'] = modified_B['Availability time'].astype(int)
        min_B = modified_B.loc[modified_B.iloc[B_IU]['Availability time'].idxmin()].name             
        modified_B['Availability time'] = useful_Beds1['Availability time'].copy(deep=True)
        
        FCFS_P.loc[p,['Bed.ID','Room','IU']] = modified_B.loc[min_B,['Bed.ID','Room','IU']] 
        FCFS_P.loc[p,'Room_Gender'] = modified_B.loc[min_B,'P_Gender']
        FCFS_P.loc[p,'IU_D'] = modified_B.loc[min_B,'D']
        FCFS_P.loc[p,'#.Bed'] = int(np.where(Beds['Bed.ID'] == FCFS_P.loc[p,'Bed.ID'])[0])                      
    ##############################################################################       
    #### Bed-EVS assignment   
        if modified_B.loc[min_B,'Status'] == 'clean': 
            FCFS_P.loc[p,'tc'] = modified_B.loc[min_B,'Availability time']
        else:
            modified_E = EVS1.copy(deep=True)
            for j in range(len(modified_E)):
                if modified_E.loc[j,'Location'] == 'ED':
                    TravelT = FCFS_P.loc[p,'IU_D']
                    modified_E.loc[j,'Traveling.E.T'] = modified_E.loc[j,'Availability time'] + TravelT 
                elif modified_E.loc[j,'Location'] == FCFS_P.loc[p,'IU']:
                    TravelT = 0
                    modified_E.loc[j,'Traveling.E.T'] = modified_E.loc[j,'Availability time']
                else:  
                    TravelT = IU_D.loc[modified_E.loc[j,'Location']-1,str(int(FCFS_P.loc[p,'IU']))]
                    modified_E.loc[j,'Traveling.E.T'] = modified_E.loc[j,'Availability time']+ TravelT
 
            modified_E['Traveling.E.T'] = modified_E['Traveling.E.T'].astype(int)
            min_E = modified_E.loc[modified_E['Traveling.E.T'].idxmin()].name 
            B_idx = int(np.where(FCFS_Beds['Bed.ID'] == modified_B.loc[min_B,'Bed.ID'])[0])   
            FCFS_Beds.loc[B_idx,'Bed.Availability.T'] = modified_B.loc[min_B,'Availability time']
            FCFS_Beds.loc[B_idx,'EVS'] = modified_E.loc[min_E,'EVS.ID']
            FCFS_Beds.loc[B_idx,'EVS_location'] = modified_E.loc[min_E,'Location']
            FCFS_Beds.loc[B_idx,'EVS.Availability.T'] = modified_E.loc[min_E,'Availability time']
            FCFS_Beds.loc[B_idx,'Traveling.S.T'] = max(FCFS_Beds.loc[B_idx,'Bed.Availability.T']-TravelT, FCFS_Beds.loc[B_idx,'EVS.Availability.T'])
            FCFS_Beds.loc[B_idx,'Traveling.E.T'] = FCFS_Beds.loc[B_idx,'Traveling.S.T'] + TravelT
            FCFS_Beds.loc[B_idx,'Cleaning.S.T'] = FCFS_Beds.loc[B_idx,'Traveling.E.T']
            FCFS_Beds.loc[B_idx,'Cleaning.E.T'] = FCFS_Beds.loc[B_idx,'Cleaning.S.T'] + S_be
            FCFS_P.loc[p,'tc'] = FCFS_Beds.loc[B_idx,'Cleaning.E.T']
            
            ### Update EVS staff
            EVS1.loc[min_E, 'Availability time'] = FCFS_Beds.loc[B_idx,'Cleaning.E.T']
            EVS1.loc[min_E, 'Location'] = FCFS_Beds.loc[B_idx,'IU']            
            ### Insert information in EVS_journey
            idxrow = int(len(EVS_journey))
            EVS_journey.loc[idxrow,'EVS.ID'] = modified_E.loc[min_E,'EVS.ID']
            EVS_journey.loc[idxrow,'Location'] = modified_E.loc[min_E,'Location']
            EVS_journey.loc[idxrow,'Availability time'] = modified_E.loc[min_E,'Availability time']
            EVS_journey.loc[idxrow,'#.Bed'] = FCFS_Beds.loc[B_idx,'#.Bed']
            EVS_journey.loc[idxrow,'Bed.ID'] = FCFS_Beds.loc[B_idx,'Bed.ID']   
            EVS_journey.loc[idxrow,'IU'] = FCFS_Beds.loc[B_idx,'IU']                
            EVS_journey.loc[idxrow,'Traveling.S.T'] = FCFS_Beds.loc[B_idx,'Traveling.S.T']
            EVS_journey.loc[idxrow,'Traveling.E.T'] = FCFS_Beds.loc[B_idx,'Traveling.E.T'] 
            EVS_journey.loc[idxrow,'Cleaning.S.T'] = FCFS_Beds.loc[B_idx,'Traveling.E.T']
            EVS_journey.loc[idxrow,'Cleaning.E.T'] = FCFS_Beds.loc[B_idx,'Cleaning.E.T']
            EVS_journey.loc[idxrow,'Idle.S.T'] = EVS_journey.loc[idxrow,'Availability time']
            EVS_journey.loc[idxrow,'Idle.E.T'] = EVS_journey.loc[idxrow,'Traveling.S.T']                              
    ##############################################################################       
    #### Transport-Patient assignment 
        modified_Tr = Tran1
        modified_Tr['Availability time'] = modified_Tr['Availability time'].astype(int)
        min_Tr = modified_Tr.loc[modified_Tr['Availability time'].idxmin()].name      
        modified_Tr['Availability time'] = Tran1['Availability time']

        FCFS_P.loc[p,'Transport.ID'] = modified_Tr.loc[min_Tr,'Tran.ID']        
        FCFS_P.loc[p,'Tran.Availability.T'] = modified_Tr.loc[min_Tr,'Availability time']    
        FCFS_P.loc[p,'Transporting.S.T'] = max(FCFS_P.loc[p,'Tran.Availability.T'],(FCFS_P.loc[p,'tc']-FCFS_P.loc[p,'IU_D']),FCFS_P.loc[p,'Admission time'])
        FCFS_P.loc[p,'T_P'] = FCFS_P.loc[p,'Transporting.S.T'] + FCFS_P.loc[p,'IU_D']
        FCFS_P.loc[p,'Waiting.T'] = FCFS_P.loc[p,'T_P'] - FCFS_P.loc[p,'Admission time'] -  FCFS_P.loc[p,'IU_D']
        if  FCFS_P.loc[p,'Waiting.T'] < 0.5 : FCFS_P.loc[p,'Waiting.T'] = 0
        ### Update Transport staff
        Tran1.loc[min_Tr,'Availability time'] = FCFS_P.loc[p,'T_P'] + FCFS_P.loc[p,'IU_D']

        ## Update Beds    
        ubidx = int(np.where(useful_Beds1['Bed.ID'] == FCFS_P.loc[p,'Bed.ID'])[0]) 
        useful_Beds1.loc[ubidx,'Availability time'] = FCFS_P.loc[p,'T_P'] + FCFS_P.loc[p,'LOS']
        useful_Beds1.loc[ubidx,'Status'] = 'occupied'
        useful_Beds1.loc[ubidx,'P_Gender'] = FCFS_P.loc[p,'Gender']
        Rooms1.loc[useful_Beds1.loc[ubidx,'Room'],'Gender']= FCFS_P.loc[p,'Gender']
        
    ##############################################################################       
    #### additional dirty Bed-EVS assignment     
    useful_Beds1 = useful_Beds1.sort_values('Availability time',ascending=True)
    useful_Beds1 = useful_Beds1.reset_index()
    useful_Beds1 = useful_Beds1.drop(['index'], axis=1)

    ## Add one row for each EVS to calculate Idle time in end of work.
    for j in range(len(EVS1)):
        idxRow = int(len(EVS_journey))
        EVS_journey.loc[idxRow,'EVS.ID'] = EVS1.loc[j,'EVS.ID']
        EVS_journey.loc[idxRow,'Location'] = EVS1.loc[j,'Location']
        EVS_journey.loc[idxRow,'Availability time'] = EVS1.loc[j,'Availability time']
        EVS_journey.loc[idxRow,'Idle.S.T'] = EVS_journey.loc[idxRow,'Availability time']
        EVS_journey.loc[idxRow,'Idle.E.T'] = T_E
    ###########
    
    for i in range(len(useful_Beds1)):
        if useful_Beds1.loc[i,'Status'] == 'dirty' or useful_Beds.loc[i,'Status'] =='occupied':
            Tneeded = EVS1
            candidate_E = []
            for j in range(len(EVS1)):
                if EVS1.loc[j,'Location'] == 'ED':
                    TravelT1 = useful_Beds1.loc[i,'D']
                elif EVS1.loc[j,'Location']  == useful_Beds1.loc[i,'IU']:
                    TravelT1 = 0
                else:  
                    TravelT1 = IU_D.loc[EVS1.loc[j,'Location']-1,str(int(useful_Beds1.loc[i,'IU']))]
                Tneeded.loc[j,'Tneeded']= S_be + TravelT1
                
            for k in range(len(EVS_journey)):
                if EVS_journey.loc[k,'Idle.E.T']-max(useful_Beds1.loc[i,'Availability time'],EVS_journey.loc[k,'Idle.S.T']) >= Tneeded.loc[int(np.where(Tneeded['EVS.ID'] == EVS_journey.loc[k,'EVS.ID'])[0]),'Tneeded']:
                    candidate_E.append(k)
            if len(candidate_E) > 0 :
                EVS_journey['Idle.S.T'] = EVS_journey['Idle.S.T'].astype(int)
                BestE = EVS_journey.loc[EVS_journey.iloc[candidate_E]['Idle.S.T'].idxmin()].name
                Bidx = int(np.where(FCFS_Beds['Bed.ID'] == useful_Beds1.loc[i,'Bed.ID'])[0])               
                FCFS_Beds.loc[Bidx,'EVS'] = EVS_journey.loc[BestE,'EVS.ID']
                FCFS_Beds.loc[Bidx,'EVS_location'] = EVS_journey.loc[BestE,'Location']
                FCFS_Beds.loc[Bidx,'EVS.Availability.T'] = EVS_journey.loc[BestE,'Availability time']
                TravelT = Tneeded.loc[int(np.where(Tneeded['EVS.ID'] == FCFS_Beds.loc[Bidx,'EVS'])[0]),'Tneeded'] - S_be
                FCFS_Beds.loc[Bidx,'Traveling.S.T'] = max(FCFS_Beds.loc[Bidx,'Bed.Availability.T']-TravelT, FCFS_Beds.loc[Bidx,'EVS.Availability.T'])
                FCFS_Beds.loc[Bidx,'Traveling.E.T'] = FCFS_Beds.loc[Bidx,'Traveling.S.T'] + TravelT
                FCFS_Beds.loc[Bidx,'Cleaning.S.T'] = FCFS_Beds.loc[Bidx,'Traveling.E.T']
                FCFS_Beds.loc[Bidx,'Cleaning.E.T'] = FCFS_Beds.loc[Bidx,'Cleaning.S.T'] + S_be
                
                ### Update EVS staff
                if FCFS_Beds.loc[Bidx,'Cleaning.E.T'] > EVS1.loc[int(np.where(EVS1['EVS.ID'] == EVS_journey.loc[BestE,'EVS.ID'])[0]), 'Availability time']:
                    EVS1.loc[int(np.where(EVS1['EVS.ID'] == EVS_journey.loc[BestE,'EVS.ID'])[0]), 'Availability time'] = FCFS_Beds.loc[Bidx,'Cleaning.E.T']
                    EVS1.loc[int(np.where(EVS1['EVS.ID'] == EVS_journey.loc[BestE,'EVS.ID'])[0]), 'Location'] = FCFS_Beds.loc[Bidx,'IU']
                    
                ### Insert information in EVS_journey
                bestidxrow = int(len(EVS_journey))
                EVS_journey.loc[bestidxrow,'EVS.ID'] = FCFS_Beds.loc[Bidx,'EVS']
                EVS_journey.loc[bestidxrow,'Location'] = FCFS_Beds.loc[Bidx,'EVS_location']
                EVS_journey.loc[bestidxrow,'Availability time'] = FCFS_Beds.loc[Bidx,'EVS.Availability.T']
                EVS_journey.loc[bestidxrow,'#.Bed'] = FCFS_Beds.loc[Bidx,'#.Bed']
                EVS_journey.loc[bestidxrow,'Bed.ID'] = FCFS_Beds.loc[Bidx,'Bed.ID']   
                EVS_journey.loc[bestidxrow,'IU'] = FCFS_Beds.loc[Bidx,'IU']                
                EVS_journey.loc[bestidxrow,'Traveling.S.T'] = FCFS_Beds.loc[Bidx,'Traveling.S.T']
                EVS_journey.loc[bestidxrow,'Traveling.E.T'] = FCFS_Beds.loc[Bidx,'Traveling.E.T'] 
                EVS_journey.loc[bestidxrow,'Cleaning.S.T'] = FCFS_Beds.loc[Bidx,'Traveling.E.T']
                EVS_journey.loc[bestidxrow,'Cleaning.E.T'] = FCFS_Beds.loc[Bidx,'Cleaning.E.T']
                EVS_journey.loc[bestidxrow,'Idle.S.T'] = EVS_journey.loc[bestidxrow,'Availability time']
                EVS_journey.loc[bestidxrow,'Idle.E.T'] = EVS_journey.loc[bestidxrow,'Traveling.S.T']
                
                EVS_journey.loc[BestE,'Idle.E.T'] = EVS_journey.loc[bestidxrow,'Traveling.S.T']  # update selected EVS in EVS_journey
    
                ## Add one row for each EVS to calculate Idle time in end of work.
                idxRow = int(len(EVS_journey))
                EVS_journey.loc[idxRow,'EVS.ID'] = EVS1.loc[j,'EVS.ID']
                EVS_journey.loc[idxRow,'Location'] = EVS1.loc[j,'Location']
                EVS_journey.loc[idxRow,'Availability time'] = EVS1.loc[j,'Availability time']
                EVS_journey.loc[idxRow,'Idle.S.T'] = EVS_journey.loc[idxRow,'Availability time']
                EVS_journey.loc[idxRow,'Idle.E.T'] = T_E
    ############################################################################## 
    FCFS_Beds = FCFS_Beds.set_index('Bed.ID')
    FCFS_Beds = FCFS_Beds.reindex(index=useful_Beds['Bed.ID'])
    FCFS_Beds = FCFS_Beds.reset_index()
     
    FCFS_Beds = FCFS_Beds[FCFS_Beds['Cleaning.E.T'].notnull()]      
    FCFS_Beds = FCFS_Beds.reset_index()
    FCFS_Beds = FCFS_Beds.drop(['index'], axis=1)     
    ##############################################################################                   
            
    FCFS_objVal1 = sum(FCFS_P['Priority']*(FCFS_P['T_P']-FCFS_P['Admission time']))   #
    FCFS_objVal2 = 0.001*sum(FCFS_P['tc'])
    return (FCFS_Beds, FCFS_P, FCFS_objVal1, FCFS_objVal2)      

