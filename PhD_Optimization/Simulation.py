##############################################################################       
### Import packages
import numpy as np 
import numpy.random
import numpy 
import pandas as pd 
from numpy import linalg
from itertools import repeat
import math
import random,string   
import scipy.stats as stats 
from matplotlib import pyplot as plt

###############################################################################
######Define Initialization function

def Initialization(Np, N_days, L, Chance, G_chance, NR, TNE, NE, EVS_Shift, NT, IU_types, Np_IU, NR_IU, IU_ED,IU2IU, S_be, bedstates, Bstatus_chance, P_R_capacity, P_priority, LOS, Bed_Utlztn) :
    
    ##### Patients
    Gender = ['Female','Male']
    interval_times = np.random.exponential(scale=L, size= Np)
    arrival_times = np.cumsum(interval_times)
    IU_chance = np.random.binomial(1, Chance, Np)
    
    ED_Patients= pd.DataFrame(index=range(Np), columns=['ID','Gender','Arrival time','Admitted'])

    for i in range(Np): ED_Patients.loc[i,'ID'] = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))    # ID of patients in ED
    ED_Patients['Gender'] = np.random.choice(Gender, size=(Np,), p=G_chance)                                           
    for i in range(Np): ED_Patients.loc[i,'Arrival time'] = arrival_times[i]    # Arrival time of patients to ED
    for i in range(Np): ED_Patients.loc[i,'Admitted'] = IU_chance[i]            # Which ED patient will be admitted to IU
    
    IU_Patients = ED_Patients.loc[ED_Patients['Admitted'] == 1]                 # total patients that will be admitted to IU in a day
    IU_Patients = IU_Patients.reset_index()
    for i in range(len(IU_Patients['ID'])): IU_Patients.loc[i,'Admission time'] = (IU_Patients.loc[i,'Arrival time']) + (np.random.lognormal(4.5, 0.1, 1))      ### Process time of each patient in ED
    IU_Patients = IU_Patients.drop(['Admitted'], axis=1)
    IU_Patients = IU_Patients.drop(['index'], axis=1)

    IU_Patients = IU_Patients.sort_values('Admission time',ascending=True)
    IU_Patients['IU'] = np.random.choice(range(IU_types), size=(len(IU_Patients),), p=Np_IU)+1
    for i in range(len(IU_Patients)):
        IU_Patients.loc[i,'IU_D'] = IU_ED[IU_Patients.loc[i,'IU']-1] 
        IU_Patients.loc[i,'Priority'] = P_priority[IU_Patients.loc[i,'IU']-1]   ## assigning priority weight based on their proper IU
        IU_Patients.loc[i,'LOS'] = LOS
        if IU_Patients.loc[i,'IU']==3 or IU_Patients.loc[i,'IU']==2 : IU_Patients.loc[i,'Ct_p'] = 1 
        else: IU_Patients.loc[i,'Ct_p'] = 0                                 ## for IU3 and IU2 patients, tier constraint is hard for IU1 patients tier constraint is soft
                
    IU_Patients = IU_Patients.reset_index()
    IU_Patients = IU_Patients.drop(['index'], axis=1)  
    
    ##### IU Distances    
    IU_D = pd.DataFrame(numpy.zeros(shape=(1+IU_types,1+IU_types)))  # 0= IU1 , 1= IU2, 2=IU3, 3= ED
    IU_D.loc[0,1]=IU2IU[0] ;  IU_D.loc[1,0]=IU2IU[0]
    IU_D.loc[0,2]=IU2IU[1] ;  IU_D.loc[2,0]=IU2IU[1]
    IU_D.loc[1,2]=IU2IU[2] ;  IU_D.loc[2,1]=IU2IU[2]
    for i in range(len(IU_D)-1):
        IU_D.loc[i,3] = IU_ED[i] ; IU_D.loc[3,i] = IU_ED[i]
    IU_D.columns = ['1','2','3','ED']    
           
    ### Rooms
    Rooms = pd.DataFrame(index=range(NR), columns=['#.Room','IU','Capacity','Gender'])
    Rooms['#.Room']=range(NR)
    Rooms['Capacity']=list(repeat(1, (NR-int(NR/2))))+ list(repeat(2, int(NR/2)))      #np.random.choice(range(2), size=(len(Rooms),), p=P_R_capacity)+1      
    IUNumbers = list(repeat(1, int(math.ceil(NR*float(NR_IU[0])))))+ list(repeat(2, int(math.ceil(NR*float(NR_IU[1]))))) + list(repeat(3, int(math.ceil(NR*float(NR_IU[2])))))
    for r in range(len(Rooms)):
        ru = np.random.choice(IUNumbers, size= 1)
        Rooms.loc[r,'IU'] = int(ru)
        IUNumbers.remove(ru)
    Rooms['Gender'] = np.random.choice(Gender, size=(len(Rooms),), p=G_chance)
                         
    ##### Resources - Beds    
    Nb = sum(Rooms['Capacity'])
    Beds = pd.DataFrame(index=range(Nb), columns=['Bed.ID','Status','Availability time','P_Gender','Room','IU','D'])
    for i in range(Nb): Beds.loc[i,'Bed.ID'] = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))            # ID of each bed

    Rlist = Rooms['#.Room'].tolist()
    caplist = Rooms['Capacity'].tolist()              
    all_Rlist = numpy.repeat(Rlist, caplist)
    all_Rlist = all_Rlist.tolist()
    Beds['Room']= all_Rlist     
    
    for i in range(len(Beds)):
        Beds.loc[i,'IU'] = int(Rooms.loc[int(np.where(Rooms['#.Room'] == Beds.loc[i,'Room'])[0]),'IU'])
    for i in range(Nb): 
        for j in range(len(IU_ED)):            
            if   Beds.loc[i,'IU'] == j+1 : Beds.loc[i,'D'] = IU_ED[j]

    for I in range(IU_types):
        Beds.loc[np.where(Beds['IU'] == I+1)[0],'Status'] = np.random.choice(bedstates, size=((len(np.where(Beds['IU'] == I+1)[0])),), p=[Bed_Utlztn[I], 0.5*(1-Bed_Utlztn[I]), 0.5*(1-Bed_Utlztn[I])])   # p=Bstatus_chance                                        # Status of each EVS staff
    
    for i in range(Nb):
        if Beds.loc[i,'Status'] == 'clean' or Beds.loc[i,'Status'] == 'dirty' :  Beds.loc[i,'Availability time'] = 0   
        if Beds.loc[i,'Status'] == 'occupied' : Beds.loc[i,'Availability time'] = LOS*random.uniform(0, 1)  
    
    for i in range(Nb):
        Beds.loc[i,'P_Gender'] = Rooms.loc[int(np.where(Rooms['#.Room'] == Beds.loc[i,'Room'])[0]),'Gender']
        if Beds.loc[i,'Availability time'] == 0 : Beds.loc[i,'P_Gender'] = 0


    ### update room gender based on beds gender
    for r in range(len(Rooms)):
        rb = np.where(Beds['Room'] == r)[0].tolist()
        rbidx = 0
        for b in rb:
            if Beds.loc[b,'P_Gender']!= Rooms.loc[r,'Gender']:
                rbidx = rbidx+1
        if rbidx == Rooms.loc[r,'Capacity']:  
            if len(np.where(Beds.iloc[rb]['P_Gender'] != 0)[0].tolist()) > 0 :
                gbidx = np.where(Beds.iloc[rb]['P_Gender'] != 0)[0].tolist()
                Rooms.loc[r,'Gender'] = Beds.loc[rb[gbidx[0]],'P_Gender']
            else: Rooms.loc[r,'Gender'] = 0
            
    ##### Resources - EVS    
    All_EVS =  pd.DataFrame(index=range(TNE), columns=['EVS.ID','Availability time', 'Cleaning_Time','Location', 'END_Shift'])
    for i in range(TNE): All_EVS.loc[i,'EVS.ID'] = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))            # ID of each bed
    for j in range(int(TNE/NE)): 
        for i in range(NE):
            All_EVS.loc[i+(j*NE),'Availability time'] = j*EVS_Shift                    # availability time of each EVS staff
            All_EVS.loc[i+(j*NE),'END_Shift'] = (j+1)*EVS_Shift                        # End of shift time for each EVS staff  ## shift time = 8 hours
        
    for i in range(TNE): All_EVS.loc[i,'Cleaning_Time'] = 50                           #np.random.normal (S_be,5)
    for i in range(TNE): All_EVS.loc[i,'Location'] = 'ED'

    for d in range(int(((N_days*1440)/EVS_Shift)-(TNE/NE)+1)):
        k = len(All_EVS)
        for i in range(NE):
            All_EVS.loc[k+i,'EVS.ID'] = All_EVS.loc[i+(d*NE),'EVS.ID']
            All_EVS.loc[k+i,'Cleaning_Time'] = 50
            All_EVS.loc[k+i,'Location'] = 'ED'
            All_EVS.loc[k+i,'Availability time'] = All_EVS.loc[k-1,'END_Shift']                                      # availability time of each EVS staff
            All_EVS.loc[k+i,'END_Shift'] = All_EVS.loc[k+i,'Availability time'] + EVS_Shift                          # End of shift time for each EVS staff  ## shift time = 8 hours
                
    ##### Resources - Transport staff    
    Tran = pd.DataFrame(index=range(NT), columns=['Tran.ID','Availability time','Location'])

    for i in range(NT): Tran.loc[i,'Tran.ID'] = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))            # ID of each bed
    for i in range(NT): Tran.loc[i,'Availability time'] = 0                    # availability time of each EVS staff    
    for i in range(NT): Tran.loc[i,'Location'] = 'ED' 
    
    ##### Write to excel
    writer = pd.ExcelWriter('Information.xlsx')    
    ED_Patients.to_excel(writer,'ED_Patients')     # All patients information came to ED
    IU_Patients.to_excel(writer,'IU_Patients')     # All patients information admitted to IU
    Beds.to_excel(writer,'Beds')       # Resource member availability time
    Rooms.to_excel(writer,'Rooms')     # Resource member Information
    All_EVS.to_excel(writer,'All_EVS')         # Resource member availability time
    Tran.to_excel(writer,'Tran')       # Resource member availability time
    IU_D.to_excel(writer,'IU_D')       # Distances between every two inpatient units        
    writer.save()
    
    Npq= 0
    P = pd.DataFrame(index=range(Npq), columns=['ID','Gender','Admission time','Priority', 'IU','IU_D','Ct_p', 'LOS'])  # Patients in queue
                   
    return (P, IU_Patients, Rooms, Beds, All_EVS, Tran, IU_D)
    








