##############################################################################       
### Import packages
from gurobipy import *
import numpy as np 
import numpy 
import pandas as pd        
from numpy import linalg
import math
import heapq
import random,string        
from FCFS import FCFS
import numpy as np
import scipy.stats as stats 
from matplotlib import pyplot as plt
from decimal import Decimal

###############################################################################
###### Define Optimization function

def Assignment(Numberoftry4OPT, itry, objval1_prev, objval, epsilon, Assignments_0, Tp_prev, PB_prev, P_FixedinTF, P, Beds, Rooms, EVS, NE, Tran, Time, IU_original, IU_D, IU_types, LOS, Balancing, Nbc, RunTimelimit):

    while Numberoftry4OPT < 4 :
        try:
            ### Parameters
            ############################################
            TH = 180    
            S_be = 50                                        # service time for cleaning a bed
            b2b = IU_D.iloc[0:IU_types,0:IU_types]
            alpha = 0.5                                      # fraction of stability solution
        
            Npq = len(P.index)                               # Numer of patients in queue (ready for assigning)    
        
            h = [[1,0.3,0.1],[0,1,0.8],[0,0,1]]    
         
            Np_types = P['IU'].value_counts()                # frequency of each type of patient in queue
            Np_types.sort_index(inplace=True)
            p_types = P.IU.unique()
            p_types.sort()
            p_types = p_types.tolist()
            b_types = Beds.IU.unique()
            b_types.sort()
            
            ### Useful_Beds- Preprocessing
            ############################################
            useful_Beds = pd.DataFrame(index=range(0), columns=['Bed.ID','Status','Availability time','IU','D'])
            UBeds_idx = []
            for i in range(Npq):
                for j in range(len(Beds)):
                    if Beds.loc[j,'IU'] >= P.loc[i,'IU']:
                        if Rooms.loc[Beds.loc[j,'Room'],'Gender']== P.loc[i,'Gender'] or Rooms.loc[Beds.loc[j,'Room'],'Gender']== 0 :
                            if Beds.loc[j,'Availability time'] <= TH+Time:
                                UBeds_idx.append(j)
                        else: 
                            if max(Beds[(Beds['Room'] == Beds.loc[j,'Room'])]['Availability time']) <= TH+Time:  ## this condition considers clean beds
                                UBeds_idx.append(j)
        
            UBeds_idx = (list(set(UBeds_idx)))                    
            useful_Beds = Beds.loc[UBeds_idx] 
            Ub_types = useful_Beds.IU.unique()
            NUb_types = useful_Beds['IU'].value_counts()
            NUb_types.sort_index(inplace=True)
            for k in p_types:
                if k in Ub_types : Nb_useful = NUb_types[k]
                else:  Nb_useful = 0 
                if Np_types[k] > Nb_useful:  
                    B1 = Beds.loc[(Beds['IU'] == k) & (~Beds.index.isin(UBeds_idx))].nsmallest((Np_types[k]-Nb_useful+5), 'Availability time')    
                    B11 = Beds.loc[(Beds['IU'] >= k) & (~Beds.index.isin(UBeds_idx))].nsmallest((Np_types[k]-Nb_useful+2), 'Availability time')            
                    useful_Beds = pd.concat([useful_Beds, B1, B11],ignore_index=False)
            ## Add Extra dirty beds                             
            if len(useful_Beds.loc[(useful_Beds['Status']=='dirty')]) + len(useful_Beds.loc[(useful_Beds['Status']=='occupied')]) < 2*len(EVS):
                if len(Beds.loc[(Beds['Status']=='dirty')]) + len(Beds.loc[(Beds['Status']=='occupied')]) > 2*len(EVS):        
                    B2= Beds.loc[(Beds['Status']!='clean')].nsmallest((2*len(EVS)), 'Availability time')
                    useful_Beds = pd.concat([useful_Beds, B2],ignore_index=False)        
            ## Add roommate of all beds    
            R_numbers = useful_Beds.Room.unique()
            ExtraBeds = Beds.loc[(Beds['Room'].isin(R_numbers)) & (~Beds.index.isin(useful_Beds.index))]
            useful_Beds = pd.concat([useful_Beds, ExtraBeds],ignore_index=False) 
            if Numberoftry4OPT > 1: useful_Beds = pd.concat([useful_Beds, Beds],ignore_index=False)
            useful_Beds = useful_Beds.drop_duplicates(subset=['Bed.ID', 'Availability time'], keep="first")
            useful_Beds = useful_Beds.reset_index()
            useful_Beds = useful_Beds.drop(['index'], axis=1)
            
            useful_Beds = useful_Beds.sort_values('Availability time',ascending=True)
            useful_Beds = useful_Beds.reset_index()
            useful_Beds = useful_Beds.drop(['index'], axis=1)    
        #    ExtraBeds_idx = (useful_Beds[useful_Beds['Bed.ID'].isin(ExtraBeds['Bed.ID'])].index).tolist()   
            ################ useful_Beds = Beds.copy(deep=True) ################       
            
            B_statindex = [[],[]]                                     # Clean / dirty and busy
            for b in range (len(useful_Beds)):
                if  useful_Beds.loc[b, 'Status'] == 'clean' : B_statindex[0].append(b)   
                if  useful_Beds.loc[b, 'Status'] == 'dirty' or useful_Beds.loc[b, 'Status'] == 'occupied' : B_statindex[1].append(b)
            B_typeindex = [[] for _ in range(IU_types)]
                    
            for i in range(len(B_typeindex)):
                for b in range(len(useful_Beds)):
                    if  useful_Beds.loc[b, 'IU'] == i+1 : B_typeindex[i].append(b) 
                   
            Nj = [len(useful_Beds.index),len(EVS.index), len(Tran.index)]       # number of members for each resource
                
            t_j1b = useful_Beds.loc[:,'Availability time']                      # availability time of dirty bed for initial service
            t_j1t = Tran.loc[: ,'Availability time']                            # availability time of each Trans staff for initial service
            t_j1e = EVS.loc[: ,'Availability time']                             # availability time of each EVS staff for initial service
            t_p = P.loc[:,'Admission time']                                     # admission time for patients
            Rip = useful_Beds.loc[:,'D']                                        #distance between beds and ED
        
            ### Computing Duplications#############
            if NE == len(EVS):
                if Npq > NE*4 : ## 4 is maximum duplication that we considered
                    NdbTH = len(useful_Beds.loc[((useful_Beds['Status']=='dirty') | (useful_Beds['Status'] =='occupied'))]) 
                    D1 = [max(1, int(math.ceil(min(Npq,NdbTH)/float(Nj[1]))))] * NE         
                else:  
                    NdbTH = len(useful_Beds.loc[((useful_Beds['Status']=='dirty') | (useful_Beds['Status'] =='occupied'))])
                    D1 = [max(1, min(4, int(math.ceil(NdbTH/float(Nj[1])))))] * NE         
        
            if NE < len(EVS):
                D1 = []
                if Npq > NE* 4:    ## check how many beds can be cleaned with current shift EVS staffs 
                    for i in range(NE): 
                        NdbTH = len(useful_Beds.loc[(useful_Beds['Availability time'] < EVS.loc[i,'END_Shift']) & ((useful_Beds['Status']=='dirty') | (useful_Beds['Status'] =='occupied'))]) 
                        D1.append(max(1, min( int(math.ceil(Npq/float(Nj[1]))), int(NdbTH - sum(D1)), int((EVS.loc[i,'END_Shift']- max(Time,EVS.loc[i,'Availability time']))/ S_be)))) 
                    for i in range (len(EVS)-NE) : 
                        NdbTH = len(useful_Beds.loc[(useful_Beds['Availability time'] < EVS.loc[NE+i,'END_Shift']) & ((useful_Beds['Status']=='dirty') | (useful_Beds['Status'] =='occupied'))]) 
                        if sum(D1) < min(Npq, NdbTH):   D1.append(max(1, min( int(math.ceil(Npq/float(len(EVS)-NE))), int(math.ceil(NdbTH/float(len(EVS)-NE))), int(NdbTH - sum(D1)), int((EVS.loc[NE+i,'END_Shift']- max(Time,EVS.loc[NE+i,'Availability time']))/ S_be))))  ## for next shift EVS consider two duplication
                        else: D1.append(max(1,min(4, int(NdbTH - sum(D1)))))
                else:  
                    for i in range(NE): 
                        NdbTH = len(useful_Beds.loc[(useful_Beds['Availability time'] < EVS.loc[i,'END_Shift']) & ((useful_Beds['Status']=='dirty') | (useful_Beds['Status'] =='occupied'))]) 
                        D1.append(max(1, min( 4, int(NdbTH - sum(D1)), int((EVS.loc[i,'END_Shift']- max(Time,EVS.loc[i,'Availability time']))/ S_be)))) 
                    NdbTH = len(useful_Beds.loc[((useful_Beds['Status']=='dirty') | (useful_Beds['Status'] =='occupied'))]) 
                    for i in range (len(EVS)-NE) : 
                        D1.append(max(1, min(4, max(0,int(math.ceil((NdbTH- sum(D1))/float(NE)))), int((EVS.loc[NE+i,'END_Shift']- max(Time,EVS.loc[NE+i,'Availability time']))/ S_be))))  ## for next shift EVS consider two duplication
            if Numberoftry4OPT > 2: D1 = [Npq]*len(EVS)             
            D = [1, D1, max(1,int(math.ceil(Npq/float(Nj[2]))))]    
        
            ################################################
            
            B2B = pd.DataFrame(index=range(len(useful_Beds)), columns=range(len(useful_Beds)))             # distance between two beds
            for i in range(len(useful_Beds)):
                for j in range(len(useful_Beds)):
                    if i==j :                                                B2B.loc[i,j] = 0
                    if useful_Beds.loc[i,'IU'] == useful_Beds.loc[j,'IU'] :  B2B.loc[i,j] = 0
                    else:                                                    B2B.loc[i,j] = b2b.iloc[int(useful_Beds.loc[i,'IU']-1),int(useful_Beds.loc[j,'IU']-1)] 
            
            gM = P.loc[:,'Gender'].copy(deep=True)                                                            # Gender parameter, gM[p] is 1 when gender of patient p is male
            gF = P.loc[:,'Gender'].copy(deep=True)                                                            # Gender parameter, gF[p] is 1 when gender of patient p is female
            for i in range(Npq):      
                if P.loc[i,'Gender']=='Male' :   gM[i]= 1  ;   gF[i]=0
                if P.loc[i,'Gender']=='Female': gM[i]= 0  ;   gF[i]=1
         
            GM0 = useful_Beds.loc[:,'P_Gender'].copy(deep=True)                                               # Gender parameter, GM0[j] is 1 when gender of patient p on bed j is male
            GF0 = useful_Beds.loc[:,'P_Gender'].copy(deep=True)                                               # Gender parameter, GF0[j] is 1 when gender of patient p on bed j is female
            for j in range(len(useful_Beds)):
                if useful_Beds.loc[j,'P_Gender']=='Male' :   GM0[j]= 1  ;   GF0[j]=0
                if useful_Beds.loc[j,'P_Gender']=='Female': GM0[j]= 0  ;   GF0[j]=1
                
            Ir = [[] for _ in range(len(Rooms))]                                                              # set of beds for each room
            for i in range(len(useful_Beds)):                                                                 # for considering all types
                for r in range (len(Rooms)):
                    if  useful_Beds.loc[i, 'Room'] == r : Ir[r].append(i)           
                    
            SBalancing = sum(Balancing)        
            for ne in range(len(Balancing)):
                if SBalancing>0 : Balancing[ne] = Balancing[ne]/SBalancing
                else: Balancing[ne] = 1
        
            ### Patients parameters- Preprocessing
            ############################################
            FixedinTF  = pd.DataFrame(index=range(0), columns=['#.P','#.Bed','Tp_prev']) 
            for f in range(len(P_FixedinTF)):
                if sum(P['ID'].isin([str(P_FixedinTF[f])]))==1:
                    FixedinTF.loc[f,'#.P'] = int(np.where(P['ID'] == P_FixedinTF[f])[0])
                    FixedinTF.loc[f,'#.Bed'] = int(np.where(useful_Beds['Bed.ID'] == PB_prev.loc[int(np.where(PB_prev['Patient.ID'] == P_FixedinTF[f])[0]),'Bed.ID'])[0])                          
                    FixedinTF.loc[f,'Tp_prev'] = Tp_prev.loc[int(np.where(Tp_prev['Patient.ID'] == P_FixedinTF[f])[0]),'Tp_prev']              
            FixedinTF = FixedinTF.reset_index()
            FixedinTF = FixedinTF.drop(['index'], axis=1)   
                           
            X_prev = pd.DataFrame(numpy.zeros(shape=(len(useful_Beds),Npq))) 
            for i in range(len(PB_prev)): 
                if sum(P['ID'].isin([PB_prev.loc[i,'Patient.ID']]))==1:
                    if sum(useful_Beds['Bed.ID'].isin([PB_prev.loc[i,'Bed.ID']])) == 1:
                        X_prev.loc[int(np.where(useful_Beds['Bed.ID'] == PB_prev.loc[i,'Bed.ID'])[0]), int(np.where(P['ID'] == PB_prev.loc[i,'Patient.ID'])[0])] = 1
        
            h_ip = [[] for _ in range(Npq)]  # Tier level of IUs for each patient 
            H = pd.DataFrame(index=range(Npq), columns=['H'])
            up = []
            for i in range(Npq):
                for j in range(IU_types):
                    if  (j+1) < P.loc[i,'IU'] : h_ip[i].append(0) 
                    elif (j+1) == P.loc[i,'IU'] : h_ip[i].append(1) 
                    else: h_ip[i].append(h[int(P.loc[i,'IU']-1)][j])
                up.append(max(h_ip[i])-min([x for x in h_ip[i] if (x > 0)]))
                H.loc[i,'H']= float(max(h_ip[i]))
                
            P_Bindex =  [[] for _ in range(Npq)]   ## appropriate bed indices (for all units) for each patients
            for i in range(Npq):
                for j in range(len(useful_Beds)):
                    if P.loc[i,'IU'] == 1 and useful_Beds.loc[j, 'IU']==1:  P_Bindex[i].append(j) 
                    if P.loc[i,'IU'] == 2 and useful_Beds.loc[j, 'IU']==2:  P_Bindex[i].append(j) 
                    if P.loc[i,'IU'] == 3 and useful_Beds.loc[j, 'IU']==3:  P_Bindex[i].append(j) 
                        
            Ct_p = pd.Series(P['Ct_p'])
            Hp = pd.Series(H['H'])
            
            ### Symmetry Patients
            SC = [[]for _ in range(IU_types)]  # Patients in the same condition 
            for u in range(IU_types):
                SC[u].append(list(map(int, np.where(np.logical_and(P['IU']==u+1 , P['Gender']=='Female'))[0])))
                SC[u].append(list(map(int, np.where(np.logical_and(P['IU']==u+1 , P['Gender']=='Male'))[0])))
        
            ### Symmetry Beds
            SB = [[[]for _ in range(2)]for _ in range(IU_types)]  # Patients in the same condition , 2= Genders
            for bu in range(IU_types):
                for ibu in range(len(useful_Beds)):
                    if useful_Beds.loc[ibu,'IU'] == bu+1: 
                        if Rooms.loc[useful_Beds.loc[ibu,'Room'],'Gender']!='Male' : SB[bu][0].append(ibu)
                        else : SB[bu][1].append(ibu)
        
            PA = P.loc[:,'Priority']     ### save priorities before changing
            for v in range(IU_types):    ### for symmetry cases of patients and patients that arriving in close times, add more priority to another to decrease complexity time
                for e in range(2):  
                    if (len(SC[v][e])>1):
                        for u in range(len(SC[v][e])-1):
                            if (P.loc[SC[v][e][u+1],'Admission time'] - P.loc[SC[v][e][u],'Admission time']) < 5 : P.loc[SC[v][e][u],'Priority'] = P.loc[SC[v][e][u],'Priority']+ 0.01
                                            
            A = P.loc[:,'Priority']                          # priority coefficient of each patient in objective function
        
            ##############################################################################       
             ### Define Variables
            m = Model("IP-1")
            
            X = m.addVars(range(len(t_j1b)), range(D[0]), range(Npq), vtype= GRB.BINARY, name= "Bed")                      # Bed j for its d service assigned to patient p (X)
            Y = m.addVars(range(len(t_j1t)), range(D[2]), range(Npq), vtype= GRB.BINARY,  name= "Trans")                    # Transporter j for its d service assigned to patient p (Y)
            Z = m.addVars(range(len(t_j1e)), range(max(D[1])), range(len(t_j1b)), range(D[0]), vtype= GRB.BINARY, name= "EVS")       # EVS staff j for its d service assigned to dirty bed k for its d' cleaning (Z)
        
            deltaRM= m.addVars(range(len(Rooms)), vtype= GRB.BINARY, name= "deltaRM")            # indicator variable for gender constraint
            deltaRF= m.addVars(range(len(Rooms)), vtype= GRB.BINARY, name= "deltaRF")
        
            theta = m.addVars(range(Npq), range(len(t_j1b)), vtype= GRB.BINARY, name= "theta")   # indicator variable for solution stability constraint
        
            thetaS = m.addVars(range(1), vtype= "C" , name="thetaS")     ## sum of theta binary variables
        
            deltap = m.addVars(range(Npq), vtype= "C", name= "deltaP")   # Penalty term for Bed-Tier constraint
        
            Stp = m.addVars(range(Npq), vtype= "C", name= "Stp")  
            tb = m.addVars(range(len(t_j1b)), range(D[0]), vtype= "C", name= "tb")               # time that bed j becomes available for its d cleaning
            tc = m.addVars(range(len(t_j1b)), range(D[0]), vtype= "C", name= "tc")               # time that clean bed j becomes available for its d service
            te = m.addVars(range(len(t_j1e)), range(max(D[1])), vtype= "C", name= "te")               # time that evs staff j becomes available for its d service
            tt = m.addVars(range(len(t_j1t)), range(D[2]), vtype= "C", name= "tt")               # time that transporter j becomes available for its d service
        
            t_bp = m.addVars(range(Npq), vtype= "C", name="t_bp")                                # time that bed is clean and ready for patient p (t_bp)
            T_p =  m.addVars(range(Npq), vtype= "C" , name="T_p")                                # time that patient p is served   (T)
            TpMax = m.addVars(range(1), vtype= "C" , name="TpMax")                               # Maximum value of Tp
            WpMax = m.addVars(range(1), vtype= "C" , name="WpMax")                              # maximum value of patients waiting times
            
            ## Set objective: 
            beta1 = [1, 0.01, 10, 0.01]
            beta2 = [1,1,1]
            beta = [beta1, beta2]
              
            if itry == 0 : 
        
                expr = beta1[0]*quicksum(A[p] * T_p[p] for p in range(Npq)) + quicksum(A[p] *(-1* t_p[p]) for p in range(Npq)) + quicksum(beta1[1]*(tc[i, 0]-Time) for i in B_statindex[1]) + quicksum(0.001*tt[j, d] for j in range(Nj[2]) for d in range(D[2])) + quicksum(beta1[2]*deltap[p] for p in range(Npq)) + beta1[3]*WpMax[0]   #+ 0.1*WpMax[0] 
            else: 
        
                expr = beta2[0]*quicksum((tc[i, 0]-Time) for i in B_statindex[1]) + beta2[1]*quicksum(deltap[p] for p in range(Npq)) + beta2[2]*WpMax[0] 
                
            m.setObjective(expr, GRB.MINIMIZE);
            
            m.modelSense = GRB.MINIMIZE
            m.update()
            
            ##############################################################################
            # Define Constraints 
            if itry >0 : 
                m.addConstr((quicksum(A[i]*T_p[i] for i in range(len(A))) <= (sum(A*t_p) + (objval1_prev + epsilon))), "Const00") #max(epsilon,0.15*objval1_prev)
                m.addConstrs((T_p[p] <= (max(Assignments_0['T_P']))    for p in range(Npq)), "Const000")        
            
            m.addConstrs((((T_p[p]-t_p[p])-Stp[p]) <= WpMax[0]    for p in range(Npq)), "Const0") 
                        
            m.addConstrs((tb[j,0] == t_j1b[j]   for j in range(len(t_j1b))),"Const1")                              # Availability times of beds for their first service
            
            m.addConstrs((te[j,0] == t_j1e[j]   for j in range(len(t_j1e))),"Const2")                              # Availability times of EVS staffs for their first service
           
            m.addConstrs((tt[j,0] == t_j1t[j]   for j in range(len(t_j1t))),"Const3")                              # Availability times of Transport staffs for their first service
        
            for j in range(len(t_j1e)):
                for d in range(D[1][j]):
                    for k in range(len(t_j1b)):
                        for l in range(D[0]):
                            if d > 0:     m.addConstr((Z[j,(d-1),k,l] == 1) >> (te[j,d] >= tc[k,l]),"Const5")
        
            for j in range(len(t_j1t)):
                for d in range(D[2]):
                    for p in range(Npq):
                        if d > 0:        m.addConstr((Y[j,(d-1),p] == 1) >> (tt[j,d] >= T_p[p] + Stp[p]),"Const6")   #Transporter becomes available for the next service after completing its (d-1)th service + coming back 
               
            for p in range(Npq):
                for j in range(len(t_j1b)):
                        for d in range(D[0]):
                            m.addConstr((X[j,d,p] == 1) >> (tc[j,d] <= t_bp[p]),"Const7")                            # time that a bed is clean and ready for patient p
        
            for j in range(len(t_j1t)):
                for d in range(D[2]):
                    for p in range(Npq):
                        m.addConstr((Y[j,d,p] == 1) >> (T_p[p] >= tt[j,d] + Stp[p]),"Const8")
        
            m.addConstrs((quicksum(Rip[j]*X[j,d,p] for j in P_Bindex[p] for d in range(D[0])) - Stp[p] == 0  for p in range(Npq)), "Const9") 
            m.addConstrs((T_p[p] >= t_p[p] + Stp[p]    for p in range(Npq)), "Const10")       
               
            m.addConstrs((T_p[p] >= t_bp[p]    for p in range(Npq)), "Const11")       
               
            for j in B_statindex[1]:
                for d in range(D[0]):
                    for i in range(len(t_j1e)):
                        for l in range(D[1][i]):
                            m.addConstr((Z[i,l,j,d] == 1) >> (tc[j,d] >= tb[j,d] + S_be),"Const12")                    #for dirty bed
            
            m.addConstrs((tc[j,d] >= tb[j,d]    for j in B_statindex[0] for d in [0]), "Const12.1")
        
            for j in B_statindex[0]:
                for d in range(D[0]):
                    for i in range(len(t_j1e)):
                        for l in range(D[1][i]):
                            if d>0: m.addConstr((Z[i,l,j,d] == 1) >> (tc[j,d] >= tb[j,d] + S_be),"Const12.2")          # for clean bed in second duplicaton
            
            if Numberoftry4OPT < 1:
                for j in B_statindex[1]:
                    for d in range(D[0]):
                        for i in range(len(t_j1e)-1):
                            for l in range(D[1][i]):
                                m.addConstr((Z[i,l,j,d] == 1) >> (tc[j,d] <= EVS.loc[i,'END_Shift']),"Const12.3")          # Doesn't assign EVS staff after their shift
        
            for i in range(len(t_j1e)):
                for l in range(D[1][i]):
                    for j1 in range(len(t_j1b)):
                        for d1 in range(D[0]):
                            if l>0: m.addConstr((Z[i,l,j1,d1] == 1) >> (tc[j1,d1] >= te[i,l] + S_be + (quicksum(B2B.loc[j1,j2]*Z[i,l-1,j2,d2] for j2 in range(len(t_j1b)) for d2 in range(D[0])))),"Const13.1")   
        
            for i in range(len(t_j1e)):
                for l in range(D[1][i]):
                    for j1 in range(len(t_j1b)):
                        for d1 in range(D[0]):
                            if l== 0:
                                m.addConstr((Z[i,l,j1,d1] == 1) >> (tc[j1,d1] >= te[i,l] + S_be + IU_D[str(EVS.loc[i,'Location'])][int(useful_Beds.loc[j1,'IU']-1)]),"Const13.2") 
                                                            
            m.addConstrs((quicksum(X[j,d,p] for j in P_Bindex[p] for d in range(D[0])) == 1 for p in range(Npq)), "Const14")    # Assign a bed to each patient
        
            m.addConstrs((quicksum(X[j,d,p] for j in range(len(t_j1b)) for d in range(D[0])) <= 1 for p in range(Npq)), "Const14-1")    # Assign a bed to each patient   
                    
            m.addConstrs((quicksum(X[j,d,p]  for p in range(Npq)) <= 1 for d in range(D[0]) for j in range(len(t_j1b))), "Const15")    # Do not assign more than 1 patient to a dirty bed 
        
            for p in range(Npq):
                m.addConstr((quicksum(Y[j, d, p] for j in range(len(t_j1t)) for d in range(D[2])) >= quicksum(X[j,d, p] for j in P_Bindex[p] for d in range(D[0]))), "Const16.1")
                m.addConstr((quicksum(Y[j, d, p] for j in range(len(t_j1t)) for d in range(D[2])) <= 1), "Const16.2")

            m.addConstrs((quicksum(Y[j,d,p] for p in range(Npq)) <= 1 for j in range(len(t_j1t)) for d in range(D[2])), "Const17")    # Do not assign more than 1 EVS staff to a dirty bed 
            
            if len(B_statindex[1]) >= sum(D[1]) :
                m.addConstrs((quicksum(Z[i,l,j,d] for j in range(len(t_j1b)) for d in range(D[0])) <= 1 for i in range(len(t_j1e))
                                                                                                        for l in range(D[1][i])), "Const18.1")    # Assign a dirty bed to each EVS member at most once
         
                m.addConstr((quicksum(Z[i,l,j,d] for j in B_statindex[1] for d in range(D[0]) for i in range(len(t_j1e)) for l in range(D[1][i])) <= sum(D[1])), "Const19")    # Assign a dirty bed to each EVS member at most once
                   
                m.addConstrs((quicksum(Z[i,l,j,d] for i in range(len(t_j1e)) for l in range(D[1][i])) <= 1  for j in B_statindex[1]
                                                                                                         for d in range(D[0])), "Const21")    # Do not assign more than 1 EVS staff to a dirty bed
            else:
                m.addConstrs((quicksum(Z[i,l,j,d] for j in range(len(t_j1b)) for d in range(D[0])) <= 1 for i in range(len(t_j1e))
                                                                                                        for l in range(D[1][i])), "Const23")    # Assign a dirty bed to each EVS member at most once
            
                m.addConstrs((quicksum(Z[i,l,j,d] for i in range(len(t_j1e)) for l in range(D[1][i])) <= 1  for j in B_statindex[1]
                                                                                                            for d in range(D[0])), "Const24")    # Do not assign more than 1 EVS staff to a dirty bed
        
            m.addConstrs((quicksum(X[j,d,p] for p in range(Npq)) <= quicksum(Z[i,l,j,d] for i in range(len(t_j1e)) for l in range(D[1][i])) for j in B_statindex[1] for d in range(D[0])), "Const20")    # Assign a transporter to each patient
              
            m.addConstrs((quicksum(Z[i,l,j,d] for j in B_statindex[0] for d in [0]) == 0 for i in range(len(t_j1e)) for l in range(D[1][i])), "Const25")    # don't assign EVS staff to clean bed
                   
            m.addConstrs((quicksum(Y[j,d,p] for p in range(Npq)) - quicksum(Y[j,(d-1),pp] for pp in range(Npq)) <= 0   for j in range(len(t_j1t))
                                                                                                                       for d in range(D[2])
                                                                                                                       if d>0), "Const27")    # Do not assign a transporter for dth service if it is not assigned (d-1)th times
            
            m.addConstrs((quicksum(Z[i,l,j,d] for j in range(len(t_j1b)) for d in range(D[0])) - quicksum(Z[i,(l-1),j,d] for j in range(len(t_j1b)) for d in range(D[0])) <= 0 for i in range(len(t_j1e))
                                                                                                                                                                               for l in range(D[1][i])
                                                                                                                                                                               if l>0), "Const28")    # assign EVS staff to dirty beds based on duplication respectively 
         
            m.addConstrs((quicksum(Z[i,l,j,d] for i in range(len(t_j1e)) for l in range(D[1][i])) - quicksum(Z[i,l,j,(d-1)] for i in range(len(t_j1e)) for l in range(D[1][i])) <= 0 for j in range(len(t_j1b))
                                                                                                                                                                               for d in range(D[0])
                                                                                                                                                                               if d>0), "Const29")
        
            m.addConstrs((deltaRM[r] + deltaRF[r] == 1 for r in range(len(Rooms))), "Const30")
        
            m.addConstrs(((deltaRM[r] == 1) >> (quicksum(gM[p]*X[i,d,p] for i in Ir[r]  for p in range(Npq) for d in range(D[0])) >= quicksum(X[i,d,p] for i in Ir[r]  for p in range(Npq) for d in range(D[0]))) for r in range(len(Rooms))),"Const31")
        
            m.addConstrs(((deltaRF[r] == 1) >> (quicksum(gF[p]*X[i,d,p] for i in Ir[r]  for p in range(Npq) for d in range(D[0])) >= quicksum(X[i,d,p] for i in Ir[r]  for p in range(Npq) for d in range(D[0]))) for r in range(len(Rooms))),"Const32")
        
            m.addConstrs(((gF[p]*X[j,d,p] == 1) >> (GM0[i]*tb[i,d] <= T_p[p])    for r in range(len(Rooms)) 
                                                                                for p in range(Npq) 
                                                                                for d in range(D[0])
                                                                                if gF[p] > 0
                                                                                for i in Ir[r] for j in Ir[r]  if j!=i),"Const33")
            
            m.addConstrs(((gM[p]*X[j,d,p] == 1) >> (GF0[i]*tb[i,d] <= T_p[p])    for r in range(len(Rooms)) 
                                                                                for p in range(Npq) 
                                                                                for d in range(D[0])
                                                                                if gM[p] > 0
                                                                                for i in Ir[r] for j in Ir[r]  if j!=i),"Const34")
             
            m.addConstrs((quicksum(h_ip[p][u]*X[i,d,p] for u in range(IU_types) for d in range(D[0]) for i in B_typeindex[u]) >= Hp[p]- deltap[p]*(1- Ct_p[p]) for p in range(Npq)) , "Const39")    
            
            for i in range(IU_types):    ### for symmetry cases of patients
                for j in range(2):  
                    m.addConstrs((T_p[p] >= T_p[g]    for p in SC[i][j] for g in SC[i][j] if g <p), "Const41.1")
                    m.addConstrs((t_bp[p] >= t_bp[g]  for p in SC[i][j] for g in SC[i][j] if g <p), "Const41.2")
        
            m.addConstrs((tc[j,d] >= Time    for j in B_statindex[1] for d in [0]), "Const42.1")     ### for limiting cleaning time of dirty beds that are not assigned and their terms are in the objective fundtion
            m.addConstrs((tc[j,d] <= TpMax[0]    for j in B_statindex[1] for d in [0]), "Const42.2")    
            m.addConstrs((T_p[p] <= TpMax[0]    for p in range(Npq)), "Const42.3")       
        
            m.addConstr((quicksum(Z[i,l,j,d] for i in range(len(t_j1e)) for l in range(D[1][i]) for j in range(len(t_j1b)) for d in range(D[0])) >= Nbc), "Const43")   ## Increasing number of cleaning of extra dirty beds
        
            #############################################################################     
            ## Initial solutions
        
            if len(Assignments_0) >0 :
                for p in range(Npq):
                    for j in range(len(t_j1b)):
                        for d in range(D[0]):
                            X[j,d,p].start = 0
                            if d == 0 : 
                                tb[j,d].start = t_j1b[j]
                                if useful_Beds.loc[j,'Status']== 'clean': tc[j,d].start = tb[j,d].start
                
                for p in range(Npq):
                    for j in range(len(t_j1t)):
                        for d in range(D[2]):
                            if d == 0 : 
                                tt[j,d].start = t_j1t[j]                    
                
                for i in range(len(t_j1e)):
                    for d1 in range(D[1][i]):
                        for j in range(len(t_j1b)):
                            for d2 in range(D[0]): 
                               Z[i,d1,j,d2].start = GRB.UNDEFINED 
            
                for p in range(Npq):
                    T_p[p].start = Assignments_0.loc[p,'T_P']
                    t_bp[p].start = Assignments_0.loc[p,'T_P']
                    
                    jb = int(np.where(useful_Beds['Bed.ID']==Assignments_0.loc[p,'Bed.ID'])[0])
                    db = int(Assignments_0.loc[0:p,'Bed.ID'].value_counts().to_dict()[Assignments_0.loc[p,'Bed.ID']]-1)    
            
                    if db == 0 : ## because we don't consider bed duplication, if we consider bed_d > 0 => remove this line
                        X[jb,db,p].start = 1
                        tc[jb,db].start = Assignments_0.loc[p,'tc']
                    if db == 0 : 
                        tb[jb,db].start = t_j1b[jb]
                        if useful_Beds.loc[jb,'Status']== 'clean': tc[jb,db].start = tb[jb,db].start
                        
                    jt = int(np.where(Tran['Tran.ID']==Assignments_0.loc[p,'Transport.ID'])[0])
                    dt = int(Assignments_0.loc[0:p,'Transport.ID'].value_counts().to_dict()[Assignments_0.loc[p,'Transport.ID']]-1)       
                    if dt < D[2]: 
                        Y[jt,dt,p].start = 1
                        tt[jt,dt].start = Assignments_0.loc[p,'tt']
                    
            #############################################################################                     
            # Model Optimization 
            # help(GRB.Attr) 
            # m.Params.SubMIPNodes = 1000
            # m.Params.Heuristics = 0
            # m.tune() 
            # m.objBound                         # Best bound of the result
            # m.Params.MIPFocus=3
            # m.Params.ImproveStartGap = 1       ## setting this parameter to 0.1 will cause the MIP solver to switch strategies once the relative optimality gap is smaller than 0.1
            # m.Params.Cuts = 2                  ## 2 = aggressive cuts
            m.Params.MIPFocus = 3
        
            m.Params.method = 1    ## dual => Options are: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex
            m.params.MIPGap = 0.001  # Gap < 0.1%      # or m.params.MIPGapAbs  = 0.01
            m.params.TimeLimit = RunTimelimit  # or less than 30 min
            m.reset()
            m.update()
            m.optimize()
            runtime = round(m.runtime, 4)
            m.printAttr('X')
            
            m.write("Rout.lp")
            # m.printQuality()      
            ##################### for infeasible solutions
            status = m.status
            if status == 3 :
                print ('Optimization was stopped with status %d' %status)
                m.computeIIS()
                for c in m.getConstrs():
                    if c.IISConstr:
                        print('%s'%c.constrName)              
                        
            break
        except:
            Numberoftry4OPT = Numberoftry4OPT +1

    #################################################################
           
    Assignments = pd.DataFrame(index=range(Npq), columns=['Bed','d_Bed','EVS','d_EVS','Transport','d_Transport','Delay_B','Delay_Tr'])
    Bed_EVS_assign = pd.DataFrame(index=range(0), columns=['Bed','Bed.ID','d_Bed','EVS','EVS.ID','d_EVS','tc'])

    for p in range(Npq): 
        for j in range (len(t_j1b)):
            for d in range (D[0]):
                if round(X[j,d,p].x) == 1 : 
                    Assignments.loc[p,'Bed'] = j
                    Assignments.loc[p,'d_Bed'] = d

    ii=0
    for j2 in range(len(useful_Beds)):
        for d2 in range (D[0]):
            for j in range (len(EVS)):
                for d in range (D[1][j]):
                    if round(Z[j,d,j2,d2].x) == 1 :
                        Bed_EVS_assign.loc[ii,'EVS'] = j
                        Bed_EVS_assign.loc[ii,'EVS.ID'] = EVS.loc[j,'EVS.ID']
                        Bed_EVS_assign.loc[ii,'d_EVS'] = d
                        Bed_EVS_assign.loc[ii,'Bed'] = j2
                        Bed_EVS_assign.loc[ii,'Bed.ID'] = useful_Beds.loc[j2,'Bed.ID']
                        Bed_EVS_assign.loc[ii,'d_Bed'] = d2
                        Bed_EVS_assign.loc[ii,'tc'] = tc[j2,d2].x
                        ii = ii+1

    for j2 in range(len(useful_Beds)):
        for d2 in range (D[0]):
            for j in range (len(EVS)):
                for d in range (D[1][j]):
                    if round(Z[j,d,j2,d2].x) == 1 :
                        if len(Assignments.loc[(Assignments['Bed'] == j2) & (Assignments['d_Bed'] == d2)]) != 0 :
                            index = Assignments.loc[(Assignments['Bed'] == j2) & (Assignments['d_Bed'] == d2)].index[0]
                            Assignments.loc[index,'EVS'] = j                                                                      
                            Assignments.loc[index,'d_EVS'] = d

    for p in range(Npq): 
        for j in range (len(t_j1t)):
            for d in range (D[2]):
                if round(Y[j,d,p].x) == 1 : 
                    Assignments.loc[p,'Transport'] = j
                    Assignments.loc[p,'d_Transport'] = d
                    
    ## Define delay variables to find bottlenecks
    for p in range(Npq):
        if tc[Assignments.loc[p,'Bed'],Assignments.loc[p,'d_Bed']].x <=  t_p[p] : 
            Assignments.loc[p,'Delay_B'] = 0
        else: 
            if tc[Assignments.loc[p,'Bed'],Assignments.loc[p,'d_Bed']].x <=  tt[Assignments.loc[p,'Transport'],Assignments.loc[p,'d_Transport']].x :
                Assignments.loc[p,'Delay_B'] = max(0,tc[Assignments.loc[p,'Bed'],Assignments.loc[p,'d_Bed']].x - t_p[p])    
            else:  Assignments.loc[p,'Delay_B'] = max(0,tc[Assignments.loc[p,'Bed'],Assignments.loc[p,'d_Bed']].x - t_p[p]- P.loc[p,'IU_D'])
                
        if tt[Assignments.loc[p,'Transport'],Assignments.loc[p,'d_Transport']].x <=  t_p[p] :
            Assignments.loc[p,'Delay_Tr'] = 0
        else: 
            Assignments.loc[p,'Delay_Tr'] = max(0,tt[Assignments.loc[p,'Transport'],Assignments.loc[p,'d_Transport']].x - t_p[p])
                                    
    Assignments_ID = pd.DataFrame(index=range(Npq), columns=['#.Patient','Patient.ID','Gender','Admission time','P.IU','#.Bed','Bed.ID','Room','B.IU','d_Bed','EVS.ID','d_EVS','Transport.ID','d_Transport','S_tp','tb','tc','te','tt','t_bp','T_P','Waiting.T','LOS','Discharge time','Tp_prev','Subtract(Tp)','Delay_B','Delay_Tr','Priority'])
          
    for i in range(Npq):
        Assignments_ID.loc[i,'Patient.ID'] = P.loc[i,'ID']
        Assignments_ID.loc[i,'Priority'] = P.loc[i,'Priority']
        Assignments_ID.loc[i,'P.IU'] = P.loc[i,'IU']
        Assignments_ID.loc[i,'Gender'] = P.loc[i,'Gender']
        Assignments_ID.loc[i,'#.Patient'] = int(np.where(IU_original['ID'] == str(Assignments_ID.loc[i,'Patient.ID']))[0])
        Assignments_ID.loc[i,'Admission time'] = P.loc[i,'Admission time']
        Assignments_ID.loc[i,'Delay_B'] = Assignments.loc[i,'Delay_B']
        Assignments_ID.loc[i,'Delay_Tr'] = Assignments.loc[i,'Delay_Tr']
        Assignments_ID.loc[i,'Bed.ID'] = useful_Beds.loc[Assignments.loc[i,'Bed'],'Bed.ID']
        Assignments_ID.loc[i,'B.IU'] = useful_Beds.loc[Assignments.loc[i,'Bed'],'IU']
        Assignments_ID.loc[i,'S_tp'] = useful_Beds.loc[Assignments.loc[i,'Bed'],'D']
        Assignments_ID.loc[i,'Room'] = useful_Beds.loc[Assignments.loc[i,'Bed'],'Room']
        Assignments_ID.loc[i,'#.Bed'] = int(np.where(Beds['Bed.ID'] == Assignments_ID.loc[i,'Bed.ID'])[0]) 
        Assignments_ID.loc[i,'d_Bed'] = Assignments.loc[i,'d_Bed']
        if numpy.isnan (Assignments.loc[i,'EVS']) :
            Assignments_ID.loc[i,'EVS.ID'] = '-'
            Assignments_ID.loc[i,'d_EVS'] = '-'
            Assignments_ID.loc[i,'te'] = '-'
        else:  
            Assignments_ID.loc[i,'EVS.ID'] = EVS.loc[Assignments.loc[i,'EVS'],'EVS.ID']
            Assignments_ID.loc[i,'d_EVS'] = Assignments.loc[i,'d_EVS']
        Assignments_ID.loc[i,'Transport.ID'] = Tran.loc[Assignments.loc[i,'Transport'],'Tran.ID']
        Assignments_ID.loc[i,'d_Transport'] = Assignments.loc[i,'d_Transport']

        Assignments_ID.loc[i,'t_bp'] = t_bp[i].x
        Assignments_ID.loc[i,'T_P'] = T_p[i].x
        Assignments_ID.loc[i,'LOS'] = P.loc[i,'Discharge time'] - Assignments_ID.loc[i,'T_P']
        Assignments_ID.loc[i,'Discharge time'] = P.loc[i,'Discharge time']
        Assignments_ID.loc[i,'Waiting.T'] = Assignments_ID.loc[i,'T_P'] - Assignments_ID.loc[i,'Admission time'] - Assignments_ID.loc[i,'S_tp']
        if  Assignments_ID.loc[i,'Waiting.T'] < 0.5 : Assignments_ID.loc[i,'Waiting.T']= 0
        if any(Tp_prev['Patient.ID'] == Assignments_ID.loc[i,'Patient.ID']):
            Assignments_ID.loc[i,'Tp_prev'] = Tp_prev.loc[int(np.where(Tp_prev['Patient.ID'] == Assignments_ID.loc[i,'Patient.ID'])[0]),'Tp_prev']
            Assignments_ID.loc[i,'Subtract(Tp)'] = Assignments_ID.loc[i,'T_P'] - Assignments_ID.loc[i,'Tp_prev']
        
    for j in Assignments.Bed.unique():
        for i in range (Npq):
            if Assignments.loc[i,'Bed'] == j :
                d = Assignments.loc[i,'d_Bed']
                Assignments_ID.loc[i,'tb'] = tb[j,d].x            
                Assignments_ID.loc[i,'tc'] = tc[j,d].x

    for j in Assignments.EVS.unique():
        for i in range (Npq):
            if Assignments.loc[i,'EVS'] == j :
                d = Assignments.loc[i,'d_EVS']
                Assignments_ID.loc[i,'te'] = te[j,d].x                           

    for j in Assignments.Transport.unique():
        for i in range (Npq):
            if Assignments.loc[i,'Transport'] == j :
                d = Assignments.loc[i,'d_Transport']
                Assignments_ID.loc[i,'tt'] = tt[j,d].x                      
                           
###################################################################
######### EVS Journey report for each iteration
    EVS_Report = pd.DataFrame(index=range(0), columns=['EVS.ID','Cleaning.T','Earlieast.Availability.T','Location','#.Bed','Bed.ID','B.IU','Bed.EA.Time','Waiting.T for Bed','Traveling.S.T','Traveling.E.T','Cleaning.S.T','Cleaning.E.T'])  # E.Availability.T=Earlieast.Availability.T
    
    if (len(Bed_EVS_assign)> 0):
        for j in range(len(EVS)):
            assignINFO = pd.DataFrame(index=range(D[1][j]), columns=['EVS.ID','Cleaning.T','Earlieast.Availability.T','Location','#.Bed','Bed.ID','B.IU','Bed.EA.Time','Waiting.T for Bed','Traveling.S.T','Traveling.E.T','Cleaning.S.T','Cleaning.E.T'])  # E.Availability.T=Earlieast.Availability.T    
            for i in range(D[1][j]):
                for j2 in range(len(useful_Beds)):
                    for i2 in range(D[0]):
                        if round(Z[j,i,j2,i2].x)==1 :
                            assignINFO.loc[i,'EVS.ID'] = EVS.loc[j,'EVS.ID']
                            if i == 0: assignINFO.loc[i,'Earlieast.Availability.T'] = EVS.loc[j,'Availability time']
                            else:      assignINFO.loc[i,'Earlieast.Availability.T'] = assignINFO.loc[i-1,'Cleaning.E.T']
                            assignINFO.loc[i,'Cleaning.T'] = S_be    # EVS.loc[j,'Cleaning_Time']
            
                            irow = int(np.where(np.logical_and(Bed_EVS_assign['d_EVS'] == i, Bed_EVS_assign['EVS'] == j))[0])
                            assignINFO.loc[i,'Cleaning.E.T'] = Bed_EVS_assign.loc[irow,'tc']
                            assignINFO.loc[i,'Cleaning.S.T'] = assignINFO.loc[i,'Cleaning.E.T'] - assignINFO.loc[i,'Cleaning.T']
                            assignINFO.loc[i,'Bed.ID'] = Bed_EVS_assign.loc[irow,'Bed.ID']
                            assignINFO.loc[i,'Bed.EA.Time'] =  Beds.loc[int(np.where(Beds['Bed.ID'] == assignINFO.loc[i,'Bed.ID'])[0]),'Availability time']
                            assignINFO.loc[i,'#.Bed'] = int(np.where(Beds['Bed.ID'] == assignINFO.loc[i,'Bed.ID'])[0])
                            assignINFO.loc[i,'B.IU'] = Beds.loc[int(np.where(Beds['Bed.ID'] == assignINFO.loc[i,'Bed.ID'])[0]),'IU']
                            assignINFO.loc[i,'Traveling.E.T'] = assignINFO.loc[i,'Cleaning.S.T']

                            # Calculating Traveling start time and update location
                            if i==0 : assignINFO.loc[i,'Location'] = EVS.loc[j,'Location'] 
                            else:     assignINFO.loc[i,'Location'] = assignINFO.loc[i-1,'B.IU']
                            if assignINFO.loc[i,'Location'] =='ED' : assignINFO.loc[i,'Traveling.S.T'] = assignINFO.loc[i,'Traveling.E.T'] - Beds.loc[int(np.where(Beds['Bed.ID'] == assignINFO.loc[i,'Bed.ID'])[0]),'D']
                            else: 
                                if assignINFO.loc[i,'Location'] == assignINFO.loc[i,'B.IU'] : assignINFO.loc[i,'Traveling.S.T'] = assignINFO.loc[i,'Traveling.E.T']
                                else: assignINFO.loc[i,'Traveling.S.T'] = assignINFO.loc[i,'Traveling.E.T'] - (b2b.iloc[int(assignINFO.loc[i,'Location']-1),int(assignINFO.loc[i,'B.IU']-1)])
                                    
                            assignINFO.loc[i,'Waiting.T for Bed'] = assignINFO.loc[i,'Traveling.S.T'] - assignINFO.loc[i,'Earlieast.Availability.T']
                            if assignINFO.loc[i,'Waiting.T for Bed'] < 0 : assignINFO.loc[i,'Waiting.T for Bed']= 0
                                
            EVS_Report = pd.concat([EVS_Report, assignINFO], axis=0)
            EVS_Report = EVS_Report[list(assignINFO.columns.values)]
            EVS_Report = EVS_Report.dropna(how='all')

#########################################################################################
######### Transport Journey report for each iteration
    Tran_Report = pd.DataFrame(index=range(0), columns=['Tran.ID','Earlieast.Availability.T','Location','#.Patient','P.ID','P.IU','Waiting.T','Resean of waiting','Transporting.S.T','Transporting.E.T'])  # E.Availability.T=Earlieast.Availability.T
    
    for j in range(len(Tran)):
        assignINFO2 = pd.DataFrame(index=range(D[2]), columns=['Tran.ID','Earlieast.Availability.T','#.Patient','P.ID','P.IU','Waiting.T','Resean of waiting','Transporting.S.T','Transporting.E.T'])  # E.Availability.T=Earlieast.Availability.T                                               
        for i in range(D[2]):
            for p in range(Npq):
                if round(Y[j,i,p].x) == 1: 
                    assignINFO2.loc[i,'Tran.ID'] = Tran.loc[j,'Tran.ID']
                    if i == 0:  assignINFO2.loc[i,'Earlieast.Availability.T'] = Tran.loc[j,'Availability time']
                    else:       assignINFO2.loc[i,'Earlieast.Availability.T'] = assignINFO2.loc[i-1,'Transporting.E.T']                                                       
                    irow = int(np.where(np.logical_and(Assignments_ID['d_Transport'] == i, Assignments_ID['Transport.ID'] == Tran.loc[j,'Tran.ID']))[0])
                    assignINFO2.loc[i,'Transporting.E.T'] = Assignments_ID.loc[irow,'T_P']
                    assignINFO2.loc[i,'P.ID'] = P.loc[int(np.where(P['ID'] == Assignments_ID.loc[irow,'Patient.ID'])[0]),'ID']
                    assignINFO2.loc[i,'#.Patient'] = int(np.where(IU_original['ID'] == Assignments_ID.loc[irow,'Patient.ID'])[0])
                    assignINFO2.loc[i,'Transporting.S.T'] = assignINFO2.loc[i,'Transporting.E.T'] - P.loc[int(np.where(P['ID'] == Assignments_ID.loc[irow,'Patient.ID'])[0]),'IU_D']                
                    assignINFO2.loc[i,'P.IU'] = P.loc[int(np.where(P['ID'] == Assignments_ID.loc[irow,'Patient.ID'])[0]),'IU']
                    assignINFO2.loc[i,'Waiting.T'] = assignINFO2.loc[i,'Transporting.S.T'] - assignINFO2.loc[i,'Earlieast.Availability.T'] - P.loc[int(np.where(P['ID'] == Assignments_ID.loc[irow,'Patient.ID'])[0]),'IU_D']
#                    assignINFO2.loc[i,'Waiting.T'] = assignINFO2.loc[i,'Transporting.S.T'] - Assignments_ID.loc[int(np.where(np.logical_and(Assignments_ID['d_Transport'] == i-1, Assignments_ID['Transport.ID']== assignINFO2.loc[i,'Tran.ID']))[0]),'T_P']
                    if assignINFO2.loc[i,'Waiting.T'] < 0 : assignINFO2.loc[i,'Waiting.T']= 0
                    if round(assignINFO2.loc[i,'Waiting.T']) > 0:
                        if t_p[irow] > tc[Assignments.loc[irow,'Bed'],Assignments.loc[irow,'d_Bed']].x : 
                            assignINFO2.loc[i,'Resean of waiting'] = 'P'         # reason of Waiting time is patient 
                        else: assignINFO2.loc[i,'Resean of waiting'] = 'C_B'     # reason of Waiting time is clean bed

        Tran_Report = pd.concat([Tran_Report, assignINFO2], axis=0)
        Tran_Report = Tran_Report[list(assignINFO2.columns.values)]
        Tran_Report = Tran_Report.dropna(how='all') 

###################################################################
######### Bed Journey report for each iteration
    Bed_Report = pd.DataFrame(index=range(len(useful_Beds)), columns=['#.Bed','Bed.ID','Room','P_Gender','Status','Earlieast.Availability.T', 'IU','D'])  # E.Availability.T=Earlieast.Availability.T
    
    for j in range(len(useful_Beds)):
        Bed_Report.loc[j,['Bed.ID','Room','P_Gender','Status','IU','D']] = useful_Beds.loc[j,['Bed.ID','Room','P_Gender','Status','IU','D']]
        Bed_Report.loc[j,'Earlieast.Availability.T'] = useful_Beds.loc[j,'Availability time']
        Bed_Report.loc[j,'#.Bed'] = int(np.where(Beds['Bed.ID'] == Bed_Report.loc[j,'Bed.ID'])[0])    

    for i in range(D[0]):
        assignINFO3 = pd.DataFrame(index=range(len(useful_Beds)), columns=['Waiting.T for EVS','EVS','d_EVS','Cleaning.E.T','Waiting.T for patient','Patient.ID','#.Patient','Arrival_T','Discharge_T'])  # Cleaning start time and end time
        for j in range(len(useful_Beds)):
            if any(Bed_EVS_assign['Bed.ID'] == useful_Beds.loc[j,'Bed.ID']):
                if  len(np.where(np.logical_and(Bed_EVS_assign['d_Bed'] == i, Bed_EVS_assign['Bed.ID'] == Bed_Report.loc[j,'Bed.ID']))[0]) > 0 :
                    jidx = int(np.where(np.logical_and(Bed_EVS_assign['d_Bed'] == i, Bed_EVS_assign['Bed.ID'] == Bed_Report.loc[j,'Bed.ID']))[0])            
                    assignINFO3.loc[j,'EVS'] = Bed_EVS_assign.loc[jidx,'EVS.ID']
                    assignINFO3.loc[j,'d_EVS'] = Bed_EVS_assign.loc[jidx,'d_EVS']
                    assignINFO3.loc[j,'Cleaning.E.T'] = Bed_EVS_assign.loc[jidx,'tc']
                    if i == 0:   assignINFO3.loc[j,'Waiting.T for EVS'] = assignINFO3.loc[j,'Cleaning.E.T'] - S_be - Bed_Report.loc[j,'Earlieast.Availability.T'] 
                    if assignINFO3.loc[j,'Waiting.T for EVS'] < 0 : assignINFO3.loc[j,'Waiting.T for EVS']= 0
            for p in range(Npq):
                if round(X[j,i,p].x) == 1: 
                    kidx = int(np.where(np.logical_and(Assignments_ID['d_Bed'] == i, Assignments_ID['Bed.ID']== Bed_Report.loc[j,'Bed.ID']))[0])
                    assignINFO3.loc[j,'Patient.ID'] = Assignments_ID.loc[kidx,'Patient.ID']
                    assignINFO3.loc[j,'#.Patient'] = int(np.where(IU_original['ID'] == assignINFO3.loc[j,'Patient.ID'])[0])
                    assignINFO3.loc[j,'Arrival_T'] = Assignments_ID.loc[kidx,'T_P']
                    assignINFO3.loc[j,'Discharge_T'] = P.loc[p,'Discharge time']
                    if Bed_Report.loc[j,'Earlieast.Availability.T'] == 0: assignINFO3.loc[j,'Cleaning.E.T'] = Bed_Report.loc[j,'Earlieast.Availability.T']
                    assignINFO3.loc[j,'Waiting.T for patient'] = assignINFO3.loc[j,'Arrival_T'] - assignINFO3.loc[j,'Cleaning.E.T']- Bed_Report.loc[j,'D']
                    if assignINFO3.loc[j,'Waiting.T for patient'] < 0 : assignINFO3.loc[j,'Waiting.T for patient']= 0

        Bed_Report = pd.concat([Bed_Report, assignINFO3], axis=1)
        
#########################################################################################        
    P.loc[:,'Priority'] = PA
    objval1 = sum((A[p]*T_p[p].x - A[p]*t_p[p]) for p in range(Npq))    ## Sum of all waiting times          #- Assignments_ID['S_tp']  
    objval2 = sum((tc[j,0].x - Time) for j in B_statindex[1])                                                ## weighting sum of Cleaning times
    objval3 = sum(Tran_Report['Earlieast.Availability.T'])                               ## weighting sum of transportes availability time
    objval4 = sum(deltap[i].x for i in range(Npq))
    objval5 = WpMax[0].x
    objval = [objval1, objval2, objval3, objval4, objval5]
    opt_Gap = round(m.MIPGap, 4)
    N_Vars = m.NumVars
    N_Constrs = m.NumConstrs
    return (Assignments_ID, EVS_Report, Tran_Report, Bed_Report, Bed_EVS_assign, D, useful_Beds, objval, opt_Gap, H, h_ip, N_Vars, N_Constrs, runtime, beta, Numberoftry4OPT)
