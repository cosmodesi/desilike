import numpy as np

path_mike='/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/sugiyama_basis/SecondGenMocks/'
#path_mike='/pscratch/sd/m/mikewang/shared/desi/measurements-v20250124/SecondGenMocks/'
#path_mike='/Users/waco/Mike_folder/sd/m/mikewang/shared/desi/measurements-v20250124/SecondGenMocks/'



pole_selection=[True,True,True,True,True]
range1=[0.00,100]
ranges=[range1,range1,range1,range1,range1]

def ExtractData(tracer,z_string,ranges=ranges,pole_selection=pole_selection,subtract_shot=False):

    k_all,pl0_,pl2_,pl4_,B000_,B202_ = ExtractDataAbacusSummit(tracer,z_string,subtract_shot=subtract_shot)
    data_=np.concatenate((pl0_.T,pl2_.T,pl4_.T,B000_.T,B202_.T)).T
    data_vector_all = np.mean(data_,axis = 0)

    k_cov,pl0_cov,pl2_cov,pl4_cov,B000_cov,B202_cov = ExtractDataEZmock(tracer,z_string)
    k_cov_all, mean_ezmocks_all, cov_array_all = covariance(k_cov,pl0_cov,pl2_cov,pl4_cov,B000_cov,B202_cov)

    mask=pole_k_selection(k_cov_all,pole_selection,ranges)

    k_cov=k_cov_all[mask]
    data_vector=data_vector_all[mask]
    cov_array=cov_array_all[np.ix_(mask, mask)]


    return ({'mask':mask, 'data_vector':data_vector, 'cov_array':cov_array, 'k_eff_Abaus_all':k_eff_all, 'k_eff_EZmocks_all':k_cov_all}) 

### Run example, oonly pkl0 and pkl2 between k=0.02 and k=0.2
# pole_selection=[True,True,False,False,False]
# range1=[0.02,0.20]
# ranges=[range1,range1,range1,range1,range1]

# extractData = ExtractData(tracer,z_string,ranges=ranges,pole_selection=pole_selection,subtract_shot=True)

# data_vector=extractData['data_vector']
# cov_array=extractData['cov_array']



def ExtractDataEZmock(tracer,z_string,path_mike=path_mike):

    mocks='EZmock'
    #mocks='AbacusSummit' or 'EZmock'
    
    # path_mike='/Users/waco/Mike_folder/sd/m/mikewang/shared/desi/measurements-v20250124/SecondGenMocks/'
    
    if (mocks=='EZmock'): folder_box='/CubicBox_6Gpc/'
    if (mocks=='AbacusSummit'): folder_box='/CubicBox/'
    
    path_bisp=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/bispec/'
    path_pk=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/powspec/'

    nummocks=2000

    pkl0=np.zeros((nummocks,80))
    pkl2=np.zeros((nummocks,80))
    pkl4=np.zeros((nummocks,80))
    B000=np.zeros((nummocks,80))
    B202=np.zeros((nummocks,80))

    
    for i in range(2000):
        j=i+1
        if (0 <= j < 10): seed='000'+str(j)
        if (10 <= j < 100): seed='00'+str(j)
        if (100 <= j < 1000): seed='0'+str(j)
        if (1000 <= j < 10000): seed=str(j)
        fileB000 = np.loadtxt(path_bisp+'bk000_diag_'+tracer+'_'+z_string+'_seed'+seed, unpack = True)
        fileB202 = np.loadtxt(path_bisp+'bk202_diag_'+tracer+'_'+z_string+'_seed'+seed, unpack = True)
        filepkl0 = np.loadtxt(path_pk  +'pk0_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True)
        filepkl2 = np.loadtxt(path_pk  +'pk2_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True)
        filepkl4 = np.loadtxt(path_pk  +'pk4_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True) 
        
        if (j==1): k_eff_all = fileB000[1]
        B000[i]=fileB000[6]
        B202[i]=fileB202[6]
        pkl0[i]=filepkl0[3]
        pkl2[i]=filepkl2[3]
        pkl4[i]=filepkl4[3]


    return k_eff_all, pkl0, pkl2, pkl4, B000, B202





def ExtractDataAbacusSummit(tracer,z_string,path_mike=path_mike,subtract_shot=False):

    #mocks='EZmock'
    mocks='AbacusSummit'
    
    # path_mike='/Users/waco/Mike_folder/sd/m/mikewang/shared/desi/measurements-v20250124/SecondGenMocks/'
    
    if (mocks=='EZmock'): folder_box='/CubicBox_6Gpc/'
    if (mocks=='AbacusSummit'): folder_box='/CubicBox/'
    
    path_bisp=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/bispec/'
    path_pk=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/powspec/'


    nummocks=25

    pkl0=np.zeros((nummocks,80))
    pkl2=np.zeros((nummocks,80))
    pkl4=np.zeros((nummocks,80))
    B000=np.zeros((nummocks,80))
    B202=np.zeros((nummocks,80))

    noiseB000=0; noiseB202=0
    noisel0=0; noisel2=0; noisel4=0
    
    for j in range(25):
        if (0 <= j < 10): seed='00'+str(j)
        if (10 <= j < 100): seed='0'+str(j)
        fileB000 = np.loadtxt(path_bisp+'bk000_diag_'+tracer+'_'+z_string+'_ph'+seed, unpack = True)
        fileB202 = np.loadtxt(path_bisp+'bk202_diag_'+tracer+'_'+z_string+'_ph'+seed, unpack = True)
        filepkl0 = np.loadtxt(path_pk  +'pk0_'       +tracer+'_'+z_string+'_ph'+seed, unpack = True)
        filepkl2 = np.loadtxt(path_pk  +'pk2_'       +tracer+'_'+z_string+'_ph'+seed, unpack = True)
        filepkl4 = np.loadtxt(path_pk  +'pk4_'       +tracer+'_'+z_string+'_ph'+seed, unpack = True)
        if (j==1): k_eff_all = fileB000[1]
        
        if (subtract_shot): 
            noisel0=filepkl0[5]
            noisel2=filepkl2[5]
            noisel4=filepkl4[5]
            noiseB000 = fileB000[8]
            noiseB202 = fileB202[8]

        
        B000[j]=fileB000[6]-noiseB000
        B202[j]=fileB202[6]-noiseB202
        pkl0[j]=filepkl0[3]-noisel0
        pkl2[j]=filepkl2[3]-noisel2
        pkl4[j]=filepkl4[3]-noisel4    
        
        

    return k_eff_all, pkl0, pkl2, pkl4, B000, B202


def ExtractDataAbacusSummit_additionalcosmologies(cxxx='c000'):

    path= '/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/alternative_cosmologies/LRG_z0.725/'+cxxx+'/pk0_CubicBox_'+cxxx+'_phmean'
    k_sk=np.loadtxt(path)
    k=k_sk[:,0]
    pkl0=k_sk[:,1]
    
    path= '/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/alternative_cosmologies/LRG_z0.725/'+cxxx+'/pk2_CubicBox_'+cxxx+'_phmean'
    k_sk=np.loadtxt(path)
    k=k_sk[:,0]
    pkl2=k_sk[:,1]
    
    path= '/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/alternative_cosmologies/LRG_z0.725/'+cxxx+'/bk000_diag_CubicBox_'+cxxx+'_phmean'
    k_sk=np.loadtxt(path)
    k=k_sk[:,0]
    B000=k_sk[:,1]
    
    path= '/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/alternative_cosmologies/LRG_z0.725/'+cxxx+'/bk202_diag_CubicBox_'+cxxx+'_phmean'
    k_sk=np.loadtxt(path)
    k_eff_all=k_sk[:,0]
    B202=k_sk[:,1]

    pkl4=pkl2

    return k_eff_all, pkl0, pkl2, pkl4, B000, B202


def covariance(k, pkl0, pkl2, pkl4, B000, B202, Nscaling = 1):

    data=np.concatenate((pkl0.T,pkl2.T,pkl4.T,B000.T,B202.T)).T
    k_cov = np.concatenate((k,k,k,k,k))

    cov=np.cov(data.T)
    cov = cov * 27  # Factor 27=6^3/2^2 accounts for the volumes: EZmocks are (6Gpch)^3 and Abacus (2Gpch)^3. 
    cov = cov/Nscaling
    
    mean_data=np.mean(data,axis = 0)

    return k_cov, mean_data, cov




def pole_k_selection(ks_vector,pole_selection,ranges):
    
    fit_selection = np.repeat(pole_selection, len(ks_vector)/5)
    
    krange = ks_vector[0:int(len(ks_vector)/5)];
    
    pkl0_range = np.logical_and(ranges[0][0] < krange, krange < ranges[0][1])
    pkl2_range = np.logical_and(ranges[1][0] < krange, krange < ranges[1][1])
    pkl4_range = np.logical_and(ranges[2][0] < krange, krange < ranges[2][1])
    B000_range = np.logical_and(ranges[3][0] < krange, krange < ranges[3][1])
    B202_range = np.logical_and(ranges[4][0] < krange, krange < ranges[4][1])
    range_selection=np.concatenate((pkl0_range,pkl2_range,pkl4_range,B000_range,B202_range))

    selection=np.logical_and(fit_selection, range_selection)
    
    return selection





def kr_pervector(kdata,pole_selection,ranges):
    
    fit_selection = np.repeat(pole_selection, len(ks_vector)/5)
    
    krange = ks_vector[0:int(len(ks_vector)/5)];
    
    pkl0_range = np.logical_and(ranges[0][0] < krange, krange < ranges[0][1])
    pkl2_range = np.logical_and(ranges[1][0] < krange, krange < ranges[1][1])
    pkl4_range = np.logical_and(ranges[2][0] < krange, krange < ranges[2][1])
    B000_range = np.logical_and(ranges[3][0] < krange, krange < ranges[3][1])
    B202_range = np.logical_and(ranges[4][0] < krange, krange < ranges[4][1])
    range_selection=np.concatenate((pkl0_range,pkl2_range,pkl4_range,B000_range,B202_range))

    selection=np.logical_and(fit_selection, range_selection)
    
    return selection













##################################################################
##############       OLD         #################################
##################################################################

def ExtractDataEZmock_old(tracer,z_string,path_mike=path_mike):

    mocks='EZmock'
    #mocks='AbacusSummit' or 'EZmock'
    
    # path_mike='/Users/waco/Mike_folder/sd/m/mikewang/shared/desi/measurements-v20250124/SecondGenMocks/'
    
    if (mocks=='EZmock'): folder_box='/CubicBox_6Gpc/'
    if (mocks=='AbacusSummit'): folder_box='/CubicBox/'
    
    path_bisp=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/bispec/'
    path_pk=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/powspec/'

    B000=[];B202=[]
    pkl0=[];pkl2=[];pkl4=[]
    
    for i in range(2000):
        j=i+1
        if (0 <= j < 10): seed='000'+str(j)
        if (10 <= j < 100): seed='00'+str(j)
        if (100 <= j < 1000): seed='0'+str(j)
        if (1000 <= j < 10000): seed=str(j)
        fileB000 = np.loadtxt(path_bisp+'bk000_diag_'+tracer+'_'+z_string+'_seed'+seed, unpack = True)
        fileB202 = np.loadtxt(path_bisp+'bk202_diag_'+tracer+'_'+z_string+'_seed'+seed, unpack = True)
        filepkl0 = np.loadtxt(path_pk  +'pk0_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True)
        filepkl2 = np.loadtxt(path_pk  +'pk2_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True)
        filepkl4 = np.loadtxt(path_pk  +'pk4_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True) 
        
        if (j==1): k_eff_all = fileB000[1]
        B000.append(fileB000[6])
        B202.append(fileB202[6])
        pkl0.append(filepkl0[3])
        pkl2.append(filepkl2[3])
        pkl4.append(filepkl4[3])

    k_eff_all=np.array(k_eff_all)
    B000=np.array(B000)
    B202=np.array(B202)
    pkl0=np.array(pkl0)
    pkl2=np.array(pkl2)
    pkl4=np.array(pkl4)

    return k_eff_all, pkl0, pkl2, pkl4, B000, B202



def ExtractDataAbacusSummit_old(tracer,z_string,path_mike=path_mike,subtract_shot=False):

    #mocks='EZmock'
    mocks='AbacusSummit'
    
    # path_mike='/Users/waco/Mike_folder/sd/m/mikewang/shared/desi/measurements-v20250124/SecondGenMocks/'
    
    if (mocks=='EZmock'): folder_box='/CubicBox_6Gpc/'
    if (mocks=='AbacusSummit'): folder_box='/CubicBox/'
    
    path_bisp=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/bispec/'
    path_pk=path_mike+mocks+folder_box+tracer+'/'+z_string+'/diag/powspec/'

    B000=[];B202=[]
    pkl0=[];pkl2=[];pkl4=[]

    noiseB000=0; noiseB202=0
    noisel0=0; noisel2=0; noisel4=0
    
    for j in range(25):
        if (0 <= j < 10): seed='00'+str(j)
        if (10 <= j < 100): seed='0'+str(j)
        fileB000 = np.loadtxt(path_bisp+'bk000_diag_'+tracer+'_'+z_string+'_ph'+seed, unpack = True)
        fileB202 = np.loadtxt(path_bisp+'bk202_diag_'+tracer+'_'+z_string+'_ph'+seed, unpack = True)
        filepkl0 = np.loadtxt(path_pk  +'pk0_'       +tracer+'_'+z_string+'_ph'+seed, unpack = True)
        filepkl2 = np.loadtxt(path_pk  +'pk2_'       +tracer+'_'+z_string+'_ph'+seed, unpack = True)
        filepkl4 = np.loadtxt(path_pk  +'pk4_'       +tracer+'_'+z_string+'_ph'+seed, unpack = True)
        if (j==1): k_eff_all = fileB000[1]
        
        if (subtract_shot): 
            noisel0=filepkl0[5]
            noisel2=filepkl2[5]
            noisel4=filepkl4[5]
            noiseB000 = fileB000[8]
            noiseB202 = fileB202[8]
        
        
        B000.append(fileB000[6]-noiseB000)
        B202.append(fileB202[6]-noiseB202) 
            
        pkl0.append(filepkl0[3]-noisel0)
        pkl2.append(filepkl2[3]-noisel2)
        pkl4.append(filepkl4[3]-noisel4)

    k_eff_all=np.array(k_eff_all)
    B000=np.array(B000)
    B202=np.array(B202)
    pkl0=np.array(pkl0)
    pkl2=np.array(pkl2)
    pkl4=np.array(pkl4)

    return k_eff_all, pkl0, pkl2, pkl4, B000, B202



 
def covariance_old(k, pkl0, pkl2, pkl4, B000, B202, Nscaling = 1):


    all=np.concatenate((pkl0.T,pkl2.T,pkl4.T,B000.T,B202.T)).T
    k_cov = np.concatenate((k,k,k,k,k))

    
    Nm = len(all)
    dim = len(k_cov)

    #print(Nm)
    #print(dim)
    mean_all=np.mean(all,axis = 0)
    
    #mean_all = np.zeros(dim)
    #for ii in range (0, Nm):
    #    mean_all += (all[ii][:])/Nm

    # Nm = len(B000)
    
    # mu_B000 = np.zeros(len(k))
    # for ii in range (0, Nm):
    #     mu_B000 += (B000[ii][:])/Nm


    
    cov = np.zeros((dim,dim))
    for i in range(0, dim):
        for j in range(0, dim):
            #differences
            diffi0 = all[:, i] - mean_all[i]
            diffj0 = all[:, j] - mean_all[j]
            #diagonals terms
            cov[i,j] = sum(diffi0*diffj0)
  
    cov = cov/(Nm-1.0)
    cov = cov * 27  # Factor 27=6^3/2^2 accounts for the volumes: EZmocks are (6Gpch)^3 and Abacus (2Gpch)^3. 
    cov = cov/Nscaling
        
    return k_cov, mean_all, cov

