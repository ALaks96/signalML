import numpy as np

from preprocessing import extractFeatures

def predict(wav, model):
    mfccs = extractFeatures(wav)
    if mfccs.shape[1] > 433:
        mfccs = mfccs[:,:433]
    elif mfccs.shape[1] < 433:
        mfccs = np.concatenate((mfccs,mfccs[:,(mfccs.shape[1] - (433-mfccs.shape[1])):mfccs.shape[1]]), axis=1)
    modelInput = mfccs.reshape(1, 40, 433, 1)
    results = model.predict(modelInput)
    predProbaList = [results[:,0][0],results[:,1][0],results[:,2][0],results[:,3][0]]
    problem = np.argmax(results)
    pred = False
    if problem == 0:
        detail = ['Component OK']
    # pred1 = predProbaList[1] >= 0.7
    if problem == 1:
        detail = ['Component is imbalanced']
    # pred2 = predProbaList[2] >= 0.7
    if problem == 2:
        detail = ['Component is clogged']
    # pred3 = predProbaList[3] >= 0.7
    if problem == 3:
        detail = ['Voltage change']
    
    if problem in [1,2,3]:
        pred = True
    
        
    response = {
        "Anomaly":bool(pred),
        "Details":{
                "Message":detail[0],
                "Probabilities":predProbaList  
            }   
        }
    
    # for var in ['mfccs','model','wav','modelInput','results','predProbaList','problem','pred','detail']:
    #     del globals()[var]
    # del globals()['var']
    
    return response