#GUI for endoscope
#copyright Guan-Chyuan Wang 2021.6

from tkinter import Tk, Label, Entry,DoubleVar, Button, messagebox, LabelFrame, Radiobutton
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, binarize
import numpy as np
from pickle import load
model = load_model('endoscope.model')
main = Tk()
main.title('Spinal endoscopic route suggestion')
main.geometry('800x600')
sex = DoubleVar()
age = DoubleVar()
BMI = DoubleVar()
duration = DoubleVar()
OPsite = DoubleVar()
ODI = DoubleVar()
VAS = DoubleVar()
dsclass = DoubleVar()
preop = DoubleVar()
herniation_type = DoubleVar()
zone = DoubleVar()
Pfrirr = DoubleVar()
MRI_type = DoubleVar()
canalstenosis = DoubleVar()

def predict_op():
    X = np.array([sex.get(),age.get(),BMI.get(),duration.get(),ODI.get(),VAS.get(),
                  OPsite.get(),dsclass.get(),preop.get(),herniation_type.get(),zone.get(),
                  Pfrirr.get(),MRI_type.get(),canalstenosis.get()])
    X = X.reshape(1,-1).astype(str)
    oe = load(open('oe.pkl','rb'))
    test_x_enc = oe.transform(X)
    predict_y_enc = model.predict(test_x_enc)
    y_hat = binarize(predict_y_enc,threshold=0.5).astype(int)
    le = load(open('le.pkl','rb'))
    predictions = le.inverse_transform(np.ravel(y_hat))
    if predictions == '1.0':
        OPmethod = 'Transforaminal'
    else:
        OPmethod = 'Interlaminar'
    return OPmethod , predict_y_enc
def showMsg():
    OPmethod,y_hat= predict_op()
    messagebox.showinfo('Result: ',f'Recommend choice: {OPmethod}')

def radioBut(master,variables,*val):
    for keys,items in enumerate(val):
        Radiobutton(master,text=items,variable=variables,value=keys+1).grid(row=0,column=keys,padx=5,pady=5)
    
    
#==sex==
SEX = LabelFrame(main, text='Sex',labelanchor='n')
SEX.grid(row=0,column=0,padx=5,pady=5,sticky='nesw')
Radiobutton(SEX,text='male',variable=sex,value=1).grid(row=0,column=0,padx=5,pady=5)
Radiobutton(SEX,text='female',variable=sex,value=2).grid(row=1,column=0,padx=5,pady=5)

#==age==
AGE = LabelFrame(main, text='Age',labelanchor='n')
AGE.grid(row=0,column=1,padx=5,pady=5,sticky='nesw')
Radiobutton(AGE,text='<=65',variable=age,value=1).grid(row=0,column=0,padx=5,pady=5)
Radiobutton(AGE,text='>65',variable=age,value=2).grid(row=1,column=0,padx=5,pady=5)
#==BMI==
BM = LabelFrame(main, text='BMI',labelanchor='n')
BM.grid(row=0,column=2,padx=5,pady=5,sticky='nesw')
Radiobutton(BM,text='<=30',variable=BMI,value=1).grid(row=0,column=0,padx=5,pady=5)
Radiobutton(BM,text='>30',variable=BMI,value=2).grid(row=1,column=0,padx=5,pady=5)

#==symptoms duration==
SD = LabelFrame(main, text='Symptom duration',labelanchor='n')
SD.grid(row=0,column=3,padx=5,pady=5,sticky='nesw')
Radiobutton(SD,text='<=3 months',variable=duration,value=1).grid(row=0,column=0,padx=5,pady=5)
Radiobutton(SD,text='>3 months',variable=duration,value=2).grid(row=1,column=0,padx=5,pady=5)

#==lumbar leve==
LV = LabelFrame(main, text='Lumbar level',labelanchor='n')
LV.grid(row=1,column=0,padx=5,pady=5,sticky='nesw',columnspan=4)
LumbarLevels = {0:'L2/L3',1:'L3/L4',2:'L4/L5',3:'L5/S1',4:'other 2 levles',
                5:'L4/L5+L5/S1',6:'L1/L2',7:'T12/L1'}
for val, lv in LumbarLevels.items():
    Radiobutton(LV,text=lv,variable=OPsite,value=val+1).grid(row=0,column=val,padx=5,pady=5)
#==ODI==
OD = LabelFrame(main, text='ODI',labelanchor='n')
OD.grid(row=2,column=0,padx=5,pady=5,sticky='nesw',columnspan=2)
radioBut(OD,ODI,'<=20','20-40','40-60','60-80','>80')

#==VAS==
VASFrame = LabelFrame(main, text='VAS',labelanchor='n')
VASFrame.grid(row=2,column=2,padx=5,pady=5,sticky='nesw')
Radiobutton(VASFrame,text='<=4',variable=VAS,value=1).grid(row=0,column=0,padx=5,pady=5)
Radiobutton(VASFrame,text='>4',variable=VAS,value=2).grid(row=1,column=0,padx=5,pady=5)

#==Previous surgical history==
PSH = LabelFrame(main, text='Previous surgery',labelanchor='n')
PSH.grid(row=2,column=3,padx=5,pady=5,sticky='nesw')
Radiobutton(PSH,text='No',variable=preop,value=0).grid(row=0,column=0,padx=5,pady=5)
Radiobutton(PSH,text='Yes',variable=preop,value=1).grid(row=1,column=0,padx=5,pady=5)

#==lumbar disease==
LDS = LabelFrame(main, text='Lumbar disease',labelanchor='n')
LDS.grid(row=3,column=0,padx=5,pady=5,sticky='nesw',columnspan=2)
radioBut(LDS,dsclass,'HIVD','Canal stenosis','Foraminal stenosis')

#==Herniation type==
HT = LabelFrame(main, text='Herniation type',labelanchor='n')
HT.grid(row=3,column=2,padx=5,pady=5,sticky='nesw',columnspan=2)
radioBut(HT,herniation_type,'Prolapse','Extrusion','Sequestration')

#==Migrating level==
ML = LabelFrame(main, text='Migrating level',labelanchor='n')
ML.grid(row=4,column=0,padx=5,pady=5,sticky='nesw',columnspan=2)
radioBut(ML,zone,'Zone I','Zone II','Zone III','Zone IV','None')

#==Pfirrmann grade==
PFG = LabelFrame(main, text='Pfirrmann grade',labelanchor='n')
PFG.grid(row=4,column=2,padx=5,pady=5,sticky='nesw',columnspan=2)
radioBut(PFG,Pfrirr,'I','II','III','IV','V')

#==axial plane==
AXIAL = LabelFrame(main, text='Lesion localization',labelanchor='n')
AXIAL.grid(row=5,column=0,padx=5,pady=5,sticky='nesw',columnspan=2)
radioBut(AXIAL,MRI_type,'Central','Subarticular','Foraminal','Extraforaminal')

#==high canal compromised==
HC = LabelFrame(main, text='>50% canal compromised',labelanchor='n')
HC.grid(row=5,column=2,padx=5,pady=5,sticky='nesw')
Radiobutton(HC,text='No',variable=canalstenosis,value=0).grid(row=0,column=0,padx=5,pady=5)
Radiobutton(HC,text='Yes',variable=canalstenosis,value=1).grid(row=1,column=0,padx=5,pady=5)

#==check==
btn = Button(main,text='Predict',command=showMsg)
btn.grid(row=5,column=3)
main.mainloop()