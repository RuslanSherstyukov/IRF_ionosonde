
"""
@author: Dr. Ruslan Sherstyukov , March 2026

Class structure:
    
    IonogramReadIRF --> DeepLearning(IonogramReadIRF) --> IonogramDatabase(DeepLearning)
    
"""

import os
import requests
import io
import h5py
import json
from datetime import datetime
import time
from dataclasses import dataclass
import numpy as np
import cv2
from matplotlib import pyplot as plt



@dataclass
class Ionogram:
    t: datetime
    f: np.ndarray
    r: np.ndarray
    I: np.ndarray
    

class IonogramReadIRF:
    
    shape = [256,256]
    path_load = "https://www2.irf.se/~ionogram/ionogram/data/rawdata/kiruna"
    fn_root_load = "KIRGRAM_"
    fn_root_save = "KIRPARA_"
    path_save = "IONOGRAM_KIR"
    
    
    def __init__(self,year,month,day,hour,minute):
        
        self.year=year
        self.month=month
        self.day=day
        self.hour=hour
        self.minute=minute
        self.io=self.get_io()
       
     
    def get_io(self):
        self.url = self.set_path(self.path_load,self.year,self.month,self.day,self.hour,offset=False,style=False)
        self.fn = self.set_fn(self.url,self.year,self.month,self.day,self.hour,self.minute,fileExt="h5",offset=False)
        self.h5 = self.get_response_h5()
        if self.h5 is not None:
            self.io = self.kir_h5_load()
            self.io = self.kir_reshape()
            return self.io
        else:
            return None
     
        
    def get_response_h5(self, retries=3, delay1=5, delay2=5):
        """Try to download the file with retries if connection fails."""
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(self.fn, timeout=10)
                
                if r.status_code == 200:
                    print("File exists.")
                    r.raise_for_status()
                    return io.BytesIO(r.content)
                else:
                    print(f"File not found. Status code: {r.status_code}. "
                          f"Date: {self.year}-{self.month}-{self.day} "
                          f"t-{self.hour}:{self.minute}")
                    return None

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt} failed: {e}")
                
                # If not the last attempt, wait before retrying
                if attempt < retries:
                    print(f"Retrying in {delay1} seconds...")
                    time.sleep(delay1)
                elif attempt >= retries:
                    print(f"Retrying in {delay2} seconds...")
                    time.sleep(delay2)
                else:
                    print("All retry attempts failed.")
                    return None
    
    
    
    def kir_h5_load(self):
        # Read timestamp from filename
        t0 = datetime.strptime(self.fn.split('/')[-1], 'ionogram-%Y-%m-%dT%H.%M.%SZ.h5')
        # Load data
        D = h5py.File(self.h5)
        # Frequencies
        f = D['I_fvec'][:,0]
        # Ranges
        r = D['I_rvec'][:]
        # Intensities, i.e. power in dB
        I = 10 * np.log10(D['I'])
        # I = D['I']
        # Frequency vector, including missing frequencies
        f2 = np.round(np.linspace(1, 16, (16-1)*10+1) * 10) / 10
        # Matrix with space for all frequencies
        I2 = np.min(I) * np.ones((len(f2), len(r)))
        # Loop through frequencies, and cope the data when the frequency has data
        for k, fk in enumerate(f2):
            try:
                I2[k,:] = I[f==fk, :]
            except ValueError:
                # print(k, fk, 'not found')
                continue

        # I2 = I2[:, (r>=0) & (r<=700)]
        # r = r[(r>=0) & (r<=700)]
        I2[:, (r>=700)]=0
        I2[:, (r<=50)]=0
        return Ionogram(t0, f2, r, I2)
    

    def kir_reshape(self):
        
        I_resized = cv2.resize(self.io.I, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
        # data = np.array(your_array)            # length 700
        r_resized = np.linspace(self.io.r[0],self.io.r[-1],self.shape[0])
        f_resized = np.linspace(self.io.f[0],self.io.f[-1],self.shape[1])
        I_resized[I_resized<0]=0
        I_resized = (I_resized - np.min(I_resized)) / (np.max(I_resized) - np.min(I_resized))
        I_resized = I_resized.T
        I_resized = np.expand_dims(np.expand_dims(I_resized, axis=-1), axis=0)
        
        return Ionogram(self.io.t,f_resized,r_resized,I_resized)
    
    
    def set_path(self,path_root,year,month,day,hour,offset,style):
        
        fn_root = self.fn_root_save
        if offset is False and style is False:
            return f"{path_root}/{year}/{month}/{day}/{hour}/"
        elif offset is True and style is False:
            return f"{path_root}/{year}_AI/{month}_AI/{day}_AI/{hour}_AI/"
        elif offset is False and style is True:
            return f"{path_root}/{fn_root}{year}/{fn_root}{year}{month}/{fn_root}{year}{month}{day}/"
        elif offset is True and style is True:
            return f"{path_root}/{fn_root}{year}_AI/{fn_root}{year}{month}_AI/{fn_root}{year}{month}{day}_AI/"

    
    def set_fn(self,fn_root,year,month,day,hour,minute,fileExt="h5",offset=False,style=False):
        
        if offset is False and style is False:
            return f"{fn_root}ionogram-{year}-{month}-{day}T{hour}.{minute}.00Z.{fileExt}"
        elif offset is True and style is False:
            return f"{fn_root}ionogram-{year}-{month}-{day}T{hour}.{minute}.00Z-AI.{fileExt}"
        elif offset is False and style is True:
            return f"{fn_root}{year}{month}{day}_{hour}{minute}00.{fileExt}"
        elif offset is True and style is True:
            return f"{fn_root}{year}{month}{day}_{hour}{minute}00_AI.{fileExt}"
   
             
class DeepLearning(IonogramReadIRF):
    
    def __init__(self,Models,year,month,day,hour,minute):
        super().__init__(year,month,day,hour,minute)
        
        self.Models = Models
        self.F2_trace = None
        self.F1_trace = None
        self.E_trace = None
        self.EF_trace = None
        self.F2_h = None
        self.F1_h = None
        self.E_h = None
        self.F2_f = None
        self.F1_f = None
        self.E_f = None
        self.F2_parameters = None
        self.F1_parameters = None
        self.E_parameters = None 
        self.flag = False
        
        self.DetermineParameters()
        
    
    def DetermineParameters(self):
        if self.io is not None:
            # self.Models = Models
            self.F2_trace = self.make_F2_trace()
            self.F2_trace = self.evaluate_trace(trace=self.F2_trace,
                                                amp_threshold=0,
                                                half_thickness_threshold=0)
            self.F1_trace = self.make_F1_trace()
            self.F1_trace = self.evaluate_trace(trace=self.F1_trace,
                                                amp_threshold=0,
                                                half_thickness_threshold=0)
            self.E_trace = self.make_E_trace()
            self.E_trace = self.evaluate_trace(trace=self.E_trace,
                                                amp_threshold=0,
                                                half_thickness_threshold=0)
            
            self.evaluate_F2()
            self.evaluate_F1()
            self.evaluate_E()
            
            self.EF_trace = self.make_EF_trace()
            
            self.F2_h = self.io.r[self.get_F2_Htrace()]
            self.F1_h = self.io.r[self.get_F1_Htrace()]
            self.E_h = self.io.r[self.get_E_Htrace()]
            
            self.F2_f = self.io.f[self.get_F2_Ftrace()]
            self.F1_f = self.io.f[self.get_F1_Ftrace()]
            self.E_f = self.io.f[self.get_E_Ftrace()]
            
            
            self.F2_parameters = self.get_F2_parameters()
            self.F1_parameters = self.get_F1_parameters()
            self.E_parameters = self.get_E_parameters()
            self.flag = True
        else:
            self.flag = False
        
    
    def evaluate_trace(self,trace,amp_threshold,half_thickness_threshold):
          
        trace=self.evaluate_f_gaps(trace)
        trace=self.evaluate_h_gaps(trace)
        trace = self.evaluate_amp(trace,amp_threshold)
        trace=self.evaluate_half_thickness(trace,half_thickness_threshold)
        return trace
    
    
    def evaluate_E(self):
        if not self.trace_exist(trace=self.F2_trace) and not self.trace_exist(trace=self.F1_trace):
            self.E_trace*=0
            
    
    def evaluate_F1(self):
        if self.trace_exist(trace=self.F2_trace) and self.trace_exist(trace=self.F1_trace):
            F2_trace_h = np.argmax(self.F2_trace[0,:,:,0] == 1, axis=0)
            F2_trace_f = np.where(F2_trace_h>0)[0]
            F1_trace_h = np.argmax(self.F1_trace[0,:,:,0] == 1, axis=0)
            F1_trace_f = np.where(F1_trace_h>0)[0]
            cross = np.intersect1d(F1_trace_f, F2_trace_f)
            if cross.size>0:
                self.F1_trace[0,:,cross[0]:,0]=0
        if self.trace_exist(trace=self.F1_trace):
            freq_diff = self.io.f[self.get_F1_parameters()[3]]-self.io.f[self.get_F1_parameters()[2]]
            if freq_diff<0.2:
                self.F1_trace*=0
            

    def evaluate_F2(self,height_threshold=180):
        if self.trace_exist(trace=self.F2_trace):
            h = np.where(self.io.r==self.io.r[self.io.r<height_threshold][-1])[0][0]
            self.F2_trace[0,:h,:,0]=0
            
        
    def make_EF_trace(self):
        return  self.F2_trace + self.F1_trace + self.E_trace
    
    
    def make_F2_trace(self,model_name="ModelTraceF2", threshold=0.3):
        trace = self.make_trace(model_name)
        trace[trace < threshold] = 0
        trace[trace >= threshold] = 1
        return trace
    
    
    def get_F2_Htrace(self):
        return self.get_trace(self.io.I,self.F2_trace,style='H')
    
    def get_F2_Ftrace(self):
        return self.get_trace(self.io.I,self.F2_trace,style='F')
            
        
    def make_F1_trace(self,model_name="ModelTraceF1", threshold=0.3):
        trace = self.make_trace(model_name)
        trace[trace < threshold] = 0
        trace[trace >= threshold] = 1
        return trace
    
    
    def get_F1_Htrace(self):
        return self.get_trace(self.io.I,self.F1_trace,style='H')
    
    
    def get_F1_Ftrace(self):
        return self.get_trace(self.io.I,self.F1_trace,style='F')
    
    
    def make_E_trace(self,model_name="ModelTraceE", threshold=0.3):
        trace = self.make_trace(model_name)
        trace[trace < threshold] = 0
        trace[trace >= threshold] = 1
        plt.imshow(trace[0,:,:,0])
        return trace
    
    
    def get_E_Htrace(self):
        return self.get_trace(self.io.I,self.E_trace,style='H')
    
    
    def get_E_Ftrace(self):
        return self.get_trace(self.io.I,self.E_trace,style='F')
    
    
    def get_F2_parameters(self):
        return self.get_parameters(trace=self.F2_trace)
    
    
    def get_F1_parameters(self):
        return self.get_parameters(trace=self.F1_trace)
    
    
    def get_E_parameters(self):
        return self.get_parameters(trace=self.E_trace)
    

    def make_trace(self,model_name):
        trace = self.pred_param(self.Models[model_name],self.io.I)
        trace[0,:,0,0]=0
        trace[0,0,:,0]=0
        return trace
    
    
    def get_trace(self,io,trace,style='H'):
        t = trace*io
        if style == 'H':
            return np.argmax(t[0,:,:,0], axis=0)
        elif style == 'F':
            return np.argmax(t[0,:,:,0], axis=1)
        
        
    def get_parameters(self,trace):
        if trace is not None:
            mask = trace[0,:,:,0] != 0
            if np.any(mask): 
                nonzero = np.where(mask.any(axis=1))[0] # Over y axis
                hmin, hmax = nonzero[0], nonzero[-1]
                nonzero = np.where(mask.any(axis=0))[0] # Over x axis
                fmin, fmax = nonzero[0], nonzero[-1]
                return (hmin,hmax,fmin,fmax)
            else:
                return (0,0,0,0)
            return None
        
        
    def get_foE(self):
        if self.trace_exist(trace=self.E_trace):
            return self.io.f[self.E_parameters[3]]
        else:
            return None
    
    
    def get_foF1(self):
        if self.trace_exist(trace=self.F1_trace):
            return self.io.f[self.F1_parameters[3]]
        else:
            return None
    
    
    def get_foF2(self):
        if self.trace_exist(trace=self.F2_trace):
            return self.io.f[self.F2_parameters[3]]
        else:
            return None
    
    
    def get_fbEs(self):
        if self.trace_exist(trace=self.F1_trace):
            return self.io.f[self.F1_parameters[2]]
        elif self.trace_exist(trace=self.F2_trace):
            return self.io.f[self.F2_parameters[2]]
        else:
            None
    
    
    def get_fmin(self):
        if self.trace_exist(trace=self.E_trace):
            return self.io.f[self.E_parameters[2]]
        elif self.trace_exist(trace=self.F1_trace):
            return self.io.f[self.F1_parameters[2]]
        elif self.trace_exist(trace=self.F2_trace):
            return self.io.f[self.F2_parameters[2]]
        else:
            None
    
    
    def get_hE(self):
        if self.trace_exist(trace=self.E_trace):
            return self.io.r[self.E_parameters[0]+ 1] # hE bias
        else:
            return None
    
    
    def get_hF1(self):
        if self.trace_exist(trace=self.F1_trace):
            return self.io.r[self.F1_parameters[0]]
        else:
            return None
    
    
    def get_hF2(self):
        if self.trace_exist(trace=self.F2_trace):
            return self.io.r[self.F2_parameters[0]]
        else:
            return None
    
    def get_hF(self):
        if self.trace_exist(trace=self.F1_trace):
            return self.io.r[self.F1_parameters[0]]
        elif self.trace_exist(trace=self.F2_trace):
            return self.io.r[self.F2_parameters[0]]
        else:
            return None
    
    
    def evaluate_amp(self,trace,threshold):
        amp_trace =trace* self.io.I
        amp_io = np.mean(self.io.I[:, (self.io.r>=50) & (self.io.r<=700)])
        amp_tr = np.mean(amp_trace[amp_trace>0])
        amp_relative = amp_tr/amp_io
        if amp_relative < threshold:
            trace *= 0   
        return trace

    
    def evaluate_f_gaps(self,trace):
        trace_h = np.argmax(trace[0,:,:,0] == 1, axis=0)
        trace_f = np.where(trace_h>0)[0]
        idx = np.where(np.diff(trace_f) > 1)[0]
        if idx.size != 0:
            traces_gaps = np.append(np.where(np.diff(trace_f)>1)[0],
                                    np.where(np.diff(trace_f)>1)[0]+1)
            traces_gaps = trace_f[traces_gaps]
            traces_gaps = np.append(traces_gaps,trace_f[0])
            traces_gaps = np.sort(np.append(traces_gaps,trace_f[-1]))
            diff = np.diff(traces_gaps)
            trace_len = diff[::2]
            # gap_len = diff[1::2]
            trace_f=np.arange(traces_gaps[np.argmax(trace_len)*2],traces_gaps[np.argmax(trace_len)*2+1])
            trace[0,:,:traces_gaps[np.argmax(trace_len)*2],0]=0
            trace[0,:,traces_gaps[np.argmax(trace_len)*2+1]:,0]=0
            trace_h = np.argmax(trace[0,:,:,0] == 1, axis=0)
            trace_f = np.where(trace_h>0)[0]
        return trace
    
    
    def evaluate_h_gaps(self,trace):
        for i in range(len(trace[0,:,:,0])):
            if len(np.where(trace[0,:,i,0]==1)[0])>0:
                h = int(np.where(trace[0,:,i,0]==1)[0][0]*1.8)
                trace[0,h:,i,0]=0 
        return trace
    
    
    def evaluate_half_thickness(self,trace,threshold):
        if self.trace_exist(trace=trace):
            trace_h = np.argmax(trace[0,:,:,0] == 1, axis=0)
            trace_h = trace_h[trace_h>0]
            # trace_f = np.where(trace_h>0)[0]
            half_thickness = self.io.r[np.max(trace_h)] - self.io.r[np.min(trace_h)]
            # print(np.max(trace_h),np.min(trace_h))
            # print(self.io.r[trace_h[-1]],self.io.r[np.min(trace_h)])
            # print(half_thickness)
            if half_thickness<threshold:
                trace*=0
        return trace

    
    def trace_exist(self,trace):
        trace_h = np.argmax(trace[0,:,:,0] == 1, axis=0)
        trace_f = np.where(trace_h>0)[0]
        if trace_f.size !=0:
            return True
        else:
            return False
        
        
    @staticmethod
    def pred_param(model,ionogram):
        return model.predict(ionogram)
    

class IonogramDatabase(DeepLearning):
    
    def __init__(self,Models,year,month,day,hour,minute):
        super().__init__(Models,year,month,day,hour,minute)
        
        self.MakeDB()
        

    def MakeDB(self):
        
        if self.flag:
            database = self.set_database_struct()
            database = self.add_database_param(database=database)
            fn = self.make_fn(offset=True, style=True, fileExt="json")
            self.save_json(fn=fn,file=database)
            self.plot_ionogram(trace=self.EF_trace)
        else:
            database = self.set_database_struct()
            database["Comment"] = "Connection failed"
            print(database["Comment"])
            fn = self.make_fn(offset=True, style=True, fileExt="json")
            self.save_json(fn=fn,file=database)
       
        return database
    
    
    def set_database_struct(self):
        data_struct = {
        "C": [None, None, None, None, None, None, None, None, None, None, None, None],
        "Comment": "Comment/huomautus: AI",
        "D": [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        "fac": [10, 1, 10, 1, 10, 1, 100, 1, 100, 10, 10, 100],
        "lim": [ [None, None], [83, 130], [1, None], [None, None], [0.7, None], [83, 130], [0.7, 5], [160, None], [1.5, 4], [1, None], [None, None], [2, 4], [" ", " "]],
        "P": ['F-MIN', 'H-ES', 'FOES', 'TYPES', 'FBES', "H'E", 'FOE ', "H'F", 'FOF1', 'FOF2', 'FXI', 'M3KF2'],
        "Q": [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        "time": [2023, 1, 1, 0, 0, 0],
        "ttl": ['fmin', "h'Es", 'foEs', 'Type Es', 'fbEs', "h'E", 'foE', "h'F", 'foF1', 'foF2', 'fxI', 'M(3000)F2', 'All'],
        "unit": ['MHz', 'km', 'MHz', '', 'MHz', 'km', 'MHz', 'km', 'MHz', 'MHz', 'MHz', ''],
        "upar": [42, 34, 30, 36, 32, 24, 20, 16, 10, 0, 51, 3],
        "User": ['AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI'],
        "V": [None, None, None, None, None, None, None, None, None, None, None, None],
        }
        
        profile_keys = [
            "E_h", "Es_h", "F1_h", "F2_h",
            "E_f", "Es_f", "F1_f", "F2_f",
            "E_amp", "Es_amp", "F1_amp", "F2_amp"
            ]
        
        for key in profile_keys:           
            data_struct[key] = [None]*255     

        return data_struct
            
    
    def get_param_list(self):
        param_list = [self.get_fmin(), None,
                      None, None,
                      self.get_fbEs(), self.get_hE(),
                      self.get_foE(), self.get_hF(),
                      self.get_foF1(),self.get_foF2(),
                      None, None]
        return param_list
    
    
    def add_database_param(self,database):
        database["V"] = self.get_param_list()
        database["E_h"] = self.E_h
        database["F1_h"] = self.F1_h
        database["F2_h"] = self.F2_h
        database["time"] = [self.year, self.month, self.day, self.hour, self.minute, 0]
        
        keys = ["E_h", "Es_h", "F1_h", "F2_h", "E_f", "Es_f", "F1_f", "F2_f","E_amp", "Es_amp", "F1_amp", "F2_amp"]
        for k in keys:
            if isinstance(database.get(k), np.ndarray):     
                database[k] = database[k].tolist()
        return database
     
            
    def save_json(self,fn,file):
        json_file = json.dumps(file)
        with open(fn, "w") as outfile:
            outfile.write(json_file)
            
        
    def plot_ionogram(self,trace):
        

        if self.io.I is not None:
            plt.title(f"{self.year}-{self.month}-{self.day} t-{self.hour}:{self.minute}")
            plt.imshow(self.io.I[0,:,:,0],origin='lower',extent=(self.io.f[0], self.io.f[-1], self.io.r[0], self.io.r[-1]),
            aspect='auto',
            cmap='viridis')
            # plt.plot(self.io.f,self.io.r[self.F2_h],'.',color='white',markersize=3)
            # plt.plot(self.io.f,self.io.r[self.F1_h],'.',color='white',markersize=3)
            # plt.plot(self.io.f,self.io.r[self.E_h],'.',color='white',markersize=3)
            
            plt.axhline(self.io.r[self.F2_parameters[0]], color='red', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F2_parameters[2]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F2_parameters[3]], color='red', linestyle='--', linewidth=1)
            
            plt.axhline(self.io.r[self.F1_parameters[0]], color='white', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F1_parameters[2]], color='white', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F1_parameters[3]], color='white', linestyle='--', linewidth=1)
            
            plt.axhline(self.io.r[self.E_parameters[0]], color='black', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.E_parameters[2]], color='black', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.E_parameters[3]], color='black', linestyle='--', linewidth=1)
            plt.colorbar()
            self.save_fig(offset=False,style=False,fileExt="png")
            # plt.show()  
            plt.close()

        if trace is not None:
            plt.title(f"{self.year}-{self.month}-{self.day} t-{self.hour}:{self.minute}")
            plt.imshow(trace[0,:,:,0],origin='lower',extent=(self.io.f[0], self.io.f[-1], self.io.r[0], self.io.r[-1]),
            aspect='auto',
            cmap='viridis')
            # plt.plot(self.io.f,self.io.r[self.F2_h],'.',color='white',markersize=3)
            # plt.plot(self.io.f,self.io.r[self.F1_h],'.',color='white',markersize=3)
            # plt.plot(self.io.f,self.io.r[self.E_h],'.',color='white',markersize=3)
            plt.axhline(self.io.r[self.F2_parameters[0]], color='red', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F2_parameters[2]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F2_parameters[3]], color='red', linestyle='--', linewidth=1)
            
            plt.axhline(self.io.r[self.F1_parameters[0]], color='white', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F1_parameters[2]], color='white', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F1_parameters[3]], color='white', linestyle='--', linewidth=1)
            
            plt.axhline(self.io.r[self.E_parameters[0]], color='black', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.E_parameters[2]], color='black', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.E_parameters[3]], color='black', linestyle='--', linewidth=1)
            plt.colorbar()
            # self.save_fig(offset=True,style=False,fileExt="png")
            # plt.show()  
            plt.close()
            
        
        if self.io.I is not None and trace is not None:
            # self.evaluate_amp(trace=self.F2_trace)
            plt.title(f"{self.year}-{self.month}-{self.day} t-{self.hour}:{self.minute}")
            plt.imshow(self.io.I[0,:,:,0] + self.io.I[0,:,:,0]*trace[0,:,:,0],origin='lower',
                       extent=(self.io.f[0], self.io.f[-1], self.io.r[0], self.io.r[-1]),
                       aspect='auto',
                       cmap='viridis')
            # plt.plot(self.io.f,self.io.r[self.F2_h],'.',color='white',markersize=3)
            # plt.plot(self.io.f,self.io.r[self.F1_h],'.',color='white',markersize=3)
            # plt.plot(self.io.f,self.io.r[self.E_h],'.',color='white',markersize=3)

            plt.axhline(self.io.r[self.F2_parameters[0]], color='red', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F2_parameters[2]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F2_parameters[3]], color='red', linestyle='--', linewidth=1)
            
            plt.axhline(self.io.r[self.F1_parameters[0]], color='white', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F1_parameters[2]], color='white', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.F1_parameters[3]], color='white', linestyle='--', linewidth=1)
            
            plt.axhline(self.io.r[self.E_parameters[0]], color='black', linestyle='--', linewidth=1)
            # plt.axhline(self.io.r[self.F2_parameters[1]], color='red', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.E_parameters[2]], color='black', linestyle='--', linewidth=1)
            plt.axvline(self.io.f[self.E_parameters[3]], color='black', linestyle='--', linewidth=1)
            plt.colorbar()
            self.save_fig(offset=True,style=False,fileExt="png")
            # plt.show()
            plt.close()
            # print(self.io.r[self.F2_parameters[0]],self.io.r[self.F2_parameters[1]],
            #       self.io.f[self.F2_parameters[2]],self.io.f[self.F2_parameters[3]])
    
    def make_fn(self, offset=False, style=False, fileExt="png"):
        path = self.set_path(
            self.path_save,
            self.year,
            self.month,
            self.day,
            self.hour,
            offset=offset,
            style=style
        )
        os.makedirs(path, exist_ok=True)
        # print("Save path:", path)

        fn = self.set_fn(
            path,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            fileExt=fileExt,
            offset=offset,
            style=style
        )
        return fn
        # print("Save file:", fn_save)
            
    
    def save_fig(self, offset=False, style=False, fileExt="png"):
        fn_save = self.make_fn(offset=offset, style=style, fileExt=fileExt)
        plt.savefig(fn_save, dpi=300, bbox_inches="tight")
                      
        

