
"""
@author: Dr. Ruslan Sherstyukov , March 2026

Class structure:
    
    ParametersPostprocessing
    
"""

import os
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


class ParametersPostprocessing:
    
    fn_root = "KIRPARA_"
    path_root_load = "IONOGRAM_KIR"
    path_root_save = "IONOGRAM_KIR"
    
    def __init__(self,year,month,day):    
        self.year = year
        self.month = month
        self.day = day
        self.ParametersCorrection()
        self.DailyPlots()
        
    def ParametersCorrection(self):
        self.database,self.json_files = self.LoadParameters()
        self.json_files = self.change_fn(self.json_files)
        
        self.foF2 = self.none_to_nan(self.get_foF2())
        self.foF2_filtered = np.copy(self.foF2)
        self.foF2_filtered,self.foF2_spline,self.lim1_F2,self.lim2_F2 = self.FilterParameters(array=self.foF2_filtered,std_lim=0.3,offtrend=False)
        self.database = self.foF2_correction(self.database)
        
        self.foF1 = self.none_to_nan(self.get_foF1())
        self.foF1_filtered = np.copy(self.foF1)
        self.foF1_filtered,self.foF1_spline,self.lim1_F1,self.lim2_F1 = self.FilterParameters(array=self.foF1_filtered,std_lim=0.2,offtrend=True)
        self.database = self.foF1_correction(self.database)
        
        self.foE = self.none_to_nan(self.get_foE())
        self.foE_filtered = np.copy(self.foE)
        self.foE_filtered,self.foE_spline,self.lim1_E,self.lim2_E = self.FilterParameters(array=self.foE_filtered,std_lim=0.1,offtrend=True)
        self.database = self.foE_correction(self.database)
        
        self.fmin = self.none_to_nan(self.get_fmin())
        self.fmin_filtered = np.copy(self.fmin)
        
        self.fbEs = self.none_to_nan(self.get_fbEs())
        self.fbEs_filtered = np.copy(self.fbEs)
        
        self.hF = self.none_to_nan(self.get_hF())
        self.hF_filtered = np.copy(self.hF)
        
        self.hE = self.none_to_nan(self.get_hE())
        self.hE_filtered = np.copy(self.hE)
        
        self.time = self.get_time()

        self.save_json_database(self.json_files, self.database)
        self.save_csv()
        
    def DailyPlots(self):
        self.plot(self.foF2_spline,markersize=2,color='red',name="foF2")
        self.plot(self.foF2,markersize=5,color='red',name="foF2")
        self.plot(self.foF2_filtered,markersize=5,color='blue',lim1=self.lim1_F2,lim2=self.lim2_F2,name="foF2")
        self.save_fig(name="foF2")
        plt.close()
        
        self.plot(self.foF1_spline,markersize=2,color='red',name="foF1")
        self.plot(self.foF1,markersize=5,color='red',name="foF1")
        self.plot(self.foF1_filtered,markersize=5,color='blue',lim1=self.lim1_F1,lim2=self.lim2_F1,name="foF1")
        self.save_fig(name="foF1")
        plt.close()
        
        self.plot(self.foE_spline,markersize=2,color='red',name="foE")
        self.plot(self.foE,markersize=5,color='red',name="foE")
        self.plot(self.foE_filtered,markersize=5,color='blue',lim1=self.lim1_E,lim2=self.lim2_E,name="foE")
        self.save_fig(name="foE")
        plt.close()
        
        self.plot(self.hF,markersize=5,color='blue',name="hF")
        self.save_fig(name="hF")
        plt.close()
        
        self.plot(self.fmin,markersize=5,color='blue',name="fmin")
        self.save_fig(name="fmin")
        plt.close()
        
        self.plot(self.fbEs,markersize=5,color='blue',name="fbEs")
        self.save_fig(name="fbEs")
        plt.close()
        
        self.plot(self.hE,markersize=5,color='red',name="hE")
        self.plot(self.hE_filtered,markersize=5,color='blue',lim1=self.lim1_E,lim2=self.lim2_E,name="hE")
        self.save_fig(name="hE")
        plt.close()
        
        
    def FilterParameters(self,array,std_lim=0.3,offtrend=False):
        # array = self.none_to_nan(array)
        array = self.filter_rare_data(array, threshold=0.15)
        spline,std = self.make_spline(array,std_lim,max_iter=25,spline_coef_range=[0,50],spline_coef_step=0.1)
        array = self.filter_outliers(array,spline,std,n_std=2)
        if offtrend:
            lim1,lim2 = self.trend_limits(array)
            array = self.filter_offtrend_data(array,lim1,lim2)
            spline,std = self.make_spline(array,std_lim,max_iter=25,spline_coef_range = [0,50],spline_coef_step=5)
        else:
            lim1,lim2 = False,False
        return array,spline,lim1,lim2
        
    
    def LoadParameters(self):   
        database = []
        path = self.set_path(self.path_root_load,
                             self.year,
                             self.month,
                             self.day,
                             offset=True)
        json_files = self.load_json_fn(path)
        
        for fn in json_files:
            data = self.open_json(fn)
            database.append(data)
        return database,json_files    
    
    
    def foF2_correction(self,database):
        filtered_data = self.filtered_data(self.foF2,self.foF2_filtered)
        print(filtered_data)
        for n in filtered_data:
            database[n]["V"][9] = None
        return database
    
    
    def foF1_correction(self,database):
        filtered_data = self.filtered_data(self.foF1,self.foF1_filtered)
        print(filtered_data)
        for n in filtered_data:
            database[n]["V"][8] = None
        return database
    
    
    def foE_correction(self,database):
        filtered_data = self.filtered_data(self.foE,self.foE_filtered)
        print(filtered_data)
        for n in filtered_data:
            database[n]["V"][6] = None
        return database
    
    
    def hE_correction(self,database):
        filtered_data = self.filtered_data(self.hE,self.foE_filtered)
        print(filtered_data)
        for n in filtered_data:
            database[n]["V"][5] = None
        return database
       
        
    def get_foF2(self):
        return [array["V"][9] for array in self.database]
    
    
    def get_foF1(self):
        return [array["V"][8] for array in self.database]
    
    
    def get_foE(self):
        return [array["V"][6] for array in self.database]
    
    
    def get_fmin(self):
        return [array["V"][0] for array in self.database]
    
    
    def get_fbEs(self):
        return [array["V"][4] for array in self.database]
    
    
    def get_hF(self):
        return [array["V"][7] for array in self.database]
    
    
    def get_hE(self):
        return [array["V"][5] for array in self.database]
    

    # def get_time(self):
    #     return [array["time"] for array in self.database]
    def get_time(self):
        return [datetime(*map(int, array["time"]))for array in self.database]
   
    
    
    def filtered_data(self,array,array_filtered):
        return np.where(np.isnan(array_filtered) & ~np.isnan(array))[0]
        
        
    def set_path(self,path_root,year,month,day,offset): 
        fn_root = self.fn_root
        if offset is False:
            return f"{path_root}/{fn_root}{year}_PP/{fn_root}{year}{month}_PP/{fn_root}{year}{month}{day}_PP/"
        elif offset is True:
            return f"{path_root}/{fn_root}{year}_AI/{fn_root}{year}{month}_AI/{fn_root}{year}{month}{day}_AI/"
        
    
    def set_fn(self,fn_root,year,month,day,name=None,fileExt="png",offset=False):
        
        if offset is False:
            return f"{fn_root}{year}{month}{day}_{name}.{fileExt}"
        elif offset is True:
            return f"{fn_root}{year}{month}{day}_AI_{name}.{fileExt}"
       
    
    def load_json_fn(self, path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
        def extract_time(fname):
            base = os.path.basename(fname)
            m = re.search(r"\d{8}_\d{6}", base)
            if not m:
                raise ValueError(f"No timestamp found in filename: {base}")
            return datetime.strptime(m.group(), "%Y%m%d_%H%M%S")

        return sorted(files, key=extract_time)

    
        
    
    def open_json(self,fn):
        with open(fn, 'r') as json_file:
            return json.load(json_file)
        
    
    def save_json_database(self, json_files, database):
        for fn, data in zip(json_files, database):
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, "w") as json_file:
                json.dump(data, json_file, indent=4)
    
    
    def change_fn(self, json_files, suffix="PP"):
        return [fn.replace("_AI", f"_AI_{suffix}") for fn in json_files]
    
    
    def none_to_nan(self,array):
        return np.array([np.nan if v is None else v for v in array])
    
    
    def filter_rare_data(self,array, threshold=0.15):
        intensity_array = (~np.isnan(array)).astype(int)
        window = len(array)//24
        kernel = np.ones(window)
        intensity_array = np.convolve(intensity_array, kernel, mode='same')/window
        array[intensity_array < threshold] = np.nan
        return array
    
    
    def make_spline(self,array,std_lim,spline_coef_step=1,max_iter=25,spline_coef_range = [0,50]):
        
        x = np.arange(len(array))
        mask = ~np.isnan(array)
        x_valid = x[mask]
        y_valid = array[mask]
        spline_coef = spline_coef_range[0]
        std = std_lim-1
        
        if np.sum(~np.isnan(array))>len(array)//3:
            while spline_coef<=spline_coef_range[1]:
                if  std >= std_lim:
                    break
                spline_coef += spline_coef_step
                spline_fit = UnivariateSpline(x_valid, y_valid, s=spline_coef)
                spline = spline_fit(x)
                std = np.std(array[mask] - spline[mask])  
        else:
            spline = std = None  
        print(std)
        return spline,std
       
    
    
    def filter_outliers(self,array,spline,std,n_std=2):
        if spline is not None:
            array[np.where(abs(array-spline)>n_std*std)]=np.nan
        return array
    
    
    def trend_limits(self,array):
        N = len(array)
        half = N // 2
        if np.any(~np.isnan(array[:half])):
            lim1 = np.nanargmin(array[:half])
        else:
            lim1 = np.nan
        if np.any(~np.isnan(array[half:])):
            lim2 = np.nanargmin(array[half:])
            lim2 = half + lim2
        else:
            lim2 = np.nan  
        return lim1,lim2
    
    
    def filter_offtrend_data(self,array,lim1,lim2):
        if ~np.isnan(lim1):
            array[:lim1-1] = np.nan
        if ~np.isnan(lim2):
            array[lim2+1:] = np.nan
        return array
        
        
    def plot(self,array,lim1=False,lim2=False,markersize=3,color="red",name=None):
        if array is not None and np.sum(~np.isnan(array))>1:
            x = np.arange(len(array))
            plt.plot(x, array,".",markersize=markersize,color=color)
            plt.xlim(0, len(array))
            plt.ylim(np.nanmin(array)-1, np.nanmax(array)+1)
            # plt.ylim(0, 10)
            if lim1 or lim2:
                plt.axvline(lim1, color=color, linestyle='--')
                plt.axvline(lim2, color=color, linestyle='--')
            if name[0]=="f":
                plt.ylabel("Frequency, MHz")
            elif name[0]=="h":
                plt.ylabel("Height, km")
            else:
                plt.ylabel(None)   
            plt.xlabel("time")
            plt.title(f"{self.year}{self.month}{self.day}_{name}")
            step = len(array)//12
            tick_positions = range(0, len(array), step)
            tick_labels = [self.time[i].strftime('%H') for i in tick_positions]
            plt.xticks(tick_positions, tick_labels)
       
            
            
    def save_fig(self,name):
        path = self.set_path(self.path_root_load,
                             self.year,
                             self.month,
                             self.day,
                             offset=True)
        fn = self.set_fn(path,
                         self.year,
                         self.month,
                         self.day,
                         name,
                         fileExt="png",
                         offset=False)
        fn = self.change_fn([fn], suffix="FIG")[0]
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        
    
    def save_csv(self):
        path = self.set_path(self.path_root_load,
                             self.year,
                             self.month,
                             self.day,
                             offset=True)
        fn = self.set_fn(path,
                         self.year,
                         self.month,
                         self.day,
                         name="Parameters",
                         fileExt="csv",
                         offset=False)
        fn = self.change_fn([fn], suffix="CSV")[0]
        os.makedirs(os.path.dirname(fn), exist_ok=True)

        df = pd.DataFrame({
            "time": self.time,
            "foF2": self.foF2,
            "foF1": self.foF1,
            "foE": self.foE,
            "fbEs": self.fbEs,
            "fmin": self.fmin,
            "hF": self.hF,
            "hE": self.hE,
            "foF2_filtered": self.foF2_filtered,
            "foF1_filtered": self.foF1_filtered,
            "foE_filtered": self.foE_filtered,
            "fbEs_filtered": self.fbEs_filtered,
            "fmin_filtered": self.fmin_filtered,
            "hF_filtered": self.hF_filtered,
            "hE_filtered": self.hE_filtered
        })
        df = df.round(2)
        df.to_csv(fn, index=False)
    