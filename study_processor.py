import os
import sys
import numpy
from pathlib import Path
import shutil
from natsort import natsorted

def seperate_studies():
    
    unique_studies = set()
    counter = 0

    for i in ["Amyloidosis", "HCM", "HTN"]:

        dir_path = Path(images_path + i)
        directory = list(dir_path.glob('**/*.png'))

        for file in directory:

            name = str(file).replace("-", "_")

            counter += 1
            filename = str(name).split('/')[-1]
            study = str(name).split('/')[-1].split('_')[0]
            view = str(name).split('/')[-1].split('_', 1)[1].split('.')[0].strip().upper()

            if view == "A2" or view == "A3" or view == "A4":

                view = view[0] + "P" + view[1]

            if view == "PSAX_V1":

                view = "PSAX_V"

            if study not in unique_studies:

                unique_studies.add(study)
                new_path = study_path + study
                Path(new_path).mkdir(parents=True, exist_ok = True)

                for j in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

                    for k in ["Amyloidosis", "HCM", "HTN"]:

                        Path(new_path + "/{0}/{1}".format(j, k)).mkdir(parents = True, exist_ok = True)

            shutil.copyfile(file, study_path + "{0}/{1}/{2}/{3}".format(study, view, i, filename))
        
        print(dir_path, counter)
        counter = 0
        
    