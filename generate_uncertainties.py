import os
import sys
from pathlib import Path
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_path, "mono_uncertainty"))
sys.path.insert(0, os.path.join(current_path, "semantic_uncertainty"))
sys.path.insert(0, os.path.join(current_path, "normal_uncertainty"))
#from mono_uncertainty.generate_single import evaluate, compute_pixelwise_eigen_errors
#from semantic_uncertainty.pipeline_demo import test_image
from normal_uncertainty.tools.test_any_images import normal_from_image	
from numpy import asarray
from numpy import savez_compressed
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
      


if __name__ == "__main__":
    config_path_depth = os.path.join(current_path, "mono_uncertainty", "configs", "config_depth.yaml")
    config_path_normal = os.path.join(current_path, "normal_uncertainty", "configs", "config_normal.yaml")
    dataset = "cityScapes_sample"
    
    
    train_set_names = ['zurich', 'stuttgart','dusseldorf']
    i = 0
    for setName in train_set_names:

        # dataset from Work
        #current_path = os.path.dirname(os.path.abspath(__file__))
        #current_path = os.path.join(current_path, "data", dataset,"img",setName)    # original img
        #current_path = os.path.join(current_path, "data", dataset,"foggy", setName)  # foggy img
        
        #  dataset from scratch
        current_path = '/cluster/scratch/boysun'
        current_path = os.path.join(current_path, dataset, 'img', setName)    # original img
        #current_path = os.path.join(current_path, dataset, 'foggy', setName)    # foggy img

        for filename in os.listdir(current_path):

            if filename.endswith(".png") :                                  # original img
            #if filename.endswith("beta_0.02.png") :                          # foggy img    

                print('current filename: ',filename)
                # ouput dir from Work
                Path(f"./test_imgs_cs/{i}").mkdir(parents=True, exist_ok=True)
                filename = current_path + "/" + filename
        
                #test_image(filename, i)                                    # semantic seg
                #evaluate(config_path_depth, filename, i)                   # depth 
                normal_from_image(config_path_normal, filename, i)           # normal

                i += 1

        # create zip file 
        # zipf = zipfile.ZipFile('test.zip', 'w', zipfile.ZIP_DEFLATED)
        # zipdir('./test_imgs_cs/', zipf)
        # zipf.close()