import os

path = "../../data"
os.mkdir(os.path.join(path, 'interim'))
os.mkdir(os.path.join(path, 'processed'))
os.mkdir(os.path.join(path, 'raw'))

pwd = os.getcwd()
os.chdir(os.path.join(path, 'raw'))
os.system("wget https://drive.google.com/u/0/uc?export=download&confirm=SOIv&id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv")
print("Dataset downloaded")
os.system("unzip CelebAMask-HQ.zip")
print("Dataset extracted")
os.chdir(pwd)

os.system("python data_preprocess.py")
