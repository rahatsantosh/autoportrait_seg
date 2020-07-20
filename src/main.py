import sys, getopt
import os

def main(argv):
   try:
      opts, args = getopt.getopt(argv,"htvp:",["help","train","validate","pretrained"])
   except getopt.GetoptError:
      print('main.py -t -v -p<directory>')
      sys.exit(2)
   for opt, arg in opts:
       if opt in ("-h", "--help"):
         print('''main.py -t  ---> train model
         main.py -v  ---> validate model
         main.py -p <directory>  ---> download pretrained model''')
         sys.exit()
       elif opt in ("-t", "--train"):
         os.system("python training/train.py")
       elif opt in ("-v", "--validate"):
         os.system("python training/validate.py")
       elif opt in ("-p", "--pretrained"):
         os.system("python data/get_pretrained.py")
       print("----Use src/trial.py or notebooks/demo.ipynb to try out the model----")

if __name__ == "__main__":
   main(sys.argv[1:])
