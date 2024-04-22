import re 
import glob 
import os 

fname_patt = "./include/thunder/coordinates/surface_elements/*hh"


def replace_with_int_pow(match) :
    arg = match.groups()[0]
    exp = int(match.groups()[1])
    return "math::int_pow<"+str(exp)+">("+str(arg)+")"

files = glob.glob(fname_patt)
print(files)
for f in files:
    path,name = os.path.split(f)
    with open(f,"r") as ff:
        body = ff.read()
        new = re.sub("Power\(([\S\s]+),([\d]+)\)",replace_with_int_pow,body)
        with open(os.path.join(path,name+"_new"),"w") as f2:
            f2.write(new)

