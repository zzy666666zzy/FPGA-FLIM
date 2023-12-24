import numpy as np

def Record_Tensor_txt(filename_txt,tensor,name):
	print ("Recording tensor "+name+" ...")
	f = open(filename_txt+name+'.txt', 'w')
	array=tensor;
	if(np.size(np.shape(array))==1):
		Record_Array1D(array,name,f)
	else:
		if(np.size(np.shape(array))==2):
			Record_Array2D(array,name,f)
		else:
			if(np.size(np.shape(array))==3):
				Record_Array3D(array,name,f)
			else:
				Record_Array4D(array,name,f)
	f.close();

def Record_Array1D(array,name,f):
	for i in range(np.shape(array)[0]):
		f.write(str(array[i])+"\n");

def Record_Array2D(array,name,f):
	for j in range(np.shape(array)[1]):
		for i in range(np.shape(array)[0]):
			f.write(str(array[i][j])+"\n");

def Record_Array3D(array,name,f):
	for k in range(np.shape(array)[2]):
		for j in range(np.shape(array)[1]):
			for i in range(np.shape(array)[0]):
				f.write(str(array[i][j][k])+"\n");
                
def Record_Array4D(array,name,f):
 	for l in range(np.shape(array)[3]):
         for k in range(np.shape(array)[2]):
             for j in range(np.shape(array)[1]):
                 for i in range(np.shape(array)[0]):
                     f.write(str(array[i][j][k][l])+"\n");

# def Record_Array4D(array,name,f):
# 	for i in range(np.shape(array)[0]):
# 		for j in range(np.shape(array)[1]):
# 			for k in range(np.shape(array)[2]):
# 				for l in range(np.shape(array)[3]):
# 					f.write(str(array[i][j][k][l])+"\n");