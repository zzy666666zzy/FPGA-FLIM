
import json

dataWidth = 16
dataIntWidth = 1
weightIntWidth = 4
inputFile = "WeigntsAndBiases.txt"
dataFracWidth = dataWidth-dataIntWidth
weightFracWidth = dataWidth-weightIntWidth
biasIntWidth = dataIntWidth+weightIntWidth
biasFracWidth = dataWidth-biasIntWidth
outputPath = "./w_b/"
headerPath = "./"
#%%
def Decimal_to_Binary(num,dataWidth,fracBits):
	if num >= 0:
		num = num * (2**fracBits) #represent in decimal
		num = int(num) #round it to nearest integer
		d = num
	else:
		num = -num
		num = num * (2**fracBits)
		num = int(num)
		if num == 0:
			d = 0
		else:
			d = 2**dataWidth - num
	return d
#%%
def twos_comp(val,integer_precision,fraction_precision):
    flipped = ''.join(str(1-int(x))for x in val)
    length = '0' + str(integer_precision+fraction_precision) + 'b'
    bin_literal = format((int(flipped,2)+1),length)
    return bin_literal
#%%
def float_to_fp(num,integer_precision,fraction_precision):   
    if(num<0):
        sign_bit = 1 #sign bit is 1 for negative numbers in 2's complement representation
        num = -1*num
    else:
        sign_bit = 0
    precision = '0'+ str(integer_precision) + 'b'
    integral_part = format(int(num),precision)
    fractional_part_f = num - int(num)
    fractional_part = []
    for i in range(fraction_precision):
        d = fractional_part_f*2
        fractional_part_f = d -int(d)        
        fractional_part.append(int(d))
    fraction_string = ''.join(str(e) for e in fractional_part)
    if(sign_bit == 1):
        binary = str(sign_bit) + twos_comp(integral_part + fraction_string,integer_precision,fraction_precision)
    else:
        binary = str(sign_bit) + integral_part+fraction_string
    return str(binary)
#%%
def genWaitAndBias(dataWidth,weightFracWidth,biasFracWidth,inputFile):
	weightIntWidth = dataWidth-weightFracWidth
	biasIntWidth = dataWidth-biasFracWidth
	myDataFile = open(inputFile,"r")
	weightHeaderFile = open(headerPath+"weightValues.h","w")
	myData = myDataFile.read()
	myDict = json.loads(myData)
	myWeights = myDict['weights']
	myBiases = myDict['biases']
	weightHeaderFile.write("int weightValues[]={")
    #weights
	for layer in range(0,len(myWeights)):
		for neuron in range(0,len(myWeights[layer])):
			fi = 'w_'+str(layer+1)+'_'+str(neuron)+'.mif'
			f = open(outputPath+fi,'w')
			for weight in range(0,len(myWeights[layer][neuron])):
				if 'e' in str(myWeights[layer][neuron][weight]):
					p = '0'
				else:
					if myWeights[layer][neuron][weight] > 2**(weightIntWidth-1):
						myWeights[layer][neuron][weight] = 2**(weightIntWidth-1)-2**(-weightFracWidth)
					elif myWeights[layer][neuron][weight] < -2**(weightIntWidth-1):
						myWeights[layer][neuron][weight] = -2**(weightIntWidth-1)
					wInDec = Decimal_to_Binary(myWeights[layer][neuron][weight],dataWidth,weightFracWidth)
					p = bin(wInDec)[2:]
				f.write(p+'\n')
				weightHeaderFile.write(str(wInDec)+',')
			f.close()
	weightHeaderFile.write('0};\n')
	weightHeaderFile.close()
	
    #bias
	biasHeaderFile = open(headerPath+"biasValues.h","w")
	biasHeaderFile.write("int biasValues[]={")
	for layer in range(0,len(myBiases)):
		for neuron in range(0,len(myBiases[layer])):
			fi = 'b_'+str(layer+1)+'_'+str(neuron)+'.mif'
			p = myBiases[layer][neuron][0]
			if 'e' in str(p): #To remove very small values with exponents
				res = '0'
			else:
				if p > 2**(biasIntWidth-1):
					p = 2**(biasIntWidth-1)-2**(-biasFracWidth)
				elif p < -2**(biasIntWidth-1):
					p = -2**(biasIntWidth-1)
				bInDec = Decimal_to_Binary(p,dataWidth,biasFracWidth)
				res = bin(bInDec)[2:]
			f = open(outputPath+fi,'w')
			f.write(res)
			biasHeaderFile.write(str(bInDec)+',')
			f.close()
	biasHeaderFile.write('0};\n')
	biasHeaderFile.close()
			
# if __name__ == "__main__":
# 	genWaitAndBias(dataWidth,weightFracWidth,biasFracWidth,inputFile)