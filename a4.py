import sys
import os.path
import warp as wp
import numpy as np
from PIL import Image

wp.init()
device = "cpu"
sharpen = 0
noise = 0


#command line validation
if (len(sys.argv) != 6):
    print("Invalid number of arguments, correct usage: python3 ./a4.py algType kernSize param inFileName outFileName")
    sys.exit()
if(sys.argv[1] == '-s'):
    sharpen = 1
if(sys.argv[1] == "-n"):
    noise = 1
if(noise == 0 and sharpen == 0):
    print("Invalid algorithm flag: possible flags include \"-s\" and \"-n\"")
    sys.exit()

#kernel size
kernelSize = int(sys.argv[2])

#kernel validation
if((kernelSize < 0) or (kernelSize % 2 == 0)):
    print("Invalid kernel size: Must be an odd postitive number")
    sys.exit()

param = float(sys.argv[3])
inputFile = sys.argv[4]
outputFile = sys.argv[5]

#Identify file exists in dir
if(os.path.exists(inputFile)):
    image = Image.open(inputFile)
else:
    print("Invalid file name!")
    sys.exit()


#declare constants
kern = wp.constant(kernelSize)
sharp = wp.constant(sharpen)
denoise = wp.constant(noise)
q = wp.constant(param)

if(image.mode == "L"):
    mode = wp.constant(0)
if(image.mode == "RGBA"):
    mode = wp.constant(1)
if(image.mode == "RGB"):
    mode = wp.constant(2)



@wp.kernel
def RGB(inArr: wp.array(dtype=float, ndim=3),
        outArr: wp.array(dtype=float, ndim=3),
        medianArr: wp.array(dtype=float, ndim=1)):

    #sharpen RGB(A)
    if(sharp == 1):    
        i, j, k = wp.tid()
    
        if ((mode == 1) and (k == 3)):
            outArr[i, j, k] = inArr[i, j, k]

            #Account for overflow
            outArr[i, j, k] = wp.clamp(outArr[i,j,k], 0.0, 255.0)
        else:
            row = (((kern - 1) / 2) * -1)
            col = (((kern - 1) / 2) * -1)
            result = float(0.0)
            for x in range(kern):
                for y in range(kern):
                    outsideBorder = 0
                    
                    if(((i + row < 0) or (i + row >= numRows)) or (j + col < 0) or (j + col >= numCols)):
                        outsideBorder = 1
                    if (outsideBorder == 0):
                        result = result + inArr[i + row, j + col, k]
                    #kernel is off the image
                    else:
                        if(((i + row < 0) or (i + row >= numRows)) and ((j + col < 0) or (j + col >= numCols))):
                            outArr[i, j, k] = inArr[i, j, k]
                            outArr[i, j, k] = inArr[i, j, k] - outArr[i, j, k]
                            outArr[i, j, k] = inArr[i, j, k] + (q * (outArr[i, j, k]))
                            return
                        else:
                            result = result + inArr[i - row, j - col, k]
                    col = col + 1
                col = (((kern - 1) / 2) * -1)
                row = row + 1
            average = result / float(kern * kern)

            #blur image
            outArr[i, j, k] = average

            #edge image
            outArr[i, j, k] = inArr[i, j, k] - average

            #sharpen image
            outArr[i, j, k] = inArr[i, j, k] + (q * (outArr[i, j, k]))

            #Account for overflow
            outArr[i, j, k] = wp.clamp(outArr[i,j,k], 0.0, 255.0)
    #de-noise
    else:
        i, j, k = wp.tid()

        if ((mode == 1) and (k == 3)):
            outArr[i, j, k] = inArr[i, j, k]

            #Account for overflow
            outArr[i, j, k] = wp.clamp(outArr[i,j,k], 0.0, 255.0)
        else:
            i, j, k = wp.tid()
            count = int(0)
            row = (((kern - 1) / 2) * -1)
            col = (((kern - 1) / 2) * -1)
            for x in range(kern):
                for y in range(kern):
                    outsideBorder = 0
                    
                    if((i + row < 0) or (i + row >= numRows) or (j + col < 0) or (j + col >= numCols)):
                        outsideBorder = 1
                    if (outsideBorder == 0):
                        medianArr[count] = inArr[i + row, j + col, k]
                    else:
                        if(((i + row < 0) or (i + row >= numRows)) and ((j + col < 0) or (j + col >= numCols))):
                            #blur image
                            outArr[i, j, k] = inArr[i, j, k]
                            return
                        else:
                            medianArr[count] = inArr[i - row, j - col, k]
                    col = col + 1
                    count = count + 1
                col = (((kern - 1) / 2) * -1)
                row = row + 1

            #sort array
            for z in range(kern * kern):
                w = z + 1
                for w in range(kern * kern):
                    if(medianArr[z] > medianArr[w]):
                        a = medianArr[z]
                        medianArr[z] = medianArr[w]
                        medianArr[w] = a 

            medResult = medianArr[((kern * kern) - 1) / 2]

            #blur image
            outArr[i, j, k] = medResult

            #Account for overflow
            outArr[i, j, k] = wp.clamp(outArr[i, j, k], 0.0, 255.0)

@wp.kernel
def greyScale(inArr: wp.array(dtype=float, ndim=2),
              outArr: wp.array(dtype=float, ndim=2),
              medianArr: wp.array(dtype=float, ndim=1)):
    #sharpen image
    if(sharp == 1):  
        i, j = wp.tid()
        
        row = (((kern - 1) / 2) * -1)
        col = (((kern - 1) / 2) * -1)
        result = float(0.0)
        for x in range(kern):
            for y in range(kern):
                outsideBorder = 0
                
                if(((i + row < 0) or (i + row >= numRows)) or (j + col < 0) or (j + col >= numCols)):
                    outsideBorder = 1
                if (outsideBorder == 0):
                    result = result + inArr[i + row, j + col]
                #off border
                else:
                    if(((i + row < 0) or (i + row >= numRows)) and ((j + col < 0) or (j + col >= numCols))):
                        #blur image
                        outArr[i, j] = inArr[i, j]

                        #edge image
                        outArr[i, j] = inArr[i, j] - outArr[i, j]

                        #sharpen image
                        outArr[i, j] = inArr[i, j] + (q * (outArr[i, j]))

                        #Account for overflow
                        outArr[i, j] = wp.clamp(outArr[i,  j], 0.0, 255.0)
                        return
                    else:
                        result = result + inArr[i - row, j - col]
                col = col + 1
            col = (((kern - 1) / 2) * -1)
            row = row + 1
        average = result / float(kern * kern)

        #blur image
        outArr[i, j] = average

        #edge image
        outArr[i, j] = inArr[i, j] - average

        #sharpen image
        outArr[i, j] = inArr[i, j] + (q * (outArr[i, j]))

        #Account for overflow
        outArr[i, j] = wp.clamp(outArr[i,j], 0.0, 255.0)
    #De-noise
    else:
        i, j = wp.tid()
        count = int(0)
        row = (((kern - 1) / 2) * -1)
        col = (((kern - 1) / 2) * -1)
        for x in range(kern):
            for y in range(kern):
                outsideBorder = 0
                
                if((i + row < 0) or (i + row >= numRows) or (j + col < 0) or (j + col >= numCols)):
                    outsideBorder = 1
                if (outsideBorder == 0):
                    medianArr[count] = inArr[i + row, j + col]
                else:
                    if(((i + row < 0) or (i + row >= numRows)) and ((j + col < 0) or (j + col >= numCols))):
                        outArr[i, j] = inArr[i, j]
                        return
                    else:
                        medianArr[count] = inArr[i - row, j - col]
                col = col + 1
                count = count + 1
            col = (((kern - 1) / 2) * -1)
            row = row + 1

        #sort array
        for z in range(kern * kern):
            w = z + 1
            for w in range(kern * kern):
                if(medianArr[z] > medianArr[w]):
                    a = medianArr[z]
                    medianArr[z] = medianArr[w]
                    medianArr[w] = a 


        medResult = medianArr[((kern * kern) - 1) / 2]

        #blur image
        outArr[i, j] = medResult

        #Account for overflow
        outArr[i, j] = wp.clamp(outArr[i,j], 0.0, 255.0)

numpyArr = np.asarray(image, dtype='float32')
numRows = wp.constant(numpyArr.shape[0])
numCols = wp.constant(numpyArr.shape[1])

#Warp data structure for input data 
inWarpData = wp.array(numpyArr, dtype=float, device=device)

#Warp data structure for output data
outWarpImage = wp.zeros(shape=numpyArr.shape, dtype=float, device=device)

#Warp data sructure for median array
medianWarp = wp.zeros(shape=(kernelSize * kernelSize), dtype=float, device=device)

print(image.mode)
#launch kernel
if(mode == 0):
    wp.launch(kernel=greyScale,
            dim = numpyArr.shape,
            inputs=[inWarpData, outWarpImage, medianWarp],
            device=device)

if(mode == 1 or mode == 2):
    wp.launch(kernel=RGB,
            dim = numpyArr.shape,
            inputs=[inWarpData, outWarpImage, medianWarp],
            device=device)

#Generate and save new image to file
numpyOutArr = outWarpImage.numpy()
imageOut = Image.fromarray(np.uint8(numpyOutArr))
imageOut.save(outputFile)