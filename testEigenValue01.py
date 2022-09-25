# testEigenValue01.py
# 固有値問題解法テスト01
import sys
import numpy as np
import pandas as pd

print('testEigenValue01.py : Start.')

srcMatrixFile = 'Matrix02.csv'
limitDv = 1.0e-09

args = sys.argv
if len(args) >= 2:
    srcMatrixFile = args[1]
print('Source Matrix File: ', srcMatrixFile)

dfMatrix = pd.read_csv(srcMatrixFile, header=None)
matrixRowSize = dfMatrix.shape[0]
matrixColSize = dfMatrix.shape[1]
matrixSize = np.min(dfMatrix.shape[0])
print(dfMatrix)
print('Row Size: ', matrixRowSize)
print('Col Size: ', matrixColSize)
print('Matrix Size: ', matrixSize)
arrayMatrix = []
for i in range(0, matrixSize):
    arrayRow = []
    for j in range(0, matrixSize):
        value = dfMatrix[i][j]
        arrayRow.append(value)
    arrayMatrix.append(arrayRow)
print('')
print('Origin Matrix: [')
for i, matrixV in enumerate(arrayMatrix):
    print(' [', end='')
    for j, x in enumerate(matrixV):
        if j > 0:
            print(',', end='')
        print('{0:9.6f}'.format(x), end='')
    print(']')
print(']')
print('')

# Unit Vectors
arrayUnitV = []
for i in range(0, matrixSize):
    unitV = []
    for j in range(0, matrixSize):
        if i == j:
            unitV.append(1.0)
        else:
            unitV.append(0.0)
    arrayUnitV.append(unitV)

# Initial Vectors
arrayRamdaV = []
arrayEigenV = []
for eigenNo in range(0, matrixSize):
    # Set Initial Vetors
    arrayTransV = []
    for i in range(0, matrixSize):
        unitV = arrayUnitV[i]
        transV = []
        for j in range(0, matrixSize):
            transV.append(unitV[j])
        arrayTransV.append(transV)
    flgContinue = True
    count = 0
    arrayTransDif = None
    eigenV = None
    while flgContinue:
        count += 1
        arrayRamda = []
        arrayWorkV = []
        arrayDifV = []
        arrayDif = []
        arrayTransDif = []

        for i in range(0, matrixSize):
            transV0 = arrayTransV[i]

            # 変換ベクトル
            transV = []
            ramdaSquare = 0.0
            for j in range(0, matrixSize):
                transVX = 0.0
                for k in range(0, matrixSize):
                    transVX += arrayMatrix[j][k] * transV0[k]
                transV.append(transVX)
                ramdaSquare += transVX ** 2
            ramda = np.sqrt(ramdaSquare)
            arrayRamda.append(ramda)
            
            # 差ベクトル
            difV = []
            dif = 0.0
            difSquare = 0.0
            for j in range(0, matrixSize):
                difX = transV[j] - transV0[j]
                difV.append(difX)
                difSquare += difX ** 2
            dif = np.sqrt(difSquare)
            arrayDif.append(dif)
            arrayDifV.append(difV)

            # 変換ベクトルと平行な単位ベクトル
            transV1 = []
            transDif = 0.0
            transDifSquare = 0.0
            for j in range(0, matrixSize):
                if ramda < limitDv:
                    transV1X = 0
                else:
                    transV1X = transV[j] / ramda
                transV1.append(transV1X)
                transDifSquare += (transV1X - transV0[j]) ** 2
            arrayWorkV.append(transV1)
            transDif = np.sqrt(transDifSquare)
            arrayTransDif.append(transDif)

        ramdaMax = max(arrayRamda)
        ramdaMaxIdx = np.argmax(arrayRamda)
        transDifMax = arrayTransDif[ramdaMaxIdx]
        if transDifMax < limitDv:
            # Finish One Axis
            arrayRamdaV.append(ramdaMax)
            eigenV = arrayTransV[ramdaMaxIdx]
            arrayEigenV.append(eigenV)
            flgContinue = False
        else:
            # Continue
            for i in range(0, matrixSize):
                arrayTransV[i] = arrayWorkV[i]
                # arrayTransV[i] = arrayDifV[i]

    print('Eigen No: {0}, Process finished on count: {1}'.format(eigenNo, count))
    
    # Prepare Next
    matrixEigen = []
    for i in range(0, matrixSize):
        arrayEigen = []
        for j in range(0, matrixSize):
            arrayEigen.append(ramdaMax * eigenV[i] * eigenV[j])
        matrixEigen.append(arrayEigen)
    for i, arrayEigen in enumerate(matrixEigen):
        for j, x in enumerate(arrayEigen):
            arrayMatrix[i][j] -= x

# Display Result
print('Eigen Value: [', end='')
for eigenNo, ramda in enumerate(arrayRamdaV):
    if eigenNo > 0:
        print(',', end='')
    print('{0:9.6f}'.format(ramda), end='')
print(']')
print('Eigen Vector: [')
for eigenNo, transV0 in enumerate(arrayEigenV):
    print('{}: ['.format(eigenNo), end='')
    for j, x in enumerate(transV0):
        if j > 0:
            print(',', end='')
        print('{0:9.6f}'.format(x), end='')
    print(']')
print(']')

print('testEigenValue01.py : End.')
sys.exit(0)