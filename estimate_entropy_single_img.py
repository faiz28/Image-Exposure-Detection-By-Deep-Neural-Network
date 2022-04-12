from audioop import avg
from distutils.command.build_scripts import first_line_re
import os
import cv2
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt
import csv





DIR = './Main Dataset/'


def main():
    Normal = os.listdir(DIR + '/Normal')
    OverExposed = os.listdir(DIR + '/OverExposed')
    UnderExposed = os.listdir(DIR + '/UnderExposed')
    Normal.sort()
    OverExposed.sort()
    UnderExposed.sort()
   
    j = 0
    k = 0 
    for i in range(len(Normal)):
        normal = Normal[i].split('-')[0][1:]
        over = 0
        under = 0
        if j< len(OverExposed):
            over = OverExposed[j].split('-')[0][1:]
            if (over < normal):
                while (over < normal):
                    j += 1
                    if j< len(OverExposed):
                        over = OverExposed[j].split('-')[0][1:]
                        if over==normal:
                            j+=1
                            break
                    else:
                        over = 0
                        break
            else:
                if normal == over:
                    j += 1
                    
    
        if k< len(UnderExposed):
            under = UnderExposed[k].split('-')[0][1:]
            if( under<normal):
                while under<normal:
                    k+=1
                    if k< len(UnderExposed):
                        under = UnderExposed[k].split('-')[0][1:]
                        if under==normal:
                            k+=1
                            break
                    else:
                        under = 0
                        break
            else:
                if normal==under:
                    k+=1
        
        if normal == over and normal == under:
            print(normal)
            head = []
            head.append(Normal[i])
            
            entropyDict = {'Red': [], 'Green': [], 'Blue': [], 'Gray': []}
            skewnessDict = {'Red': [], 'Green': [], 'Blue': [], 'Gray': []} 
            
            Label = ['/Normal/', '/OverExposed/', '/UnderExposed/']
            data = [str(Normal[i]),str(OverExposed[j-1]),str(UnderExposed[k-1])]
            all_en_skew = []
            for kk  in range(len(Label)):
                path = DIR+Label[kk]+data[kk]
                chImgSet, titleSet, histSet = different_label_image(path)
                # plot_img(chImgSet, titleSet, histSet)
                
                for ts in titleSet:
                    all_en_skew.append(ts)
                
            
            for aes in all_en_skew:
                head.append(str(aes.split(',')[1]))
                head.append(str(aes.split(',')[2]))
                skewnessDict[aes.split(',')[0]].append(float(aes.split(',')[1]))
                entropyDict[aes.split(',')[0]].append(float(aes.split(',')[2]))
          
            
            for ch in ['Red', 'Green', 'Blue', 'Gray']:			
                skewness_avg,skewness_std = estimate_avg_std_performance(skewnessDict[ch])
                entropy_avg,entropy_std  = estimate_avg_std_performance(entropyDict[ch])
                head.append(skewness_avg)
                head.append(skewness_std)
                head.append(entropy_avg)
                head.append(entropy_std)
            break

            

def different_label_image(imgPath):  
    chImgSet = []
    titleSet = []
    histSet = []

    imgSet = load_img(imgPath)

    
    for ch in ['Red', 'Green', 'Blue', 'Gray']:
        hist, entropy, skewness = estimate_skewness_entropy(imgSet[ch])
        msg = ch + ',' + str(skewness) + ',' + str(entropy)
        titleSet.append(msg)
        chImgSet.append(imgSet[ch])
        histSet.append(hist)
        plot_img(imgSet, titleSet, hist)
    
    return chImgSet, titleSet, histSet
    
            

def plot_img(imgSet, titleSet, histSet):
	plt.figure(figsize = (20, 20))
	j = 1
	for i in range(4):
		plt.subplot(2, 4, j)
		j += 1
		plt.imshow(imgSet[i], cmap = 'gray')
		plt.axis('off')
		plt.subplot(2, 4, j)
		j += 1
		plt.plot(histSet[i], color = 'k')
		plt.title(titleSet[i])
	plt.show()
	plt.close()

def estimate_skewness_entropy(img):
	hist = cv2.calcHist([img],[0], None, [256], [0,256])
	skewness = skew(hist)[0]
	normalizedData = hist / hist.sum() + 0.0000001
	entropy = -(normalizedData * np.log(normalizedData)).sum()
	print('Skewness: {}, Entropy: {}'.format(skewness, entropy))
	
	return hist, entropy, skewness

def estimate_avg_std_performance(performanceList):
	avg = np.average(np.array(performanceList))
	std = np.std(np.array(performanceList))

	return avg,std
	
def load_img(imgPath):
    bgrImg = cv2.imread(imgPath)
    grayImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2GRAY)
    blueImg, greenImg, redImg = cv2.split(bgrImg)
    imgSet = {'Red': redImg, 'Green': greenImg, 'Blue': blueImg, 'Gray': grayImg}
    return imgSet


if __name__ == '__main__':
	main()
