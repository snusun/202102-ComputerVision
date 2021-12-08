import math
import glob
import numpy as np
from PIL import Image, ImageDraw


# parameters

datadir = './data'
resultdir='./result'

# you can calibrate these parameters
sigma=3
highThreshold=0.3
lowThreshold=0.7
rhoRes=1
thetaRes = math.pi/360
nLines=20


def ConvFilter(Igs, G):
    # TODO ...
    igs = Igs.shape
    g = G.shape 
    img_pad = np.pad(Igs, ((int(g[0]/2), int(g[0]/2)),(int(g[1]/2), int(g[1]/2))), 'edge') 
    Iconv = np.zeros(igs)
    for i in range(igs[0]):
        for j in range(igs[1]):
            Iconv[i, j] = (G * img_pad[i:i + g[0], j:j + g[1]]).sum()
    return Iconv

def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...
    x, y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-((d)**2 / (2.0 * sigma**2)))

    gau_img = ConvFilter(Igs, g)
    Ix = ConvFilter(gau_img, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
    Iy = ConvFilter(gau_img, np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))
    Im_s = np.sqrt(Ix*Ix +Iy*Iy)
    Io = np.arctan2(Iy, Ix)/thetaRes

    ### non maximal suppression ###
    im = Im_s.shape
    Im_n = np.copy(Im_s)
    angle = np.copy(Io)
    angle[angle<0] += 180

    for i in range(im[0]):
        for j in range(im[1]):
            try:
                p = 255
                q = 255
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180): # angle 0
                    p = Im_s[i, j+1]
                    q = Im_s[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5): #angle 45
                    p = Im_s[i+1, j-1]
                    q = Im_s[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5): #angle 90
                    p = Im_s[i+1, j]
                    q = Im_s[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5): #angle 135
                    p = Im_s[i-1, j-1]
                    q = Im_s[i+1, j+1]
                if not ((Im_s[i,j] >= p) and (Im_s[i,j] >= q)):
                    Im_n[i,j] = 0
            except IndexError as e:
                pass

    ### double thresholding ###
    h = Im_n.max() * highThreshold
    l = highThreshold * lowThreshold
    
    im_n = Im_n.shape
    Im = np.zeros((im_n[0], im_n[1]), dtype=np.int32)
    
    s_1, s_2 = np.where(Im_n>h)
    z_1, z_2 = np.where(Im_n<l)
    w_1, w_2 = np.where((Im_n<=h) & (Im_n>=l))
    
    Im[z_1, z_2] = 0
    Im[s_1, s_2] = 1
    Im[w_1, w_2] = 0
    
    return Im, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...
    s = Im.shape
    theta_max = math.pi
    r_max = math.hypot(s[0], s[1])
    r_dim = 200 
    theta_dim = 300

    H = np.zeros((r_dim,theta_dim))

    for x in range(s[0]):
        for y in range(s[1]):
            if Im[x,y] == 1: continue
            for itheta in range(theta_dim):
                theta = itheta * theta_max / theta_dim
                r = x*math.cos(theta) + y*math.sin(theta)
                ir = r_dim * r / r_max
                H[int(ir),itheta] += 1
    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...
    m = 35
    n = 15
    s =  H.shape
    H_pad = np.pad(H, ((int(m/2), int(m/2)), (int(n/2), int(n/2))), 'wrap')
    H_pad[0:int(m/2), :] = np.flip(H_pad[0:int(m/2), :], 1)
    H_pad[s[0]:s[0]+int(m/2), :] = np.flip(H_pad[s[0]:s[0]+int(m/2), :], 1)
    vote = np.zeros(s)

    for i in range(s[0]):
        for j in range(s[1]):
            if (H_pad[i:i+m, j:j+n]).max() == H[i,j]:
                vote[i,j] = H[i,j]
                H_pad[i+int(m/2), j+int(n/2)] += s[0]*s[1]
            else:
                vote[i, j] = 0

    loc = np.argsort(vote.ravel())[:-nLines-1:-1]
    lTheta, lRho = np.unravel_index(loc, vote.shape)

    ### non maximal suppression
    '''
    n_Theta = np.zeros(nLines)
    n_lRho = np.zeros(nLines)
    
    n2 = 0
    for n1 in range(0,nLines):
        for i in range(nLines):
            if n1 == 0:
                n_Theta[n2] = lTheta[n1]
                n_lRho[n2] = lRho[n1]
                n2 += 1
            else:
                if lRho[i] < 0:
                    lRho[i]*=-1
                    lTheta[i]-=np.pi
                
                closeness_rho = np.isclose(lRho[i],n_lRho[0:n2],atol = 10)
                closeness_theta = np.isclose(lTheta[i],n_Theta[0:n2],atol = np.pi/36)
                closeness = np.all([closeness_rho,closeness_theta],axis=0)
                #print(closeness, closeness_rho, closeness_theta)
                if not any(closeness) and n2 < 4:
                    n_Theta[n2] = lTheta[n1]
                    n_lRho[n2] = lRho[n1]
                    n2 += 1
                    '''

    return lRho, lTheta

def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...
    return l


def main():

    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        H = HoughTransform(Im, rhoRes, thetaRes)
        lRho, lTheta =HoughLines(H,rhoRes,thetaRes,nLines)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments

        H_img = Image.fromarray(np.clip(H, 0, 255).astype('uint8'))
        Im_img = Image.fromarray(np.clip(Im*255/Im.max(), 0, 255).astype('uint8'))
        H_img.save(resultdir+"/H_"+img_path[7:])
        Im_img.save(resultdir+"/Im_"+img_path[7:])

if __name__ == '__main__':
    main()