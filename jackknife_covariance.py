import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import math
from shapely.geometry import Polygon, Point


#I compute the covariance matrix from a set of clustering measurements in
#independent zones calculated from the jackknife technique
#the clustering measurements can be replaced by any other wanted measurement

#I read the fits file to obtain the wanted data
event_filename = get_pkg_data_filename('60fields.fits')
events = Table.read(event_filename, hdu=1)
hdul = fits.open(event_filename)
data = hdul[1].data  

#extract the colums from the table
dec = data['DEC']
ra = data['RA']
redshift = data['Z']

#some specific selection
dec_sel = dec[:1593]
ra_sel = ra[:1593]
redshift_sel = redshift[:1593]

#redshift selection from 3 to 6
select = (redshift_sel >= 3 ) & (redshift_sel <= 6.)
Zf_wide = redshift_sel[select]
DECf_wide = dec_sel[select] 
RAf_wide = ra_sel[select] 


#function to split the sample area in different jackknife zones
def dec_cut(ra_values,a,b):
    #a and b: parameters of the line y=mx+a
    y=a*ra_values+b #equation of the line
    return y

cm = plt.cm.get_cmap('jet') 
fig = plt.figure().add_subplot(111)
plt.scatter(RAf_wide,DECf_wide, s=10, c=Zf_wide, marker='o', cmap=cm)
plt.gca().invert_xaxis()
plt.text(53.25,-27.73,'zone3')
plt.text(53.27,-27.825,'zone4')
plt.text(53.17,-27.71,'zone5')
plt.text(53.23,-27.87,'zone6')
plt.text(53.11,-27.73,'zone7')
plt.text(53.17,-27.88,'zone8')
plt.text(53.065,-27.75,'zone9')
plt.text(53.1,-27.9,'zone10')
colorbar=plt.colorbar()
plt.plot(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)]+0.004,dec_cut(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)],-2.664,113.95),color='k') 
plt.plot(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)]-0.055,dec_cut(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)],-2.664,113.95),color='k') 
plt.plot(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)]-0.115,dec_cut(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)],-2.664,113.95),color='k')     
plt.plot(RAf_wide,dec_cut(RAf_wide,0.4192,-50.08213)+0.005,color='k')                                                                                            
colorbar.set_label('z')
colorbar.ax.tick_params( direction='in')
plt.clim(3., 6)  
fig.xaxis.set_ticks_position('both')
fig.yaxis.set_ticks_position('both')
fig.xaxis.set_tick_params(direction='in', which='both')
fig.yaxis.set_tick_params(direction='in', which='both')
plt.xlabel("RA", fontsize=14)
plt.ylabel("Dec", fontsize=14)
plt.grid(False)
plt.tight_layout()
#plt.savefig('Jacknife zones',dpi=500)
plt.show()

#select all zones but one
RAf_wide3=np.hstack((RAf_wide[(RAf_wide<np.min(RAf_wide)+0.06*3)],RAf_wide[(DECf_wide<np.mean(DECf_wide))&(RAf_wide>np.min(RAf_wide)+0.06*3)]))
DECf_wide3=np.hstack((DECf_wide[(RAf_wide<np.min(RAf_wide)+0.06*3)],DECf_wide[(DECf_wide<np.mean(DECf_wide))&(RAf_wide>np.min(RAf_wide)+0.06*3)]))
Zf_wide3=np.hstack((Zf_wide[(RAf_wide<np.min(RAf_wide)+0.06*3)],Zf_wide[(DECf_wide<np.mean(DECf_wide))&(RAf_wide>np.min(RAf_wide)+0.06*3)]))

#calculate the clustering in the above subsample with the K-estimator from adelberger et al. 2005
#transverse and radial separations, rij and zij
zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide3):
    phi = np.sqrt((RAf_wide3[k]-RAf_wide3[k+1:])**2+(DECf_wide3[k]-DECf_wide3[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide3[k+1:])/2).value * phi * 0.7)      
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide3[k+1:]).value)*0.7)  

#clustering
kab_wide3 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide3 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                     
         kab_wide3 = np.append(kab_wide3, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide3 = np.append(err_wide3, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                        
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide3, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide3))])                                   
ax.errorbar(binp, kab_wide3, yerr=err_wide3, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show() 

#second subsample, all zones but one (different one than previously)
RAf_wide4=np.hstack((RAf_wide[(RAf_wide<np.min(RAf_wide)+0.06*3)],RAf_wide[(DECf_wide>np.mean(DECf_wide))&(RAf_wide>np.min(RAf_wide)+0.06*3)] ))
DECf_wide4=np.hstack((DECf_wide[(RAf_wide<np.min(RAf_wide)+0.06*3)],DECf_wide[(DECf_wide>np.mean(DECf_wide))&(RAf_wide>np.min(RAf_wide)+0.06*3)] ))
Zf_wide4=np.hstack((Zf_wide[(RAf_wide<np.min(RAf_wide)+0.06*3)],Zf_wide[(DECf_wide>np.mean(DECf_wide))&(RAf_wide>np.min(RAf_wide)+0.06*3)] ))

zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide4):
    phi = np.sqrt((RAf_wide4[k]-RAf_wide4[k+1:])**2+(DECf_wide4[k]-DECf_wide4[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide4[k+1:])/2).value * phi * 0.7)     
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide4[k+1:]).value)*0.7)  

kab_wide4 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide4 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                 
         kab_wide4 = np.append(kab_wide4, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide4 = np.append(err_wide4, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                         
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide4, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide4))])                                   
ax.errorbar(binp, kab_wide4, yerr=err_wide4, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show() 



#define polygons for the jackknife zones

#cross point of lines, vertices for a future polygon
m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.97 # slope & intercept (left vertical line)
x1 = (b1-b2) / (m2-m1)
y1 = m1 * x1 + b1
print('intersecting point of lines',x1,y1)

m1, b1 = 0.4192, -50.00213 # slope & intercept (top horizontal line)
m2, b2 = -2.664, 113.97 # slope & intercept (left vertical line)
x2 = (b1-b2) / (m2-m1)
y2 = m1 * x2 + b1
print('intersecting point of lines',x2,y2)

m1, b1 = 0.4192, -50.00213 # slope & intercept (top horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x3 = (b1-b2) / (m2-m1)
y3 = m1 * x3 + b1
print('intersecting point of lines',x3,y3)

m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x4 = (b1-b2) / (m2-m1)
y4 = m1 * x4 + b1
print('intersecting point of lines',x4,y4)

 
#Create a Polygon with the intersecting points, third subsample
coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
poly = Polygon(coords)
RAf_wide5=np.array([])
DECf_wide5=np.array([])
Zf_wide5=np.array([])
for i, item in enumerate(RAf_wide):
    if Point(item,DECf_wide[i]).within(poly)==False:  #check if point is within the polygon: true false answer
        RAf_wide5=np.append(RAf_wide5,item)
        DECf_wide5=np.append(DECf_wide5,DECf_wide[i])
        Zf_wide5=np.append(Zf_wide5,Zf_wide[i])

#clustering computation 
zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide5):
    phi = np.sqrt((RAf_wide5[k]-RAf_wide5[k+1:])**2+(DECf_wide5[k]-DECf_wide5[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide5[k+1:])/2).value * phi * 0.7)     
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide5[k+1:]).value)*0.7)  

kab_wide5 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide5 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                   
         kab_wide5 = np.append(kab_wide5, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide5 = np.append(err_wide5, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                         
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide5, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide5))])                                   
ax.errorbar(binp, kab_wide5, yerr=err_wide5, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show() 


#cross point of lines, vertices for a future polygon
m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.97 # slope & intercept (left vertical line)
x1 = (b1-b2) / (m2-m1)
y1 = m1 * x1 + b1
print('intersecting point of lines',x1,y1)

m1, b1 = 0.4192, -50.14713 # slope & intercept (bottom horizontal line)
m2, b2 = -2.664, 113.97 # slope & intercept (left vertical line)
x2 = (b1-b2) / (m2-m1)
y2 = m1 * x2 + b1
print('intersecting point of lines',x2,y2)

m1, b1 = 0.4192, -50.14713 # slope & intercept (bottom horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x3 = (b1-b2) / (m2-m1)
y3 = m1 * x3 + b1
print('intersecting point of lines',x3,y3)

m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x4 = (b1-b2) / (m2-m1)
y4 = m1 * x4 + b1
print('intersecting point of lines',x4,y4)
 

#Create a Polygon with the intersecting points, forth subsample
coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
poly = Polygon(coords)
RAf_wide6=np.array([])
DECf_wide6=np.array([])
Zf_wide6=np.array([])
for i, item in enumerate(RAf_wide):
    if Point(item,DECf_wide[i]).within(poly)==False:  
        RAf_wide6=np.append(RAf_wide6,item)
        DECf_wide6=np.append(DECf_wide6,DECf_wide[i])
        Zf_wide6=np.append(Zf_wide6,Zf_wide[i])


zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide6):
    phi = np.sqrt((RAf_wide6[k]-RAf_wide6[k+1:])**2+(DECf_wide6[k]-DECf_wide6[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide6[k+1:])/2).value * phi * 0.7)     
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide6[k+1:]).value)*0.7)  

kab_wide6 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide6 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                    
         kab_wide6 = np.append(kab_wide6, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide6 = np.append(err_wide6, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                         
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide6, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide6))])                                   
ax.errorbar(binp, kab_wide6, yerr=err_wide6, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show() 


#cross point of lines, vertices for a future polygon
m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x1 = (b1-b2) / (m2-m1)
y1 = m1 * x1 + b1
print('intersecting point of lines',x1,y1)

m1, b1 = 0.4192, -50.00213 # slope & intercept (top horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x2 = (b1-b2) / (m2-m1)
y2 = m1 * x2 + b1
print('intersecting point of lines',x2,y2)

m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x3 = (b1-b2) / (m2-m1)
y3 = m1 * x3 + b1
print('intersecting point of lines',x3,y3)

m1, b1 = 0.4192, -50.00213 # slope & intercept (top horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x4 = (b1-b2) / (m2-m1)
y4 = m1 * x4 + b1
print('intersecting point of lines',x4,y4)


#Create a Polygon with the intersecting points
coords = [(x1, y1), (x2, y2), (x4, y4), (x3, y3)] #careful bc the order of the vertices matters
poly = Polygon(coords)
RAf_wide7=np.array([])
DECf_wide7=np.array([])
Zf_wide7=np.array([])
for i, item in enumerate(RAf_wide):
    if Point(item,DECf_wide[i]).within(poly)==False:  
        RAf_wide7=np.append(RAf_wide7,item)
        DECf_wide7=np.append(DECf_wide7,DECf_wide[i])
        Zf_wide7=np.append(Zf_wide7,Zf_wide[i])

#check that we excluded right the zone
cm = plt.cm.get_cmap('jet') 
fig = plt.figure().add_subplot(111)
plt.scatter(RAf_wide7, DECf_wide7, s=10, c=Zf_wide7, marker='o', cmap=cm)
plt.gca().invert_xaxis()
plt.scatter(x1,y1,marker='x',color='r',s=30)
plt.scatter(x2,y2,marker='x',color='r',s=30)
plt.scatter(x3,y3,marker='x',color='r',s=30)
plt.scatter(x4,y4,marker='x',color='r',s=30)
plt.plot(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)]+0.01,dec_cut(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)],-2.664,113.95),color='k') 
plt.plot(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)]-0.055,dec_cut(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)],-2.664,113.95),color='k') 
plt.plot(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)]-0.115,dec_cut(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)],-2.664,113.95),color='k') 
plt.plot(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)]-0.175,dec_cut(RAf_wide[(RAf_wide<53.25)&(RAf_wide>53.16)],-2.664,113.95),color='k')    
plt.plot(RAf_wide,dec_cut(RAf_wide,0.4192,-50.07713),color='k')                                                                                            
plt.plot(RAf_wide,dec_cut(RAf_wide,0.4192,-50.00213),color='k')
plt.plot(RAf_wide,dec_cut(RAf_wide,0.4192,-50.14713),color='k')
fig.xaxis.set_ticks_position('both')
fig.yaxis.set_ticks_position('both')
fig.xaxis.set_tick_params(direction='in', which='both')
fig.yaxis.set_tick_params(direction='in', which='both')
plt.xlabel("RA", fontsize=14)
plt.ylabel("Dec", fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.show()

 
zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide7):
    phi = np.sqrt((RAf_wide7[k]-RAf_wide7[k+1:])**2+(DECf_wide7[k]-DECf_wide7[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide7[k+1:])/2).value * phi * 0.7)      
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide7[k+1:]).value)*0.7)  

kab_wide7 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide7 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                     
         kab_wide7 = np.append(kab_wide7, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide7 = np.append(err_wide7, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                         
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide7, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide7))])                                   
ax.errorbar(binp, kab_wide7, yerr=err_wide7, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show() 


#cross point of lines, vertices for a future polygon
m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x1 = (b1-b2) / (m2-m1)
y1 = m1 * x1 + b1
print('intersecting point of lines',x1,y1)

m1, b1 = 0.4192, -50.14713 # slope & intercept (bottom horizontal line)
m2, b2 = -2.664, 113.82 # slope & intercept (middle1 vertical line)
x2 = (b1-b2) / (m2-m1)
y2 = m1 * x2 + b1
print('intersecting point of lines',x2,y2)

m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x3 = (b1-b2) / (m2-m1)
y3 = m1 * x3 + b1
print('intersecting point of lines',x3,y3)

m1, b1 = 0.4192, -50.14713 # slope & intercept (bottom horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x4 = (b1-b2) / (m2-m1)
y4 = m1 * x4 + b1
print('intersecting point of lines',x4,y4)


#Create a Polygon with the intersecting points
coords = [(x1, y1), (x2, y2), (x4, y4), (x3, y3)] #careful bc the order of the vertices matters
poly = Polygon(coords)
RAf_wide8=np.array([])
DECf_wide8=np.array([])
Zf_wide8=np.array([])
for i, item in enumerate(RAf_wide):
    if Point(item,DECf_wide[i]).within(poly)==False:  
        RAf_wide8=np.append(RAf_wide8,item)
        DECf_wide8=np.append(DECf_wide8,DECf_wide[i])
        Zf_wide8=np.append(Zf_wide8,Zf_wide[i])


 
zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide8):
    phi = np.sqrt((RAf_wide8[k]-RAf_wide8[k+1:])**2+(DECf_wide8[k]-DECf_wide8[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide8[k+1:])/2).value * phi * 0.7)    
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide8[k+1:]).value)*0.7)  

kab_wide8 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide8 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                   
         kab_wide8 = np.append(kab_wide8, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide8 = np.append(err_wide8, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                       
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide8, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide8))])                                   
ax.errorbar(binp, kab_wide8, yerr=err_wide8, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show()



#cross point of lines, vertices for a future polygon
m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x1 = (b1-b2) / (m2-m1)
y1 = m1 * x1 + b1
print('intersecting point of lines',x1,y1)

m1, b1 = 0.4192, -50.00213 # slope & intercept (top horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x2 = (b1-b2) / (m2-m1)
y2 = m1 * x2 + b1
print('intersecting point of lines',x2,y2)

m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.95-0.47  # slope & intercept (right vertical line)
x3 = (b1-b2) / (m2-m1)
y3 = m1 * x3 + b1
print('intersecting point of lines',x3,y3)

m1, b1 = 0.4192, -50.00213 # slope & intercept (top horizontal line)
m2, b2 = -2.664, 113.95-0.47  # slope & intercept (right vertical line)
x4 = (b1-b2) / (m2-m1)
y4 = m1 * x4 + b1
print('intersecting point of lines',x4,y4)


#Create a Polygon with the intersecting points
coords = [(x1, y1), (x2, y2), (x4, y4), (x3, y3)] #careful bc the order of the vertices matters
poly = Polygon(coords)
RAf_wide9=np.array([])
DECf_wide9=np.array([])
Zf_wide9=np.array([])
for i, item in enumerate(RAf_wide):
    if Point(item,DECf_wide[i]).within(poly)==False:  
        RAf_wide9=np.append(RAf_wide9,item)
        DECf_wide9=np.append(DECf_wide9,DECf_wide[i])
        Zf_wide9=np.append(Zf_wide9,Zf_wide[i])


zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide9):
    phi = np.sqrt((RAf_wide9[k]-RAf_wide9[k+1:])**2+(DECf_wide9[k]-DECf_wide9[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide9[k+1:])/2).value * phi * 0.7)  
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide9[k+1:]).value)*0.7)  

kab_wide9 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide9 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                    
         kab_wide9 = np.append(kab_wide9, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide9 = np.append(err_wide9, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                     
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide9, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide9))])                                   
ax.errorbar(binp, kab_wide9, yerr=err_wide9, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show()


#cross point of lines, vertices for a future polygon
m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x1 = (b1-b2) / (m2-m1)
y1 = m1 * x1 + b1
print('intersecting point of lines',x1,y1)

m1, b1 = 0.4192, -50.07713 # slope & intercept (middle horizontal line)
m2, b2 = -2.664, 113.95-0.47  # slope & intercept (right vertical line)
x2 = (b1-b2) / (m2-m1)
y2 = m1 * x2 + b1
print('intersecting point of lines',x2,y2)

m1, b1 = 0.4192, -50.14713 # slope & intercept (bottom horizontal line)
m2, b2 = -2.664, 113.95-0.3 # slope & intercept (middle2 vertical line)
x3 = (b1-b2) / (m2-m1)
y3 = m1 * x3 + b1
print('intersecting point of lines',x3,y3)

m1, b1 = 0.4192, -50.14713 # slope & intercept (bottom horizontal line)
m2, b2 = -2.664, 113.95-0.47  # slope & intercept (right vertical line)
x4 = (b1-b2) / (m2-m1)
y4 = m1 * x4 + b1
print('intersecting point of lines',x4,y4)
 


#Create a Polygon with the intersecting points
coords = [(x1, y1), (x2, y2), (x4, y4), (x3, y3)] #careful bc the order of the vertices matters
poly = Polygon(coords)
RAf_wide10=np.array([])
DECf_wide10=np.array([])
Zf_wide10=np.array([])
for i, item in enumerate(RAf_wide):
    if Point(item,DECf_wide[i]).within(poly)==False:  
        RAf_wide10=np.append(RAf_wide10,item)
        DECf_wide10=np.append(DECf_wide10,DECf_wide[i])
        Zf_wide10=np.append(Zf_wide10,Zf_wide[i])

 
zij = np.array([]) 
rij = np.array([])
phi = np.array([])
for k, zk in enumerate(Zf_wide10):
    phi = np.sqrt((RAf_wide10[k]-RAf_wide10[k+1:])**2+(DECf_wide10[k]-DECf_wide10[k+1:])**2)*math.pi/180  
    rij= np.append(rij, cosmo.comoving_distance((zk + Zf_wide10[k+1:])/2).value * phi * 0.7)      
    zij = np.append(zij, abs(cosmo.comoving_distance(zk).value-cosmo.comoving_distance(Zf_wide10[k+1:]).value)*0.7)  

kab_wide10 = np.array([])                                                                        
bins=np.array([0.155,0.17,0.42,0.595,1.09,1.79,3.5,6.,11,20,35])
err_wide10 = np.array([])                                                                      
binp = np.array([])                                                                     
for k, bini in enumerate(bins):                                                         
     if k < len(bins)-1:                                                         
         idxtrans = (rij >= bini) & (rij < (bini+bins[k+1]))                                     
         idxlos1 = (zij > 0) & (zij < 7)                                                    
         idxlos2 = (zij > 0) & (zij < 45)                                                    
         kab_wide10 = np.append(kab_wide10, sum(idxtrans & idxlos1)/sum(idxtrans & idxlos2))              
         err_wide10 = np.append(err_wide10, math.sqrt(sum(idxtrans & idxlos1))/sum(idxtrans & idxlos2))   
         binp = np.append(binp, bini + (bins[k+1]-bini)/2)                                                                                                                                                                                     
ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab_wide10, s=10, c = 'b', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab_wide10))])                                   
ax.errorbar(binp, kab_wide10, yerr=err_wide10, xerr=None, c = 'b', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
plt.show()


#lets plot the clustering of those zones
ax = plt.figure().add_subplot(111)   
ax.scatter(binp, kab_wide3, s=10, c = 'g', marker='o',label='zone3') 
ax.scatter(binp, kab_wide4, s=10, c = 'orange', marker='o',label='zone4') 
ax.scatter(binp, kab_wide5, s=10, c = 'pink', marker='o',label='zone5') 
ax.scatter(binp, kab_wide6, s=10, c = 'violet', marker='o',label='zone6') 
ax.scatter(binp, kab_wide7, s=10, c = 'gray', marker='o',label='zone7') 
ax.scatter(binp, kab_wide8, s=10, c = 'lightgreen', marker='o',label='zone8') 
ax.scatter(binp, kab_wide9, s=10, c = 'lightblue', marker='o',label='zone9') 
ax.scatter(binp, kab_wide10, s=10, c = 'brown', marker='o',label='zone10') 
horiz_line = np.array([7/45 for m in range(len(kab_wide10))])                                   
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
plt.legend(bbox_to_anchor=(-0.1,.9,1,0.2),ncol=4, prop={'size': 8.7}).get_frame().set_edgecolor('black')
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False) 
#plt.savefig('clustering in eight jacknife zones',dpi=500)
plt.show()


#compute covariance matrix
covariance=np.cov(np.stack((kab_wide3,kab_wide4,kab_wide5,kab_wide6,kab_wide7,kab_wide8,kab_wide9,kab_wide10), axis=1))*(8-1)*(8-1)/8

#the diagonal of the covariance matrix are the variance (std deviation**2)
diagonal=np.diagonal(covariance)
std_deviation=np.sqrt(diagonal)


#plot the covariance matrix with imshow
ax=plt.figure().add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_tick_params(direction='in', which='both')
ax.yaxis.set_tick_params(direction='in', which='both')
plt.imshow(covariance,cmap=plt.cm.Greys, origin='lower', aspect="auto")
colorbar=plt.colorbar()
colorbar.ax.tick_params( direction='in')
plt.xlabel('$i$', fontsize=14)
plt.ylabel('$j$', fontsize=14)
#plt.savefig('covariance matrix',dpi=500)
plt.show()

#let's plot the normalized covariance matrix for a better visibility
normalized=np.array([])
for i, item in enumerate(covariance):
    for j, itemj in enumerate(item):
        normalized=np.append(normalized,itemj/np.sqrt(covariance[i][i]*covariance[j][j]))
norm_covariance=normalized.reshape((len(binp),len(binp)))        
        
ax=plt.figure().add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_tick_params(direction='in', which='both')
ax.yaxis.set_tick_params(direction='in', which='both')
plt.imshow(norm_covariance,cmap=plt.cm.Greys, origin='lower', aspect="auto")
colorbar=plt.colorbar()
colorbar.ax.tick_params( direction='in')
plt.xlabel(r'$i$', fontsize=14)
plt.ylabel(r'$j$', fontsize=14)
plt.clim(-0.56, 1)
#plt.savefig('normalized covariance matrix',dpi=500)
plt.show()


#for a given j, plot V_{ij}/sqrt(V_{ii}V_{jj}) as a function of i.
colors=['b','r','g','k','orange','pink','brown','lightblue','lightgreen','darkgreen']
for i, item in enumerate(covariance):
    normalized=np.array([])
    for j, itemj in enumerate(item):        
        normalized=np.append(normalized,itemj/np.sqrt(covariance[i][i]*covariance[j][j]))
    plt.plot(np.arange(0,10,1),normalized,color=colors[i],label='i='+str(i))
plt.xlabel(r'$j$', fontsize=14)
plt.ylabel(r'$\frac{V_{ij}}{\sqrt{V_{ii}V_{jj}}}$', fontsize=14)
plt.ylim(-0.65,1.05)
plt.legend(bbox_to_anchor=(.2,.7,1,0.2),ncol=1, prop={'size': 11}).get_frame().set_edgecolor('black')
plt.tight_layout()
#plt.savefig('normalized full covariance as a function of j',dpi=500)
plt.show()



#clustering measurements and scales of data sample
kab45=np.array([0.37815126, 0.33185841, 0.28832117, 0.2457265,  0.2112742,  0.19542942,
 0.18455267, 0.17503774, 0.16128381, 0.16357504 ]) 

ax = plt.figure().add_subplot(111)                                                      
ax.scatter(binp, kab45, s=10, c = 'blue', marker='o')                                     
horiz_line = np.array([7/45 for m in range(len(kab45))])                                   
ax.errorbar(binp, kab45, yerr=std_deviation, xerr=None, c = 'blue', ls='None', capsize=2, elinewidth=1)
ax.plot(binp, horiz_line, 'k-', linewidth = 1)                  
plt.xlabel(r'$R_{ij}$ [$h^{-1}$Mpc]', fontsize=14)              
plt.ylabel(r'$K^{0,7}_{7,45}$', fontsize=14)                   
ax.xaxis.set_ticks_position('both')                      
ax.yaxis.set_ticks_position('both')                                                     
ax.xaxis.set_tick_params(direction='in', which='both')                                  
ax.yaxis.set_tick_params(direction='in', which='both')                                  
plt.tick_params(labelsize = 'large')                                                    
ax.set_xscale('log') 
for axis in [ax.xaxis, ax.yaxis]:
     axis.set_major_formatter(ScalarFormatter())
plt.tight_layout()                                                                      
plt.grid(False)  
#plt.savefig('K-estimator with errors from covariance matrix',dpi=500)                                      
plt.show()






  