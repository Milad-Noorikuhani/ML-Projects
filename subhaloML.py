import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from Corrfunc.theory.xi import xi

## The code is for one of the four boxes shown in the PDF file, i.e. box 1. 
# Everything can just be repeated for the other three boxes.

# We load the halo catalog data below. They can be downloaded
# from https://www.cosmosim.org/cms/files/

dataa1 = np.loadtxt("")

# collecting subhalos with peak masses of at least 10^11 Msun/h:

subhalos11 = []
for i in range(len(dataa1)):
    if dataa1[i, 6] != -1 and dataa1[i, 60] >= 10**11:
        subhalos11.append(dataa1[i, 6])

# hosts is a matrix whose each row contains information about one host halo
# Each row of hosts contains: Mvir, Mpeak, Vmax, Vpeak, concentration, 
# a1/2 (half-mass scale), Bullock spin, x, y, z, vx, vy, vz, index, subhalo number count.

subhalo_counter11 = Counter(subhalos11)
hosts = []
subhalo_lookup = {dataa1[i, 1]: subhalo_counter11[dataa1[i, 1]]\
                  for i in range(len(dataa1))}


for i in range(len(dataa1)):
    if dataa1[i, 6] == -1:
        hosts.append([dataa1[i, 10], dataa1[i, 60], dataa1[i, 16], dataa1[i, 62], \
                      dataa1[i, 11]/dataa1[i, 12], dataa1[i, 63], dataa1[i, 45],\
                      *dataa1[i, 17:23], i, subhalo_lookup[dataa1[i, 1]]])

hosts = np.array(hosts)


# Train/test split by separating box1 (test set) from the remaining
# volume (training set). Then applying the regressors.

mask1 = np.all((0 <= hosts[:, 7:10]) & (hosts[:, 7:10] <= 200), axis=1)
box1 = hosts[mask1]
box1r = hosts[~mask1]

rfr60_b1 = RandomForestRegressor(n_estimators = 60)
rfr60_b1.fit(box1r[:,0:7], box1r[:,14])
yprd1 = rfr60_b1.predict(box1[:,0:7])

rgr1 = DecisionTreeRegressor(random_state = 5)
rgr1.fit(box1r[:,0:7], box1r[:,14])
yprd1g = rgr1.predict(box1[:,0:7])

yprd1t = yprd1/2 + yprd1g/2

# A function for finding the predicted and actual average number counts
# based on binned feature values

def average_halo_count(feature, yact, yprd, min_var, max_var, bin_size):
    var = min_var
    binned_var = []
    N_actual = []
    N_pred = []
    while var <= max_var - bin_size:
        indcs = (var < feature) & (feature <= var + bin_size)
        N_act = yact[indcs]
        N_prd = yprd[indcs]
        binned_var.append(var + bin_size/2)
        N_actual.append(np.average(N_act))
        N_pred.append(np.average(N_prd))
        var = var + bin_size  
    return(np.array(binned_var), np.array(N_actual), np.array(N_pred))

logm1, Nact_m1, Nprd_m1 = average_halo_count(np.log10(box1[:,0]),\
                                             box1[:,14], yprd1t, 11.55, 15.001, 0.3)

v1, Nact_v1, Nprd_v1 = average_halo_count(box1[:,2],\
                                          box1[:,14], yprd1t, 100, 1200, 100)


a1, Nact_a1, Nprd_a1 = average_halo_count(box1[:,5],\
                                          box1[:,14], yprd1t, 0.4, 0.99, 0.05)
                        


# For each subhalo, collecting the ID of its host,
# and its positions and velocities. Then separating those in box 1 from the rest.

allsubs = []
for i in range(len(dataa1)):
    if dataa1[i, 6] != -1 and dataa1[i, 60] >= 10**11:
        allsubs.append([dataa1[i, 6], *dataa1[i,17:23])])

allsubs = np.array(allsubs)

XVMsub1 = []
for i in range(len(allsubs)):
    if 0 <= allsubs[i,1] <= 200 and 0 <= allsubs[i,2] <= 200 and 0 <= allsubs[i,3] <= 200:
        XVMsub1.append([allsubs[i,1], allsubs[i,2], allsubs[i,3], allsubs[i,4],
                        allsubs[i,5], allsubs[i,6]])

XVMsub1 = np.array(XVMsub1)


# Now we build the "mapping features" explained in the summary PDF
# and apply the knn algorithm for identifying the "donor" hosts

box1hn = []

box1rhn = []

for i in range(len(box1)):
    if round(yprd1[i]) != 0:
        box1hn.append([box1[i,0], box1[i,5], round(yprd1t[i]), round(yprd1t[i]),\
                       round(yprd1t[i]), round(yprd1t[i]), box1[i,13]])


for i in range(len(box1r)):
    if box1r[i, 14] != 0:
        box1rhn.append([box1r[i, 0], box1r[i, 5], box1r[i, 14], box1r[i, 14], \
                        box1r[i, 14], box1r[i, 14], box1r[i,13]])



box1hn = np.array(box1hn)
box1rhn = np.array(box1rhn)
map_fit1n = KNeighborsRegressor(n_neighbors = 1)
map_scaler1n = StandardScaler()
map_fit1n.fit(map_scaler1n.fit_transform(box1rhn[:,0:6]), box1rhn[:,6])
mapper1n = map_fit1n.predict(map_scaler1n.transform(box1hn[:,0:6]))


# Now we map positions and velocities from donors to target hosts. 

box1fitxvm = []
for i in range(len(box1hn)):
    ihn = int(box1hn[i, 6])
    ih = int(mapper1n[i])
    box1fitxvm .append([dataa1[ih, 1], dataa1[ihn, 17] - dataa1[ih, 17],\
                        dataa1[ihn, 18] - dataa1[ih, 18],\
                        dataa1[ihn, 19] - dataa1[ih, 19],\
                        dataa1[ihn, 20] - dataa1[ih, 20],\
                        dataa1[ihn, 21] - dataa1[ih, 21],\
                        dataa1[ihn, 22] - dataa1[ih, 22]])         


box1fitxvm = np.array(box1fitxvm)
inds1 = box1fitxvm[:,0].argsort()
indsub = allsubs[:,0].argsort()
Xprdi1n = []
Yprdi1n = []
Zprdi1n = []
Vxprdi1n = []
Vyprdi1n = []

i = j = l= 0
while i < len(inds1) and j < len(indsub):
    if box1fitxvm[inds1[i], 0] == allsubs[indsub[j], 0]:
        Xprdi1n.append(box1fitxvm[inds1[i],1] + allsubs[indsub[j], 1])
        Yprdi1n.append(box1fitxvm[inds1[i],2] + allsubs[indsub[j], 2])
        Zprdi1n.append(box1fitxvm[inds1[i],3] + allsubs[indsub[j], 3])
        Vxprdi1n.append(box1fitxvm[inds1[i],4] + allsubs[indsub[j], 4])
        Vyprdi1n.append(box1fitxvm[inds1[i],5] + allsubs[indsub[j], 5])
        Vzprdi1n.append(box1fitxvm[inds1[i],6] + allsubs[indsub[j], 6])
        j += 1
        l += 1
    elif box1fitxvm[inds1[i], 0] < allsubs[indsub[j], 0] and i < len(inds1) - 1 \
         and box1fitxvm[inds1[i+1], 0] == box1fitxvm[inds1[i], 0]:                                                               
        i += 1
        j = j - l
        l = 0
    elif box1fitxvm[inds1[i], 0] < allsubs[indsub[j], 0]:
        i += 1 
        l = 0
    else:
        j += 1
        l = 0
    if j == len(indsub) and i < len(inds1) - 1 and\
       box1fitxvm[inds1[i+1], 0] == box1fitxvm[inds1[i], 0]:
        i += 1
        j = j -l 
        l = 0


XVM1s = []
for i in range(len(Xprdi1n)):
    if 0 <= Xprdi1n[i] <= 200 and 0 <= Yprdi1n[i] <= 200 and 0 <= Zprdi1n[i] <= 200:
        XVM1s.append([Xprdi1n[i], Yprdi1n[i], Zprdi1n[i], Vxprdi1n[i],\
                      Vyprdi1n[i], Vzprdi1n[i], Mprdi1n[i]])


# Here we identify bins and calculate the actual
# and predicted correlation functions

XVM1s = np.array(XVM1s)
bin_file = np.logspace(np.log10(0.2), np.log10(20), 20)
xi1_prd = xi(boxsize = 200, nthreads = 2, binfile = bin_file,\
             X = XVM1s[:,0], Y = XVM1s[:,1],\
             Z = XVM1s[:,2], output_ravg=True)
xi1_act = xi(boxsize = 200, nthreads = 2, binfile = bin_file,\
             X = XVMsub1[:,0], Y = XVMsub1[:,1],\
             Z = XVMsub1[:,2], output_ravg=True)
r1a = []
r1p = []
xi1a = []
xi1p = []
for i in range(len(xi1_act)):
    r1a.append(xi1_act[i][2])
    r1p.append(xi1_prd[i][2])
    xi1a.append(xi1_act[i][3])
    xi1p.append(xi1_prd[i][3])

r1a = np.array(r1a)
r1p = np.array(r1p)
xi1a = np.array(xi1a)
xi1p = np.array(xi1p)

# Plot for average subhalo number count vs binned logMvir with percent difference panel


fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]} )
ax1.scatter(logm1, Nact_m1, s=20, alpha = 1, color = "red", label = "actual")
ax1.scatter(logm1, Nprd_m1, s=20, alpha= 1,color = "blue",label = "prediction")
ax2.scatter(logm1, ((Nprd_m1 - Nact_m1)/Nact_m1)*100, s=20, alpha=1, color = "black")
ax1.set_ylabel("$ < N_{sub} >$", fontsize = 12.8)
ax2.set_xlabel("log$M_{vir}$", fontsize = 14)
ax2.set(ylabel = "% difference")
ax1.set_yscale('log')
ax1.text(13, 50, "Box 1", fontsize = 15)
ax1.legend(fontsize = 12.5)
fig1.tight_layout()


# Plot for average subhalo number count vs binned Vmax with percent difference panel

fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]} )
ax1.scatter(v1, Nact_v1, s=20, alpha=1, color = "red", label = "actual")
ax1.scatter(v1, Nprd_v1, s=20, alpha=1, color = "blue",label = "prediction")
ax2.scatter(v1, ((Nprd_v1 - Nact_v1)/Nact_v1)*100, s=20, alpha=1, color = "black")
ax1.set_ylabel("$ < N_{sub} > $", fontsize = 12.8)
ax1.text(180, 60, "Box 1", fontsize = 15)
ax2.set_xlabel("$V_{max}$", fontsize = 14)
ax2.set_ylabel("% difference")
ax1.set_yscale('log')
ax1.legend(loc = 'right', fontsize = 12.5)
fig1.tight_layout()

# Plot for average subhalo number count vs binned half-mass scale with 
# percent difference panel

fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]} )
ax1.scatter(a1, Nact_a1, s=20, alpha=1, color = "red", label = "actual")
ax1.scatter(a1, Nprd_a1, s=20, alpha=1, color = "Blue",label = "prediction")
ax2.scatter(a1, ((Nprd_a1 - Nact_a1)/Nact_a1)*100, s=20, alpha=1, color = "black")
ax1.set_ylabel("$ < N_{sub} > $", fontsize = 12.8)
ax1.text(0.83, 0.02, "Box 1", fontsize = 15)
ax2.set_xlabel("half mass scale", fontsize = 14)
ax2.set_ylabel("% difference")
ax1.legend(loc = 'lower center', fontsize = 12.5)
fig1.tight_layout()



# Plot for two-point correlation comparisons with percent difference panel

fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]} )
ax1.scatter(r1a, xi1a, s=20, alpha=1, color = "red", label = "actual")
ax1.scatter(r1p, xi1p, s=20, alpha=1, color = "Blue",label = "prediction")
ax2.scatter(r1a, ((xi1p - xi1a)/xi1a)*100, s=20, alpha=1, color = "black")
ax1.set_ylabel("$ \\xi$($r$)", fontsize = 12.8)
ax1.text(1, 554, "Box 1", fontsize = 14)
ax2.set_xlabel("$r$ (Mpc/h)", fontsize = 12)
ax2.set_ylabel("% difference")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax2.set_ylim((-8,7.5))
ax1.legend(fontsize = 12.5)
fig1.tight_layout()

