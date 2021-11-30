import pandas as pd
from pylab import *
from scipy import stats
from sklearn import model_selection, tree, ensemble, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

def avg_trials(x, num_points=1000): #returns the average of all trials for 1 channel.
	x_avg_seg = np.zeros(num_points)
	num_segments = int(len(x)/num_points)
	for i in range(num_segments):
		x_avg_seg += x[i*num_points : (i+1)*num_points]
	return x_avg_seg/num_segments

def move_avg(x, window):
    if window%2 != 1:
        print("window must be odd")
        window += 1
    filt_x = zeros(len(x), dtype=float)
    for i in range(int(window/2),  len(x) - int(window/2)):  # filter middle of data
        filt_x[i] = sum(x[i - int(window/2):i + int(window/2) + 1])/window
    for i in range(int(window/2)):  # filter ends of data
        filt_x[i] = sum(x[0:(i + int(window/2))]) / len(x[0:(i + int(window/2))])
        filt_x[len(x) - 1 - i] = sum(x[(len(x) - i - int(window/2)):len(x)]) / len(x[(len(x) - i - int(window/2)):len(x)])
    return filt_x

def count_peaks(x, filt_window, num_filt, x_threshold=0.0):
	for i in range(num_filt):
		x = move_avg(x, filt_window)
	num_peaks = 0.0
	for i in range(1, len(x)-1):
		if x[i-1] < x[i] and x[i] > x[i+1]:
			if x[i] > x_threshold:
				num_peaks += 1.0
	return num_peaks

def time2peak(x, filt_window, num_filt, x_threshold=0.0):
	for i in range(num_filt):
		x = move_avg(x, filt_window)
	samples_till_peak = 0
	for i in range(1, len(x)-1):
		if x[i-1] < x[i] and x[i] > x[i+1]:
			if x[i] > x_threshold:
				samples_till_peak = i
				break
	return samples_till_peak


#s01_Aa_c_all_data = pd.read_csv("s01_Aa_c.STIM.csv", header=0, delimiter =",", quoting=3).to_numpy()
#np.save(r'C:\Users\Justin\Desktop\EEG\s01_Aa_c.npy', s01_Aa_c_all_data)
#s01_Vv_c_all_data = pd.read_csv("s01_Vv_c.STIM.csv", header=0, delimiter =",", quoting=3).to_numpy()
#np.save(r'C:\Users\Justin\Desktop\EEG\s01_Vv_c.npy', s01_Vv_c_all_data)
#s01_Aa2_c_all_data = pd.read_csv("s01_Aa2_c.STIM.csv", header=0, delimiter =",", quoting=3).to_numpy()
#np.save(r'C:\Users\Justin\Desktop\EEG\s01_Aa2_c.npy', s01_Aa2_c_all_data)
#s01_Vv2_c_all_data = pd.read_csv("s01_Vv2_c.STIM.csv", header=0, delimiter =",", quoting=3).to_numpy()
#np.save(r'C:\Users\Justin\Desktop\EEG\s01_Vv2_c.npy', s01_Vv2_c_all_data)

#Combining Aa1 + Aa2 and Vv1 + Vv2
#Xa_all = np.concatenate(    (np.load(r'C:\Users\Justin\Desktop\EEG\s01_Aa_c.npy'), np.load(r'C:\Users\Justin\Desktop\EEG\s01_Aa2_c.npy')),  axis=0)
#np.save(r'C:\Users\Justin\Desktop\EEG\s01_Aa_c_all.npy', Xa_all)
#Xv_all = np.concatenate(    (np.load(r'C:\Users\Justin\Desktop\EEG\s01_Vv_c.npy'), np.load(r'C:\Users\Justin\Desktop\EEG\s01_Vv2_c.npy')),  axis=0)
#np.save(r'C:\Users\Justin\Desktop\EEG\s01_Vv_c_all.npy', Xv_all)



Xa = np.load(r'C:\Users\Justin\Desktop\EEG\s01_Aa_c_all.npy')
Xv = np.load(r'C:\Users\Justin\Desktop\EEG\s01_Vv_c_all.npy')
num_samples = int(min(len(Xa), len(Xv))/1000)
stim_start_window = 500   #the time point to start extracting feature information
feat_window = 10   #averaging window for each feature (10 is good)
x_pts =  25 #int(250/feat_window)     #number of features to make (20 or 25 is good)

Channel = 136     #Best indexes: 124, 136, 150, [10, 12, 13, 31, 36, 37, 46, 124, 135, 136, 148]
y_all = np.concatenate( (1*ones(num_samples, dtype=int), 2*ones(num_samples, dtype=int)), axis=0  ) #targets: 1=auditory, 2=visual

#Constructing features from averaged windows of the signal
X_all = np.zeros( (2*num_samples, x_pts) )
for i in range(num_samples):
	for j in range(x_pts):
		seg_start = i*1000 + stim_start_window + j*feat_window
		X_all[i, j]             = sum( Xa[seg_start : seg_start + feat_window, Channel ] )/feat_window
		X_all[i+num_samples, j] = sum( Xv[seg_start : seg_start + feat_window, Channel ] )/feat_window
#X_all = np.delete(X_all, [0,1,2,3,4,5], axis=1) #taking out features

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_all, y_all, test_size=0.1, random_state=30) #10%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2/0.9, random_state=30) #20%

SVM = svm.SVC(C=1.0, kernel='rbf', gamma='scale').fit(X_train, y_train)
print("Score =", SVM.score(X_test, y_test) )


PI = permutation_importance(SVM, X_test, y_test, n_repeats=10, random_state=44)
PI_avg = PI.importances_mean
PI_dev = PI.importances_std

feature_labels = linspace(0, len(PI_avg)-1, len(PI_avg))
plt.figure(figsize=(8, 4))
plt.plot(feature_labels, PI_avg, "o", color='black')
plt.errorbar(feature_labels, PI_avg, yerr=PI_dev, color='black', capsize=0, ls='none')
plt.xticks(feature_labels)
plt.xlabel('Feature Index')
plt.ylabel('Loss in Accuracy Score')





'''
Cs = linspace(1.0, 0.8, 5)
Gammas = linspace(10**(-4), 10**(-2), 50)
plt.figure()
for j in range(len(Cs)):
	print(Cs[j])
	Scores = zeros(len(Gammas))
	for i in range(len(Scores)):
		SVM = svm.SVC(C=Cs[j], kernel='rbf', gamma=Gammas[i]).fit(X_train, y_train)
		#print("Score =", SVM.score(X_test, y_test) )
		Scores[i] = SVM.score(X_test, y_test)
	plt.plot(Scores)
'''


plt.figure()
for i in range(10):
	xa = move_avg(Xa[i*1000 + 500 : i*1000 + 750, Channel], 51)
	xv = move_avg(Xv[i*1000 + 500 : i*1000 + 750, Channel], 51)
	plt.plot(xa, color='red', linewidth=0.2)
	plt.plot(xv, color='blue', linewidth=0.2)







'''
#Constructing features using signal properties, such as RMS, skewness, min...
window_start = 500
window_stop = 700
for chan in range(len(Channels)):
	if chan%10 == 0: #just printing things to keep track of progress
		print(Channels[chan])
	X_all = np.zeros( (2*num_samples, 6) )
	for i in range(num_samples):
		signal_a = Xa[i*1000 + window_start : i*1000 + window_stop, int(Channels[chan]) ]
		signal_v = Xv[i*1000 + window_start : i*1000 + window_stop, int(Channels[chan]) ]
		X_all[i, 0]             = signal_a.max()
		X_all[i+num_samples, 0] = signal_v.max()
		X_all[i, 1]             = signal_a.min()
		X_all[i+num_samples, 1] = signal_v.min()
		X_all[i, 2]             = signal_a.mean()
		X_all[i+num_samples, 2] = signal_v.mean()
		X_all[i, 3]             = signal_a.std()
		X_all[i+num_samples, 3] = signal_v.std()
		X_all[i, 4]             = stats.skew(signal_a)
		X_all[i+num_samples, 4] = stats.skew(signal_v)
		X_all[i, 5]             = stats.kurtosis(signal_a)
		X_all[i+num_samples, 5] = stats.kurtosis(signal_v)
	#X_all = np.delete(X_all, [0,1,2,3,4,5], axis=1) #taking out features
'''






'''
plt.figure()
plt.scatter(X_all[:num_samples, 0], X_all[:num_samples, 1], color='red', s=5)
plt.scatter(X_all[num_samples:, 0], X_all[num_samples:, 1], color='blue', s=5)
plt.figure()
plt.scatter(X_all[:num_samples, 0], X_all[:num_samples, 6], color='red', s=5)
plt.scatter(X_all[num_samples:, 0], X_all[num_samples:, 6], color='blue', s=5)

plt.figure()
for i in range(0, 20):
	plt.plot( X_all[i, :], color='red', linewidth=0.2)
	plt.plot( X_all[i+num_samples, :], color='blue', linewidth=0.2)

x_avg_a = np.zeros(x_pts)
x_avg_v = np.zeros(x_pts)
for i in range(num_samples):
	x_avg_a += X_all[i, :]/num_samples
	x_avg_v += X_all[i+num_samples, :]/num_samples
plt.figure()
plt.plot(x_avg_a, "o-", color='red')
plt.plot(x_avg_v, "o-", color='blue')

plt.figure()
plt.plot(avg_trials(Xa), color='red')
plt.plot(avg_trials(Xv) + 10, color='blue')
plt.axvline(x=500, color='black')
plt.xlabel('samples')
plt.ylabel(r'Voltage ($\mu$V)')
'''






