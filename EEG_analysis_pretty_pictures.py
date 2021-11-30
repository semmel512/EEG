import pandas as pd
from pylab import *
from scipy import stats
from sklearn import model_selection, tree, ensemble, svm, preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
import seaborn as sns
import random

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

def find_deriv(x, filt_window, num_filt):
	for i in range(num_filt):
		x = move_avg(x, filt_window)
	dx = zeros(len(x)-2)
	for i in range(len(dx)):
		dx[i] = (x[i+2]-x[i])/2
	return dx

def width_height_ratio(x):
    m = max(x)
    arg = argmax(x)
    s = where(x[0:arg]<m/2)[0]
    s = append(s, 0)
    a = max(s)
    u = where(x[arg:]<m/2)[0] + arg
    u = append(u, 500)
    b = min(u)
    return (b-a)/m


def find_Q(xf): #use psd
    xf_max = max(xf)
    for i in range(len(xf)):
        if xf[i] == xf_max:
            xf_max_index = i
            break
    counter = 0
    while xf[xf_max_index + counter] > 0.5*xf_max and (xf_max_index + counter) < len(xf)-1:
        counter += 1
    right_index = xf_max_index + counter
    counter = 0
    while xf[xf_max_index - counter] > 0.5*xf_max and (xf_max_index - counter) > 0:
        counter += 1
    left_index = xf_max_index - counter
    return (right_index - left_index)


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
y_all = np.concatenate( (1*ones(num_samples, dtype=int), 2*ones(num_samples, dtype=int)), axis=0  ) #targets: 1=auditory, 2=visual
GNB_Scores, RF_Scores, SVM_Scores = np.zeros(256), np.zeros(256), np.zeros(256)

num_features = 8
avg_a_ft = np.zeros(125)
avg_v_ft = np.zeros(125)
for chan in range(136, 137):
	if chan%10 == 0: #just printing things to keep track of progress
		print(chan)
	X_all = np.zeros( (2*num_samples, num_features) )
	for i in range(num_samples):
		signal_a = Xa[i*1000 + 500: i*1000 + 750, chan]
		signal_v = Xv[i*1000 + 500: i*1000 + 750, chan]
		signal_a_ft = np.abs(fft(signal_a)[:int(len(signal_a)/2)])
		signal_v_ft = np.abs(fft(signal_v)[:int(len(signal_v)/2)])
		avg_a_ft += signal_a_ft / num_samples
		avg_v_ft += signal_v_ft / num_samples
		#signal_a_slope = find_deriv(signal_a, 31, 2)
		#signal_v_slope = find_deriv(signal_v, 31, 2)
		X_all[i, 0]             = count_peaks(signal_a, 51, 2, x_threshold=0.0)
		X_all[i+num_samples, 0] = count_peaks(signal_v, 51, 2, x_threshold=0.0)
		X_all[i, 1]             = signal_a.std()
		X_all[i+num_samples, 1] = signal_v.std()
		X_all[i, 2]             = stats.kurtosis(signal_a)
		X_all[i+num_samples, 2] = stats.kurtosis(signal_v)
		X_all[i, 3]             = signal_a.max()
		X_all[i+num_samples, 3] = signal_v.max()
		X_all[i, 4]             = signal_a_ft[1]
		X_all[i+num_samples, 4] = signal_v_ft[1]
		X_all[i, 5]             = signal_a_ft[2]
		X_all[i+num_samples, 5] = signal_v_ft[2]
		X_all[i, 6]             = signal_a_ft[75:].mean()
		X_all[i+num_samples, 6] = signal_v_ft[75:].mean()
		X_all[i, 7]             = signal_a_ft[10]
		X_all[i+num_samples, 7] = signal_v_ft[10]
	#X_all = np.delete(X_all, [0, 1, 2, 3, 5, 7, 8, 9], axis=1) #taking out featuresX_all = preprocessing.scale(X_all) #Scaling all of the data
	#X_all = np.delete(X_all, [0, 2, 3, 5, 6, 7, 8, 9], axis=1)
	#X_all = np.delete(X_all, [0, 3, 4, 5, 6, 7, 8, 9], axis=1)
	#X_all = preprocessing.scale(X_all)  # Scaling all of the data
	X_train, X_val, y_train, y_val = model_selection.train_test_split(X_all, y_all, test_size=0.1, random_state=30) #10%
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.2/0.9, random_state=30) #20%
	#X_train, X_test, X_val = preprocessing.scale(X_train), preprocessing.scale(X_test), preprocessing.scale(X_val)  #scaling data

	GNB = GaussianNB().fit(X_train, y_train)
	GNB_Scores[chan] = GNB.score(X_test, y_test)
	RF = ensemble.RandomForestClassifier(n_estimators=100, max_depth=20).fit(X_train, y_train) #max_depth=none
	RF_Scores[chan] = RF.score(X_test, y_test)
	SVM = svm.SVC(C=1.0, kernel='rbf', gamma='scale').fit(X_train, y_train)
	SVM_Scores[chan] = SVM.score(X_test, y_test)

print("mean GNB score =", GNB_Scores.mean())
print("mean RF score =",   RF_Scores.mean())
print("mean SVM score =", SVM_Scores.mean())


'''
plt.figure(figsize=(5,5))
h = 0.02
x_min, x_max = X_all[:, 0].min() - 0.2, X_all[:, 0].max() + 0.2
y_min, y_max = X_all[:, 1].min() - 0.2, X_all[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = SVM.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r, alpha=0.25)
plt.scatter(X_all[:num_samples, 0], X_all[:num_samples, 1], color='red', label = 'Auditory', s=1.5)
plt.scatter(X_all[num_samples:, 0], X_all[num_samples:, 1], color='blue', label = 'Visual', s=1.5)
plt.legend()
plt.xlabel('RMS')
plt.ylabel('Peak Frequency')
'''

for i in range(len(X_all[0])):
	plt.figure(figsize=(4,4))
	sns.distplot(X_all[:num_samples, i], hist = True, kde = False, kde_kws = {'shade': True, 'linewidth': 3}, label = 'Auditory', color='red')
	sns.distplot(X_all[num_samples:, i], hist = True, kde = False, kde_kws = {'shade': True, 'linewidth': 3}, label = 'Visual', color='blue')
	plt.xlabel(str(i))
	plt.ylabel('Probability Density')
	plt.tight_layout()


'''
plt.figure()
plt.hist(GNB_Scores, alpha=0.5, label='Naive Bayes', color='blue')
plt.hist(RF_Scores, alpha=0.5, label='Random Forrest', color='green')
plt.hist(SVM_Scores, alpha=0.5, label='SVM', color='red')
plt.legend()
plt.xlabel("Accuracy Score")
plt.ylabel("Number of Channels")
plt.show()
'''


'''
plt.figure(figsize=(5, 5))
plt.plot(linspace(0, 125, 125), avg_a_ft, "o-", color='red', label='Auditory')
plt.plot(linspace(0, 125, 125), avg_v_ft, "o-", color='blue', label='Visual')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Voltage ($\mu$V)')
plt.xlim(80, 128)
plt.ylim(0, 45)
'''

'''
SVM_Scores = array([0.67912773, 0.77570093, 0.73831776, 0.73208723, 0.72897196,
       0.67912773, 0.66666667, 0.62616822, 0.63239875, 0.75389408,
       0.75700935, 0.7258567 , 0.73208723, 0.71028037, 0.68535826,
       0.67912773, 0.62616822, 0.78193146, 0.73208723, 0.74454829,
       0.76012461, 0.76012461, 0.71339564, 0.69158879, 0.77570093,
       0.75077882, 0.75077882, 0.74766355, 0.73208723, 0.69781931,
       0.77570093, 0.77570093, 0.78193146, 0.76323988, 0.74766355,
       0.75077882, 0.78816199, 0.79439252, 0.77258567, 0.75389408,
       0.74143302, 0.71028037, 0.64174455, 0.57009346, 0.57943925,
       0.78816199, 0.7694704 , 0.75077882, 0.76012461, 0.70404984,
       0.71028037, 0.59190031, 0.56386293, 0.80373832, 0.67601246,
       0.78193146, 0.7165109 , 0.63551402, 0.65732087, 0.70404984,
       0.67912773, 0.72897196, 0.71962617, 0.64797508, 0.62616822,
       0.59501558, 0.71028037, 0.61370717, 0.63551402, 0.65109034,
       0.62928349, 0.61993769, 0.74766355, 0.69470405, 0.79439252,
       0.70404984, 0.61682243, 0.57632399, 0.62305296, 0.65109034,
       0.62305296, 0.74766355, 0.73208723, 0.77570093, 0.70716511,
       0.57320872, 0.63239875, 0.61993769, 0.61993769, 0.56697819,
       0.68847352, 0.69158879, 0.75389408, 0.73208723, 0.67912773,
       0.70093458, 0.6728972 , 0.67601246, 0.64485981, 0.60124611,
       0.63239875, 0.71962617, 0.72274143, 0.71028037, 0.72897196,
       0.7258567 , 0.69781931, 0.71028037, 0.68535826, 0.63862928,
       0.74454829, 0.76012461, 0.72897196, 0.7258567 , 0.72274143,
       0.72274143, 0.69781931, 0.69781931, 0.71028037, 0.73831776,
       0.71962617, 0.69470405, 0.73831776, 0.74766355, 0.73831776,
       0.75077882, 0.75389408, 0.72897196, 0.63862928, 0.57632399,
       0.57009346, 0.60124611, 0.71339564, 0.69781931, 0.74766355,
       0.76635514, 0.79439252, 0.74143302, 0.7694704 , 0.76635514,
       0.75700935, 0.6728972 , 0.60124611, 0.6105919 , 0.73831776,
       0.76635514, 0.75077882, 0.80373832, 0.79750779, 0.79750779,
       0.81308411, 0.81308411, 0.73520249, 0.63551402, 0.5482866 ,
       0.78193146, 0.80996885, 0.81308411, 0.79750779, 0.79750779,
       0.79750779, 0.78193146, 0.70093458, 0.60436137, 0.79750779,
       0.80062305, 0.81308411, 0.79127726, 0.7788162 , 0.7694704 ,
       0.7694704 , 0.6728972 , 0.61682243, 0.80062305, 0.79439252,
       0.79750779, 0.80373832, 0.81308411, 0.70404984, 0.69781931,
       0.65732087, 0.58566978, 0.52024922, 0.57320872, 0.58255452,
       0.59501558, 0.80062305, 0.82554517, 0.85046729, 0.80685358,
       0.72897196, 0.69158879, 0.6635514 , 0.59501558, 0.56386293,
       0.63862928, 0.61993769, 0.58566978, 0.81931464, 0.79750779,
       0.78193146, 0.63239875, 0.57943925, 0.6635514 , 0.68847352,
       0.6635514 , 0.65109034, 0.80685358, 0.80062305, 0.73831776,
       0.59813084, 0.74766355, 0.71962617, 0.70404984, 0.70716511,
       0.79127726, 0.7694704 , 0.71028037, 0.67912773, 0.61370717,
       0.70716511, 0.74454829, 0.73520249, 0.73831776, 0.63239875,
       0.72897196, 0.70404984, 0.71962617, 0.71339564, 0.71339564,
       0.69470405, 0.7258567 , 0.74766355, 0.75077882, 0.74454829,
       0.7258567 , 0.7694704 , 0.79127726, 0.75389408, 0.73831776,
       0.7788162 , 0.80373832, 0.78504673, 0.77570093, 0.7258567 ,
       0.78193146, 0.7788162 , 0.76323988, 0.74454829, 0.78816199,
       0.72274143, 0.70404984, 0.74766355, 0.72897196, 0.72897196,
       0.69158879])
'''


'''
plt.figure(figsize=(10, 5))
t = int(random.random()*801)
t = 543
segment = Xv[t*1000+500:t*1000+750, 136]
chunks = np.array([  segment[i*10:(i+1)*10].mean() for i in range(25) ])
plt.plot(linspace(2, 3, 250), segment, color='blue')
plt.plot(linspace(2.02, 3.02, 25), chunks, "o", color='red')
plt.xlabel('Time (sec)')
plt.ylabel(r'Voltage ($\mu$V)')
plt.xlim(2, 3)
'''


'''
plt.figure(figsize=(12, 6))
counter = 0
Lbl = 'Visual'
for i in range(4):
	t = int(random.random()*801)
	plt.plot(linspace(0, 4, 1000), Xv[t*1000:(t+1)*1000, 136] + counter, color='blue', label=Lbl)
	Lbl = "_nolegend_"
	counter += 45
Lbl = 'Auditory'
for i in range(4):
	t = int(random.random()*801)
	plt.plot(linspace(0, 4, 1000), Xa[t*1000:(t+1)*1000, 136] + counter, color='red', label=Lbl)
	counter += 45
	Lbl = "_nolegend_"
plt.axvline(x=2.0, color='black', ls='dashed', linewidth=2)
plt.axvline(x=3.0, color='black', ls='dashed', linewidth=2)
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel(r'Voltage ($\mu$V)')
plt.ylim(-50, 400)
plt.xlim(1.5, 3.5)
'''

'''
xa_avg = avg_trials(Xa[:, 36], num_points=1000)
xv_avg = avg_trials(Xv[:, 36], num_points=1000)
plt.figure(figsize=(8, 4))
plt.plot(linspace(0, 4, 1000), xa_avg + 40, color='red', label='Auditory')
plt.plot(linspace(0, 4, 1000), xv_avg, color='blue', label='Visual')
plt.axvline(x=2.0, color='black', ls='dashed', linewidth=2)
plt.axvline(x=3.0, color='black', ls='dashed', linewidth=2)
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel(r'Voltage ($\mu$V)')
plt.ylim(-20, 60)
'''




'''
for i in range(len(X_all[0])):
	plt.figure()
	plt.hist(X_all[:num_samples, i], histtype='step', density='true', bins=20, alpha=1.0, color='red', label='Auditory')
	plt.hist(X_all[num_samples:, i], histtype='step', density='true', bins=20, alpha=1.0, color='blue', label='Visual')
	#plt.hist(X_all[:num_samples, i], density='true', bins=20, alpha=0.5, color='red', label='Auditory')
	#plt.hist(X_all[num_samples:, i], density='true', bins=20, alpha=0.5, color='blue', label='Visual')
	plt.legend()
	plt.xlabel(str(i))
'''


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

'''
plt.figure()
for i in range(10):
	xa = move_avg(Xa[i*1000 + 500 : i*1000 + 750, Channel], 51)
	xv = move_avg(Xv[i*1000 + 500 : i*1000 + 750, Channel], 51)
	plt.plot(xa, color='red', linewidth=0.2)
	plt.plot(xv, color='blue', linewidth=0.2)
'''






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






