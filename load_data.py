import data_processing
import pickle as pickle

batches = data_processing.load_data()

print(len(batches))
with open('data_v4.pkl', 'wb') as f:
    pickle.dump(batches, f)

print("pid: " + batches[1][0])

print("\nno. visit: ", len(batches[1][2])) # no. visit
print("\nintervals: ", batches[1][1]) # no. days between current visit’s date to first visit’s date
print("\ncontext codes: ", batches[1][2]) # context code (diag) from patient’s 1st visit to second last visit
      
print("\none hot label of diag code of last visit: ", batches[1][3]) # ICD9 code to predict, translated from patients last visit
print("\nlabel of readmission: ", batches[1][4]) # readmission label to predict
     


