# GBDT_KgluSite
In this paper, a new lysine glutarylation(Kglu) site prediction model GBDT_Kglu was proposed， which adopted seven feature encoding methods  to convert protein sequences into digital information, including BE, BLOSUM62, EAAC, CTDC, PSSM, CKSAAP, and Secondary Structural information. Then, the NearMiss-3 method dealed with the imbalanced data set issue ,and Elastics Net was used to filter redundant information in the features. Finally, the prediction model for identify Kglu site  based on GBDT was established
#Requirement
Backend = Tensorflow(1.14.0)
keras(2.3.1)
Numpy(1.20.2)
scikit-learn(1.0.2)
pandas(1.3.5)
matplotlib(3.5.2)
#Dateset
The data uploaded in DataSet is the original data before dividing the dataset, with 707 positive samples and 4369 negative samples, all with a sample length of 33, where X stands for virtual amino acids.
#Model
GBDT_ KgluSite.py can be directly used to predict glutarylation modification sites
#Contact
Feel free to contact us if you nedd any help: flyinsky6@gmail.com
