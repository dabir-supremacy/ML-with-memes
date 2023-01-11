import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import texthero as hero
import matplotlib.pyplot as plt
df=pd.read_csv('./FINAL/labels.csv', index_col=False, header=0)
df
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
df.dropna(inplace=True)
df['text_corrected']=hero.clean(df['text_corrected'])
df.head()
change={'very_positive': 1, 'positive':1,'negative':-1,'very_negative':-1,'neutral':0}
df["Labels"]=df["overall_sentiment"].map(change)
X=df.drop(["Labels","overall_sentiment","text_ocr","image_name"], axis=1)
X=list(X)
y=df["Labels"]
y-np.array(y)
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize,word_tokenize
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(df.text_corrected)
from sklearn.preprocessing import MaxAbsScaler
MAS=MaxAbsScaler()
new_scaled=MAS.fit_transform(X)
X=(new_scaled)
X_train, X_test, y_train, y_test =train_test_split(X, y)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predicty=dtree.predict(X_test)
F1score=f1_score(y_test,predicty,average='weighted')
print("Accuracy:",accuracy_score(y_test,predicty))
print("F1 SCORE:",F1score)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(dtree, X_test, y_test, cmap=plt.cm.Blues)


print(classification_report(y_test,predicty))
import pickle
pickle.dump(dtree, open('model1.pkl', 'wb'))
import sklearn.model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)
regr=LogisticRegression(multi_class='multinomial',random_state=1)
regr.fit(X_train,y_train)
print('Interccept: \n', regr.intercept_)
# The coefficients

print('Coefficients: \n', regr.coef_,'')
pred=regr.predict(X_test)
pred
print("F1 score: ",f1_score(y_test,pred,average='weighted'))
pickle.dump(regr,open('model2.pkl', 'wb'))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier(n_estimators=50,random_state=1,n_jobs=3)
clf.fit(X_train,y_train)
PRED=clf.predict(X_test)
print("F1 SCORE: ",f1_score(y_test,PRED,average='weighted'))
pickle.dump(clf,open('model3.pkl','wb'))
KNN=KNeighborsClassifier(n_neighbors=5,algorithm='brute')
KNN.fit(X_train,y_train)
predd=KNN.predict(X_test)
print("F1 SCORE: ",f1_score(y_test,predd,average='weighted'))
pickle.dump(KNN,open('model4.pkl','wb'))
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
sc = StandardScaler(with_mean=False)
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

prcptrnFS = Perceptron(eta0=1, random_state=152)
prcptrnFS.fit(X_train_std, y_train)
 
Y_predict_std = prcptrnFS.predict(X_test_std)
print("Misclassified examples %d" %(y_test != Y_predict_std).sum())
 
from sklearn.metrics import accuracy_score
print("Accuracy Score %0.3f" % accuracy_score(y_test, Y_predict_std))
print("F1 SCORE: ",f1_score(y_test,Y_predict_std, average='weighted'))

pickle.dump(prcptrnFS,open('model5.pkl','wb'))
from sklearn.ensemble import VotingClassifier
estimator=[]
estimator.append(('LR',LogisticRegression(solver='sag',multi_class='multinomial',max_iter=500)))
# estimator.append(('KNN',KNeighborsClassifier(n_neighbors=10)))
estimator.append(('DTC',DecisionTreeClassifier()))
estimator.append(('PRCPTRN',Perceptron(eta0=0.1, random_state=152)))

eclf1=VotingClassifier(estimators=estimator,voting='hard')
eclf1=eclf1.fit(X_train,y_train)
P=eclf1.predict(X_test)
print("F1 SCORE: ",f1_score(y_test,P,average='weighted')*100)
pickle.dump(eclf1,open('model6.pkl','wb'))
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.filters import prewitt_h,prewitt_v

images = df['image_name']
sentiments1 = df['Labels']
data_for_images_prewitt_v_features = []
sentiments1.reset_index(inplace=True, drop=True)
def find_data_for_images():
    img_size = 80
    count = 0
    j = 0
    temp=[]


    for i in images:
        try:
            image1 = imread("./FINAL/images/{}".format(i),as_gray=True)
            
            image_arr = resize(image1,(img_size,img_size))
            
            pre_ver = prewitt_v(image_arr)
            
            data_for_images_prewitt_v_features.append( [pre_ver , sentiments1[j]] )
            
            count+=1
            j+=1
            if(count==500):
                print(i)
                count=0
        except:
            temp.append(np.nan)
            print('NaN')

    
    imshow(pre_ver,cmap='gray')

    print(image1.shape)
    
find_data_for_images()
images_data=data_for_images_prewitt_v_features
X_data_images=[]
y_data_images=[]

for i in images_data:
    X_data_images.append(i[0].flatten())
    y_data_images.append(i[1])
df_X = pd.DataFrame(X_data_images)
df_y = pd.DataFrame(y_data_images)
X_train, X_test, y_train, y_test =train_test_split(df_X, df_y, random_state=21)
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)
from sklearn import tree
model_1 = tree.DecisionTreeClassifier(max_depth=20)
model_1 = model_1.fit(X_train, y_train)
pred_1=model_1.predict(X_test)
model_1.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,pred_1)*100)  
print("Report : \n", classification_report(y_test, pred_1))
print("F1 Score : ",f1_score(y_test, pred_1, average='weighted')*100)
pickle.dump(model_1,open('model_1.pkl','wb'))
from sklearn.neighbors import KNeighborsClassifier
model_2=KNeighborsClassifier(n_neighbors=8,weights='distance',algorithm='brute')
model_2.fit(X_train, y_train)
pred_2=model_2.predict(X_test)
model_2.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,pred_2)*100)  
print("Report : \n", classification_report(y_test, pred_2))
print("F1 Score : ",f1_score(y_test, pred_2, average='weighted')*100)
pickle.dump(model_2,open('model_2','wb'))
confusion_matrix(pred_1,y_test)
confusion_matrix(pred_2,y_test)
from sklearn.ensemble import RandomForestClassifier

model_3 = RandomForestClassifier(n_estimators=12, random_state=21)
model_3.fit(X_train, y_train)
pred_3=model_3.predict(X_test)
model_3.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,pred_3)*100)  
print("Report : \n", classification_report(y_test, pred_3))
print("F1 Score : ",f1_score(y_test, pred_3, average='weighted')*100)
pickle.dump(model_3,open('model_3.pkl','wb'))
confusion_matrix(pred_3,y_test)
X = df.drop(["Labels","overall_sentiment","text_ocr","image_name"], axis = 1)
print(X)
X=list(X)
y=df["Labels"]
print(X)
y=np.array(y)
print(y)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text_corrected)
#H=pd.DataFrame.sparse.from_spmatrix(X)
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
scaled_val=scaler.fit_transform(X)
scaled_val
X=(scaled_val)
#ndarray = x.toarray()
#g = ndarray.tolist()
X_train1, X_test1, y_train1, y_test1 =train_test_split(X, y, test_size=0.3,random_state=42)
from sklearn.svm import SVC
from sklearn import svm
model_4 = svm.SVC(kernel='linear', C = 5.0)
model_4.fit(X_train1, y_train1)
pred_4= model_4.predict(X_test1)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print("Accuracy : " , accuracy_score(y_test1,pred_4)*100)  
print("Report : \n", classification_report(y_test1, pred_4))
print("F1 Score : ",f1_score(y_test1, pred_4, average='weighted')*100)
confusion_matrix(pred_4,y_test1)
pickle.dump(model_4,open('model_4.pkl','wb'))
from sklearn.linear_model import SGDClassifier
model_5 = SGDClassifier()
model_5.fit(X_train1, y_train1)
pred_5= model_5.predict(X_test1)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print("Accuracy : " , accuracy_score(y_test1,pred_5)*100)  
print("Report : \n", classification_report(y_test1, pred_5))
print("F1 Score : ",f1_score(y_test1, pred_5, average='weighted')*100)
confusion_matrix(pred_5,y_test1)
pickle.dump(model_5,open('model_5.pkl','wb'))
from sklearn.naive_bayes import MultinomialNB
model_6 = MultinomialNB()
model_6.fit(X_train1, y_train1)
pred_6= model_6.predict(X_test1)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print("Accuracy : " , accuracy_score(y_test1,pred_6)*100)  
print("Report : \n", classification_report(y_test1, pred_6))
print("F1 Score : ",f1_score(y_test1, pred_6, average='weighted')*100)
confusion_matrix(pred_6,y_test1)
pickle.dump(model_6,open('model_6.pkl','wb'))
