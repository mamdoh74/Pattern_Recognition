import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (LinearRegression,Ridge)

np.random.seed(1234)
def create_dummy_data(func,sample_size,std):
    x=np.linspace(0,1,sample_size)
    t=func(x)+np.random.normal(scale=std,size=x.shape)
    return x,t

def data_pattern(x):
    return np.sin(2*np.pi*x)

X_train , y_train =create_dummy_data(data_pattern,200,0.25)
X_train=X_train.reshape(len(X_train),1)
y_train=y_train.reshape(len(y_train),1)
x_test=np.linspace(0,1,100)
x_test=x_test.reshape(len(x_test),1)
y_test=data_pattern(x_test)
y_test=y_test.reshape(len(y_test),1)

plt.scatter(X_train,y_train,facecolor='none',edgecolor='b',s=50,label="training data")
plt.plot(x_test,y_test,c='green',label='sin')
plt.legend()
plt.show()

poly=PolynomialFeatures(degree=2)
poly2=PolynomialFeatures(degree=3)
poly3=PolynomialFeatures(degree=4)

X_train2=poly.fit_transform(X_train)
x_test1=poly.fit_transform(x_test)

X_train3=poly2.fit_transform(X_train)
x_test3=poly2.fit_transform(x_test)

X_train4=poly3.fit_transform(X_train)
x_test4=poly3.fit_transform(x_test)

print('x train is : ',X_train[:5])
print('Polynominal x train with degree 2 is : ',X_train2[:5])
print('x train shape is : ',X_train.shape)
print('Polynominal x train with degree 2 shape is : ',X_train2.shape)
print('='*50)

print('x train is : ',X_train[:5])
print('Polynominal x train with degree 3 is : ',X_train3[:5])
print('x train shape is : ',X_train.shape)
print('Polynominal x trainwith degree 3 shape is : ',X_train3.shape)
print('='*50)

print('x train is : ',X_train[:5])
print('Polynominal x train with degree 4 is : ',X_train4[:5])
print('x train shape is : ',X_train.shape)
print('Polynominal x train with degree 4 shape is : ',X_train4.shape)
print('='*50)

print(x_test.shape)

# making regression 
lrmodel=LinearRegression()
lrmodel.fit(X_train,y_train)
y_pred=lrmodel.predict(x_test)
print('score of linear rigression training of poly 1 is :' ,lrmodel.score(X_train,y_train))
print('score of linear rigression testing of poly 1 is :' ,lrmodel.score(x_test,y_test))

lrmodel2=LinearRegression()
lrmodel2.fit(X_train2,y_train)
y_pred2=lrmodel2.predict(x_test1)
print('score of linear rigression training of poly 2 is :' ,lrmodel2.score(X_train2,y_train))
print('score of linear rigression testing of poly 2 is :' ,lrmodel2.score(x_test1,y_test))

lrmodel3=LinearRegression()
lrmodel3.fit(X_train3,y_train)
y_pred3=lrmodel3.predict(x_test3)
print('score of linear rigression training of poly 3 is :' ,lrmodel3.score(X_train3,y_train))
print('score of linear rigression testing of poly 3 is :' ,lrmodel3.score(x_test3,y_test))

lrmodel4=LinearRegression()
lrmodel4.fit(X_train4,y_train)
y_pred4=lrmodel4.predict(x_test4)
print('score of linear rigression training of poly 4 is :' ,lrmodel4.score(X_train4,y_train))
print('score of linear rigression testing of poly 4 is :' ,lrmodel4.score(x_test4,y_test))

plt.subplot(221)
plt.scatter(X_train,y_train,facecolor='none',edgecolor='b',s=50,label="training data")
plt.plot(x_test,y_pred,c='green',label='sin')
#plt.legend()
#plt.show()

plt.subplot(222)
plt.scatter(X_train,y_train,facecolor='none',edgecolor='b',s=50,label="training data")
plt.plot(x_test1,y_pred2,c='green',label='sin')
#plt.legend()
#plt.show()

plt.subplot(223)
plt.scatter(X_train,y_train,facecolor='none',edgecolor='b',s=50,label="training data")
plt.plot(x_test3,y_pred3,c='green',label='sin')
#plt.legend()
#plt.show()

plt.subplot(224)
plt.scatter(X_train,y_train,facecolor='none',edgecolor='b',s=50,label="training data")
plt.plot(x_test4,y_pred4,c='green',label='sin')
#plt.legend()
plt.show()


print('='*50)

ridge=Ridge()
ridge.fit(X_train,y_train)
print('score of ridge training of poly 1 is :' ,lrmodel.score(X_train,y_train))
print('score of ridge testing of poly 1 is :' ,lrmodel.score(x_test,y_test))


ridge2=Ridge()
ridge2.fit(X_train2,y_train)
print('score of ridge training of poly 2 is :' ,ridge2.score(X_train2,y_train))
print('score of ridge testing of poly 2 is :' ,ridge2.score(x_test1,y_test))


ridge3=Ridge()
ridge3.fit(X_train3,y_train)
print('score of ridge training of poly 3 is :' ,ridge3.score(X_train3,y_train))
print('score of ridge testing of poly 3 is :' ,ridge3.score(x_test3,y_test))


ridge4=Ridge()
ridge4.fit(X_train4,y_train)
print('score of ridge training of poly 4 is :' ,ridge4.score(X_train4,y_train))
print('score of ridge testing of poly 4 is :' ,ridge4.score(x_test4,y_test))
