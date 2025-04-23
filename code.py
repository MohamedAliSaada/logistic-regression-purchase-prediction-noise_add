import numpy as np    #study it's basics
import pandas as pd   #study it's basics
import matplotlib.pyplot as plt
from pprint import pprint  #it's only have pprint
import sympy as sym

# --- read the data 
df = pd.read_csv('logistic-regression-purchase-predictionn.csv')
X=np.array(df[['Age','Income']])
Y=np.array(df['Purchased']).reshape(-1,1)

# --- apply feature scalling method
X_n = (X - X.mean(axis=0)) / X.std(axis=0)

# --- expand polynomial features to capture non-linear relationships
# --- features: x1, x2, x1^2, x2^2, x1^3, x2^3, x1^4, x2^4, x1*x2
X_poly =np.c_[X,X**2,X**3,X**4,X[:,0]*X[:,1]]


# --- apply feature scaling after polynomial expansion
X__n = (X_poly - X_poly.mean(axis=0)) / X_poly.std(axis=0)


# --- hyperparameters
alpha=.01
m=X__n.shape[0]
epochs = 10000 #use patch gradient decent 

# --- parameters
b=0
w=np.zeros(X__n.shape[1]).reshape(-1,1)

# --- make cost functions
def cost_value(m, Y, y_pred):
    cost = -(1/m) * np.sum(Y * np.log(y_pred + 1e-15) + (1 - Y) * np.log(1 - y_pred + 1e-15))
    return cost

#loop to solve for w and b
cost_history=[]
for i in range(epochs):
    z=np.dot(X__n , w)+b
    y_pred = 1 / (1 + (np.exp(-z)))

    cost = cost_value(m,Y,y_pred)
    cost_history.append(cost)

    db = (1/m) * np.sum(y_pred - Y)
    dw = (1/m) * np.dot(X__n.T, (y_pred - Y))

    b = b - alpha * db
    w = w - alpha * dw
    

# --- plot to check convergence
plt.figure()
plt.plot(range(epochs),cost_history,color='k',label='evaluation of cost value over time')
plt.title('check convergence')
plt.xlabel("epochs")
plt.ylabel("cost value")
plt.legend()
plt.show()

print("----------------------------------------------------------")
print("----------------------------------------------------------")
#module accuracy 
def accuracy(y_pred,Y):
    values = (y_pred >= .5 ).astype(int)
    accuracy = (values == Y).mean()
    return f'{accuracy*100}% accuracy'
print(f'{accuracy(y_pred,Y)} @ and cost error {cost_history[-1]}')

print("----------------------------------------------------------")
x1,x2 = sym.symbols("x1 x2",real=True)
f=w[0,0]*x1+w[1,0]*x2+w[2,0]*x1**2+w[3,0]*x2**2+w[4,0]*x1**3+w[5,0]*x2**3+w[6,0]*x1**4+w[7,0]*x2**4+w[8,0]*x1*x2
x1_=np.linspace(X_n[:,0].min(),X_n[:,0].max(),500)

x2_=[]
for i in range(len(x1_)):
    k=x1_[i]
    f_n=f.subs([(x1,k)])
    x2__ = sym.solve(f_n,x2)
    x2_.append(x2__)
    
x2_=np.array(x2_)
print("----------------------------------------------------------")

#solve for decsion boundry

# --- plot the Data that scale featured
plt.figure()
plt.scatter(X_n[Y[:,0]==1][:,0],X_n[Y[:,0]==1][:,1],color='r',marker='+',label='Purchased')
plt.scatter(X_n[Y[:,0]==0][:,0],X_n[Y[:,0]==0][:,1],color='b',marker='o',label='Not Purchased')
plt.plot(x1_,x2_[0:,1])
plt.title('The data befor apply logistic regression')
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()
