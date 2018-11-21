import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

sys.stdout.write("Hello!")
data = pd.read_table(r'''C:\Users\saifmehyar\Downloads\Cities Regression Model.txt''',delimiter=',')
X = pd.DataFrame(data['TotalRegistrants'])
Y = pd.DataFrame(data['TotalPaying'])

plt.plot(X,Y,'.',lw=0)
plt.xlabel('Registrants')
plt.ylabel('Paying')
plt.show()

lm = linear_model.LinearRegression()
model = lm.fit(X,Y)

predictions = lm.predict(X)

#plt.axis([0,3000,0,210])
plt.plot(Y,predictions,'.',lw=0)
plt.show()

print(predictions)

arrayboy = np.array([[341,657]]) #Madison / Oklahoma City Test Model 
arrayboy2 = np.reshape(arrayboy,(-1,1))
print(model.predict(arrayboy2))
truevalue1 = arrayboy2[1]/2
#truevalue2 = arrayboy2[2]/2
print(truevalue1)


sys.stdout.write("I'm Done!")


