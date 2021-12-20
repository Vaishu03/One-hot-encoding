# One-hot-encoding
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"E:\ML\carprices.csv")
dum = pd.get_dummies(df.Cars)
merge = pd.concat([df,dum],axis = 'columns')
final = merge.drop(['Cars','Toyota'],axis = 'columns')
obj = LinearRegression()
X = final.drop(['Price'],axis = 'columns')
y = final.Price
obj.fit(X,y)
#price of a mercedez benz i.e. 4 yr old with mileage 45000
obj.predict([['45000','4','0','0','1']])
#price of a BMW X5 i.e. 7 yr old with mileage 86000
obj.predict([['86000','7','0','1','0']])
#predicting the accuracy of the model
obj.score(X,y)
