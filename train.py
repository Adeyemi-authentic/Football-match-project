import pandas as pd
import mysql.connector
import numpy as np
#using my sql credentials 
from config import get_db_connection
conn = get_db_connection()

query="SELECT * FROM Matches_data;"

df=pd.read_sql(query,conn)

df

conn.close()

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=df.copy()
data.info()

data.columns

X=data.drop(['id','home_score', 'away_score', 'Result', 'matchday', 'date', 'home_points', 'away_points','GG', 'Cumulative_Home_Matches',
             'Cumulative_Away_Matches', 'home_wins', 'home_draws', 'home_losses', 'away_wins', 'away_draws','away_losses','home_goal_ratio', 'away_goal_ratio'],axis=1)

y=data['Result']

sc=StandardScaler()
X=sc.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


X_train.dtype
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.to_numpy())
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test.to_numpy())



print("Class distribution:")
print(np.unique(y_test_tensor, return_counts=True))


class FootballML(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_size2,output_dim,dropout_prob=0.35):
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2=nn.Linear(hidden_size,hidden_size2)
        self.dropout2=nn.Dropout(dropout_prob)
        self.fc3=nn.Linear(hidden_size2,output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply first linear layer and activation
        x = self.dropout1(x)    # Apply dropout1
        x=torch.relu(self.fc2(x))  # second hidden layer activation
        x=self.dropout2(x)
        output = self.fc3(x)         # Final linear layer
        return output

input_size=14
hidden_size=42
hidden_size2=18
dropout_prob=0.35
output_dim=3



from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)


model=FootballML(input_size,hidden_size,hidden_size2,output_dim,dropout_prob)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer=optim.Adam(model.parameters(),lr=0.008)

num_epochs=880

for epoch in range(num_epochs):
    outputs=model(X_train_tensor)
    loss=criterion(outputs,y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch +1) % 80 ==0:
        print(f"Epoch[{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    
        

model.eval()
with torch.no_grad():
    logits=model(X_test_tensor)
    probabilities=torch.softmax(logits,dim=1)
    _,pred_classes=torch.max(logits,1)
    # Validation

    val_outputs = model(X_test_tensor)
    val_loss = criterion(val_outputs, y_test_tensor)
    _, val_preds = torch.max(val_outputs, 1)
    val_acc1 = (val_preds == y_test_tensor).float().mean()
    

y_pred=pred_classes.numpy()

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

print(classification_report(y_test_tensor,y_pred))

print(confusion_matrix(y_test_tensor,y_pred))


torch.save(model.state_dict(),"model.pth")