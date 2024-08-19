import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spaceship_train_Set = pd.read_csv('train.csv')
Spaceship_test_Set = pd.read_csv('test.csv')
print(Spaceship_train_Set)
print(Spaceship_test_Set)


##########################################################
################# Cleaning Data ##########################
##########################################################

#Number of Columns and rows.
print(f"Spaceship_train_Set.shape is:\n {Spaceship_train_Set.shape}")
print(f"Spaceship_test_Set.shape is:\n {Spaceship_test_Set.shape}")
print(100*"-")

#Sum for nulls 
print(f"Number of null values in each column for the train set:\n{Spaceship_train_Set.isnull().sum()}")
print(f"Number of null values in each column for the test set:\n{Spaceship_test_Set.isnull().sum()}")
print(100*"-")

#SUM for duplicates
print(f"Number of duplicate rows for the train set:\n {Spaceship_train_Set.duplicated().sum()}")
print(f"Number of duplicate rows for the test set:\n {Spaceship_train_Set.duplicated().sum()}")
print(100*"-")
# print(f"Descriptive statistics of DataFrame:\n {Spaceship_train_Set.describe()}")
# print(100*"-")


# I take only the number of members in cambin.
Spaceship_train_Set['Cabin_num'] = Spaceship_train_Set['Cabin'].astype(str).apply(lambda x: x.split('/')[1] if '/' in x else None)
Spaceship_test_Set['Cabin_num'] = Spaceship_test_Set['Cabin'].astype(str).apply(lambda x: x.split('/')[1] if '/' in x else None)
Spaceship_train_Set.drop('Cabin', axis=1, inplace=True)
Spaceship_test_Set.drop('Cabin', axis=1, inplace=True)


Spaceship_train_Set.fillna(0, inplace=True)
Spaceship_test_Set.fillna(0, inplace=True)

#I delete the "_" from the PassengerId column.
Spaceship_train_Set['PassengerId'] = Spaceship_train_Set['PassengerId'].str.replace('_', '')
Spaceship_test_Set['PassengerId'] = Spaceship_test_Set['PassengerId'].str.replace('_', '')

print(f"Number of null values in each column for the train set:\n{Spaceship_train_Set.isnull().sum()}")
print(f"Number of null values in each column for the test set:\n{Spaceship_test_Set.isnull().sum()}")



# The '_' deleted from the PassengerId columns.
Spaceship_test_Set['PassengerId'] = Spaceship_test_Set['PassengerId'].str.replace('_', '')
Spaceship_test_Set['PassengerId'] = Spaceship_test_Set['PassengerId'].str.replace('_', '')
print(Spaceship_test_Set)
print(Spaceship_test_Set)





#The three below columns droped because aren't usefull for my prediction (we want only the PassengerId)
Spaceship_train_Set.drop(['Destination', 'HomePlanet', 'Name'], axis=1, inplace=True)
Spaceship_test_Set.drop(['Destination', 'HomePlanet', 'Name'], axis=1, inplace=True)
print(Spaceship_train_Set)
print(Spaceship_test_Set)


print(Spaceship_train_Set.dtypes)
print(Spaceship_test_Set.dtypes)


# Select the columns you want to convert
columns_to_convert = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Convert the selected columns to integers
Spaceship_train_Set[columns_to_convert] = Spaceship_train_Set[columns_to_convert].astype(int)
Spaceship_test_Set[columns_to_convert] = Spaceship_test_Set[columns_to_convert].astype(int)

# 
Spaceship_train_Set['VIP'] = Spaceship_train_Set['VIP'].astype(bool)
Spaceship_test_Set['VIP'] = Spaceship_test_Set['VIP'].astype(bool)

Spaceship_train_Set['CryoSleep'] = Spaceship_train_Set['CryoSleep'].astype(bool)
Spaceship_test_Set['CryoSleep'] = Spaceship_test_Set['CryoSleep'].astype(bool)

# Select boolean columns
bool_columns_for_trainSet = Spaceship_train_Set.select_dtypes(include='bool').columns
bool_columns_for_testSet = Spaceship_test_Set.select_dtypes(include='bool').columns

# Convert boolean columns to integers
Spaceship_train_Set[bool_columns_for_trainSet] = Spaceship_train_Set[bool_columns_for_trainSet].astype(int)
Spaceship_test_Set[bool_columns_for_testSet] = Spaceship_test_Set[bool_columns_for_testSet].astype(int)



Spaceship_train_Set = Spaceship_train_Set.astype(int)
Spaceship_test_Set = Spaceship_test_Set.astype(int)
Spaceship_test_Set
#############################################################################################################################
############################################ Correlation matrix for the cols of interest (int64) ############################
#############################################################################################################################

########################## For the Spaceship_train_Set.
# Assuming recruitment_data is your DataFrame
correlation_matrix = Spaceship_train_Set.corr()

# # Display the correlation matrix
# print(correlation_matrix)

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()




##############################  For the Spaceship_test_Set.
# Assuming recruitment_data is your DataFrame
correlation_matrix = Spaceship_test_Set.corr()

# # Display the correlation matrix
# print(correlation_matrix)

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.show()


######################################################################################
################################  Prediction  #######################################
######################################################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = Spaceship_train_Set.drop('Transported', axis=1)  # Features
y = Spaceship_train_Set['Transported']  # Target variable

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
test_predictions = model.predict(Spaceship_test_Set)


# Assuming that Spaceship_test_Set and test_predictions already exist

# Create the DataFrame for submission
submission_df = pd.DataFrame({
    'PassengerId': Spaceship_test_Set['PassengerId'],
    'Transported': test_predictions  # Assuming 'test_predictions' contains your predictions
})


# Apply the lambda function to format the PassengerId
submission_df['PassengerId'] = submission_df['PassengerId'].astype(str).apply(
    lambda x: f"00{x[:-2]}_{x[-2:]}" if len(x[:-2]) == 2 else
              f"0{x[:-2]}_{x[-2:]}" if len(x[:-2]) == 3 else
              f"{x[:-2]}_{x[-2:]}"
)



# Convert predictions to boolean True/False
submission_df['Transported'] = submission_df['Transported'].astype(bool)

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Display the DataFrame for confirmation
print(submission_df)


