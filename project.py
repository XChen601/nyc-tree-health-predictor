'''
Name: Xing Chen
Email: xing.chen14@myhunter.cuny.edu
Resources: Sckit-learn documentation, Pandas documentation
Note: This file was converted from .ipynb to .py so theres some comments for cells
'''


# %%
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


# %% [markdown]
# # Visualizing and cleaning the data

# %%
tree_df = pd.read_csv('2015_Street_Tree_Census_-_Tree_Data_20231020.csv')

# new column called street name to match business df
def get_street_name(address):
    street_name = address.split(' ')[1:]
    street_name = ' '.join(street_name)
    street_name = street_name.replace('AVENUE', 'AVE').replace('STREET', 'ST')

    # Append "TH" to the end of numbers
    street_name = re.sub(r"(\d+)", r"\1TH", street_name)

    return street_name
tree_df['Address Street Name'] = tree_df['address'].apply(get_street_name)
tree_df.head()

# %% [markdown]
# ### Combining the 2 datasets

# %%
def merge_datasets(tree_df, business_df):
    """
    Merges the tree and business datasets
    """
    business_df = business_df[business_df['Address Street Name'].notna()]
    business_df = business_df[business_df['License Status'] == "Active"]
    business_count_df = business_df.groupby("Address Street Name").size().reset_index(name="Number of Businesses")
    merged_df = tree_df.merge(business_count_df, on='Address Street Name', how='inner')
    return merged_df

# %%
business_df = pd.read_csv('Legally_Operating_Businesses_20231020.csv')
merged_df = merge_datasets(tree_df, business_df)

# %% [markdown]
# ### Box plot showing tree diameter

# %%
sns.boxplot(data=merged_df, x='tree_dbh')
plt.show()

# %% [markdown]
# ### Pie chart showing distrubution of tree health by percentage

# %%
merged_df['health'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.axis('equal')
plt.show()


# %% [markdown]
# ### Bar chart showing distribution of tree health by borough by count

# %%
sns.countplot(data=merged_df, x='health', hue='borough')
plt.show()


# %% [markdown]
# # Viewing dataframe

# %%
# Check unique values for tree health
merged_df['health'].unique()

# %%
merged_df['curb_loc'].unique()

# %%
print(len(merged_df))

# %%
unique_trees = merged_df['spc_common'].unique()
unique_trees = [str(value) for value in unique_trees]
unique_trees = sorted(unique_trees)
unique_trees

# %%
print(len(unique_trees))

# %%
merged_df['nta'].unique()

# %% [markdown]
#

# %%
print(len(merged_df['st_senate'].unique()))
print(len(merged_df['boro_ct'].unique()))
print(len(merged_df['st_assem'].unique()))
print(len(merged_df['cncldist'].unique()))
print(len(merged_df['postcode'].unique()))


# %% [markdown]
# # Cleaning dataset

# %%
cleaned_df = merged_df.dropna(subset=['health', 'latitude', 'longitude'])

# drop inital columns that are not useful
drop_cols = ['nta_name', 'state', 'created_at', 'bbl', 'bin', 'census tract', 'address',
             'tree_id', 'block_id', 'spc_latin', 'borough', 'status', 'zip_city',
             "Address Street Name", 'nta', 'boro_ct', 'postcode', 'st_assem']

# check for columns with missing values
for col in cleaned_df.columns:
    missing_count = cleaned_df[col].isna().sum()
    # if more than 30% missing, drop the column
    if missing_count > cleaned_df.shape[0] * 0.3:
        print(col, missing_count)
        drop_cols.append(col)
# Drop all the columns in drop_cols
cleaned_df = cleaned_df.drop(columns=drop_cols)

# %%
for col in cleaned_df.columns:
    missing_count = cleaned_df[col].isna().sum()
    if missing_count > 0:
        print(col, missing_count)

# %%
# only 1 missing value for sidewalk so just drop it, same for spc_common
nan_count = cleaned_df['sidewalk'].isna().sum()
cleaned_df = cleaned_df.dropna(subset=['sidewalk', 'spc_common'])

# %%
# impute council district with the most popular one
mode_council_district = cleaned_df['council district'].mode()[0]
cleaned_df['council district'].fillna(mode_council_district, inplace=True)

# %%
# Drop rows where tree health is missing
cleaned_df = cleaned_df.dropna(subset=['health'])

# %%
cleaned_df.to_csv('cleaned_data.csv', index=False)

# %%
cleaned_df.head()

# %% [markdown]
# # Create a map visualization

# %%
cleaned_df = cleaned_df.dropna(subset=['latitude', 'longitude'])

subset_data = cleaned_df.sample(n=min(10000, len(cleaned_df)), random_state=1)
m = folium.Map(location=[subset_data.iloc[0]['latitude'], subset_data.iloc[0]['longitude']], zoom_start=13)

# Function to choose a color based on the health of the tree
def color_producer(health_status):
    """
    matches tree health to a color
    """
    if health_status == 'Good':
        return 'green'
    if health_status == 'Fair':
        return 'orange'
    if health_status == 'Poor':
        return 'red'

# Add points to the map
for idx, row in subset_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color_producer(row['health']),
        fill=True,
        fill_color=color_producer(row['health']),
        fill_opacity=0.7,
        popup=f"Health: {row['health']}<br>Species: {row['spc_common']}"
    ).add_to(m)

# Create a legend
LEGEND_HTML = '''
<div style="position: fixed;
     bottom: 50px; left: 50px; width: 150px; height: 90px;
     border:2px solid grey; z-index:9999; font-size:14px; background-color: white;
     ">&nbsp; <b>Tree Health Legend</b> <br>
     &nbsp; Green: Good <br>
     &nbsp; Orange: Fair <br>
     &nbsp; Red: Poor <br>
</div>
'''

# Add the legend to the map
m.get_root().html.add_child(folium.Element(LEGEND_HTML))

# Save the updated map with a legend
m.save('tree_health_map.html')

# %% [markdown]
# # Data Transformation

# %%
# convert curb_loc to numeric
cleaned_df['curb_loc'] = cleaned_df['curb_loc'].apply(lambda x: 1 if x == 'OnCurb' else 0)

# %%
# convert tree health to numeric
health_encoding = {'Good': 1, 'Fair': 0, 'Poor': 0}
cleaned_df['health'] = cleaned_df['health'].map(health_encoding)

# %%
# convert sidewalk to numeric
cleaned_df['sidewalk'] = cleaned_df['sidewalk'].apply(lambda x: 1 if x == 'Damage' else 0)

# %%
# one hot encode user_type
cleaned_df = pd.get_dummies(cleaned_df, columns=['user_type'], dtype=int)


# %%
# remove latitude and longitude columns for model
cleaned_df = cleaned_df.drop(columns=['latitude', 'longitude', 'y_sp', 'x_sp'])

# %%
cleaned_df.head()

# %%
changed_cols = ['root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light',
                'trnk_other', 'brch_light', 'brch_shoe', 'brch_other']
for col in changed_cols:
    print(col, cleaned_df[col].unique())

# %%
# change the columns with No and Yes to numerical
for col in changed_cols:
    cleaned_df[col] = cleaned_df[col].apply(lambda x: 1 if x == 'Yes' else 0)


# %%
cleaned_df['spc_common'].nunique()



# %% [markdown]
# ### Convert spc_common into numerical values

# %%
# encode spc_common
le = LabelEncoder()
cleaned_df['spc_common_encoded'] = le.fit_transform(cleaned_df['spc_common'].astype(str))
cleaned_df = cleaned_df.drop(columns=['spc_common'])

# %%
cleaned_df.head()

# %%
# export transformed data
cleaned_df.to_csv('transformed_data.csv', index=False)

# %% [markdown]
# # Creating the machine learning models


# %%
# sampled_data = cleaned_df.sample(n=20000, random_state=1) # so i can run the models faster
sampled_data = cleaned_df

# balance the dataset
minority = sampled_data[sampled_data['health'] == 0]
majority_downsampled = sampled_data[sampled_data['health'] == 1].sample(
    len(minority), random_state=123)

sampled_data = pd.concat([majority_downsampled, minority])
y = sampled_data['health']
X = sampled_data.drop(columns=['health'])

# %%
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %% [markdown]
# ### Logistic Regression

# %%
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)


# %%
lr_conf_matrix = confusion_matrix(y_test, y_pred)

# %% [markdown]
# ### KNN

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Evaluating the model
knn_accuracy = accuracy_score(y_test, predictions)

# %%
knn_conf_matrix = confusion_matrix(y_test, y_pred)


# %% [markdown]
# ### Decision Tree Classifier



decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred)


# %%
dt_conf_matrix = confusion_matrix(y_test, y_pred)


# %% [markdown]
# ### random forest

# %%
def rf_grid_search():
    """
    Performs a grid search on the random forest classifier
    """
    param_grid = {
        'n_estimators': [50, 75, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=123),
                            param_grid=param_grid,
                            cv=3,
                            n_jobs=-1,
                            verbose=2)

    grid_search.fit(X_train, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    return best_params
rf_grid_search()

# %%
rf = RandomForestClassifier(
    max_depth= 30, min_samples_leaf= 2, min_samples_split= 2, n_estimators=150)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_accuracy = rf.score(X_test, y_test)

# %%
# Calculate the confusion matrix
rf_conf_matrix = confusion_matrix(y_test, y_pred)

# Create a heatmap from the confusion matrix
plt.figure(figsize=(3, 2))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Unhealthy', 'Healthy'], yticklabels=['Unhealthy', 'Healthy'])

plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Random Forest Confusion Matrix')

plt.show()

# %% [markdown]
# # Results

# %%
accuracy_scores = [lr_accuracy, knn_accuracy, dt_accuracy, rf_accuracy]
labels = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']
plt.bar(labels, accuracy_scores)
plt.ylabel('Accuracy Score')
plt.xlabel('Models')
plt.title('Accuracy Scores for Different Models')
plt.show()


# %%
def print_model_accs(labels, accuracy_scores):
    for label, score in zip(labels, accuracy_scores):
        print(f'{label}: {score}')
print_model_accs(labels, accuracy_scores)

# %% [markdown]
# Random Forest performed the best with an accuracy of .6734 while a simple
# logistic regression model performed the worse with an accuracy of .57

# %% [markdown]
#


