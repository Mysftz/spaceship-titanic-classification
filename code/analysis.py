import os, tensorflow as tf, tensorflow_decision_forests as tfdf, pandas as pd, numpy as np, seaborn as sb

dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_df, test_df = pd.read_csv(dir+'/source/train.csv'), pd.read_csv(dir+'/source/test.csv') 
#---------------------------------------------------------------------------------------------------------------------
train_df = train_df.drop(['PassengerId', 'Name'], axis=1)

train_df.isnull().sum().sort_values(ascending=False)

train_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
train_df.isnull().sum().sort_values(ascending=False)

label = "Transported"
train_df[label] = train_df[label].astype(int)

train_df['VIP'] = train_df['VIP'].astype(int)
train_df['CryoSleep'] = train_df['CryoSleep'].astype(int)

train_df[["Deck", "Cabin_num", "Side"]] = train_df["Cabin"].str.split("/", expand=True)

try:
    train_df = train_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(train_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)

tfdf.keras.get_all_models()
rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"])

rf.fit(x=train_ds)
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

inspector = rf.make_inspector()
inspector.evaluation()

evaluation = rf.evaluate(x=valid_ds,return_dict=True)

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")
#---------------------------------------------------------------------------------------------------------------------
'''
submission_id = test_df.PassengerId

# Replace NaN values with zero
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)

# Creating New Features - Deck, Cabin_num and Side from the column Cabin and remove Cabin
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)

# Convert boolean to 1's and 0's
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Convert pd dataframe to tf dataset
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

# Get the predictions for testdata
predictions = rf.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool)
output = pd.DataFrame({'PassengerId': submission_id,
                       'Transported': n_predictions.squeeze()})

output.head()
'''