from collections import defaultdict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sklearn.metrics as mx
from metaphone import doublemetaphone
import states
import jellyfish


class PatientMatching:
    def __init__(self, df):
        self.df = df

    def __clean_sex__(self):
        for i, row in self.df.iterrows():
            if not pd.isna(row['Sex']):
                self.df.at[i, 'Sex'] = str(row['Sex']).upper()[0]
            else:
                self.df.at[i, 'Sex'] = 'U'

    def __hashing_names__(self):
        for i, row in self.df.iterrows():
            if not pd.isna(row['First Name']):
                #  self.df.at[i, 'First Name'] = jellyfish.soundex(row['First Name'])
                self.df.at[i, 'First Name'] = doublemetaphone(row['First Name'])[0]
            else:
                self.df.at[i, 'First Name'] = ''
            if not pd.isna(row['Last Name']):
                # self.df.at[i, 'Last Name'] = jellyfish.soundex(row['Last Name'])
                self.df.at[i, 'Last Name'] = doublemetaphone(row['Last Name'])[0]
            else:
                self.df.at[i, 'Last Name'] = ''

    def __clean_names__(self):
        self.df['First Name'] = self.df['First Name'].str.lower()
        self.df['Last Name'] = self.df['Last Name'].str.lower()
        self.__hashing_names__()

    def __clean_dates__(self):
        for i, row in self.df.iterrows():
            if not pd.isna(row['Date of Birth']):
                dt = str(row['Date of Birth']).strip().split('/')
                if len(dt) == 3:
                    self.df.at[i, 'Date_month'] = dt[0].strip()
                    self.df.at[i, 'Date_day'] = dt[1].strip()
                    self.df.at[i, 'Date_year'] = dt[2].strip() if '+' not in dt[2] else dt[2].strip()[:-1] + dt[2].strip()[-2]
                else:
                    max_possible_dt = []
                    for idx in dt:
                        if idx.strip().isnumeric():
                            max_possible_dt.append(int(idx))
                    self.df.at[i, 'Date_month'] = max_possible_dt[0] if len(max_possible_dt) >= 1 else ''
                    self.df.at[i, 'Date_day'] = max_possible_dt[1] if len(max_possible_dt) >= 2 else ''
                    self.df.at[i, 'Date_year'] = max_possible_dt[3] if len(max_possible_dt) >= 3 and len(str(max_possible_dt[3])) == 4 else ''
            else:
                self.df.at[i, 'Date_month'] = ''
                self.df.at[i, 'Date_day'] = ''
                self.df.at[i, 'Date_year'] = ''

    def __clean_states__(self):
        for i, row in self.df.iterrows():
            if not pd.isna(row['Current State']):
                if str(row['Current State']).lower() in states.us_states:
                    self.df.at[i, 'Current State'] = states.us_states[str(row['Current State']).lower()]
            else:
                self.df.at[i, 'Sex'] = ''

    def model(self):
        X = self.df[['First Name', 'Sex', 'Last Name', 'Date_month', 'Date_day', 'Date_year', 'Current Zip Code', 'Current State']].copy()
        X = pd.get_dummies(X, drop_first=True)
        y = self.df['GroupID']
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        forest = RandomForestClassifier(random_state=0, bootstrap=True)

        accuracies = defaultdict(list)
        f1_scores = defaultdict(list)
        for train, test in skf.split(X, y):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            forest.fit(X_train, y_train)
            y_pred_forest = forest.predict(X_test)
            accuracies['forest'].append(mx.accuracy_score(y_test, y_pred_forest))
            f1_scores['forest'].append(mx.f1_score(y_test, y_pred_forest, average='weighted'))

        print('Accuracy of Random Forest is: %.2f' % (np.mean(accuracies['forest']) * 100))
        # print('F1 Score of Random Forest is: ', np.mean(f1_scores['forest']))

        self.df['predicted_label'] = forest.predict(X)

    def __jaccard_similarity__(self, test_set, data):
        similarity = dict()
        for i, row in data.iterrows():
            actual_set = {str(row['First Name']), str(row['Last Name']), str(row['Sex']), str(row['Current State']),
                          str(row['Date_month']) + '/' + str(row['Date_day']) + '/' + str(row['Date_year']), str(row['Current Zip Code'])}
            numerator = len(test_set.intersection(actual_set))
            denominator = len(test_set.union(actual_set))
            similarity[i] = float(numerator) / denominator
        return data.iloc[max(similarity.items(), key=lambda x: x[1])[0]]['GroupID']

    def __split_data__(self):
        df = self.df.copy()
        groups = set(df['GroupID'].tolist())
        train = pd.DataFrame()
        for grp in groups:
            train = pd.concat([train, df.loc[grp == df['GroupID']].head(1)], ignore_index=True)
        cond = df['GroupID'].index.isin(train['GroupID'].index)
        test = df.drop(df[cond].index)
        return train, test

    def get_jaccard_similarity(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        train, test = self.__split_data__()
        train.drop_duplicates('GroupID', inplace=True)
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)
        pred_groupids = []
        correct = 0
        for i, row in test.iterrows():
            txt = {str(row['First Name']), str(row['Last Name']), str(row['Sex']), str(row['Sex']), str(row['Current State']),
                   str(row['Date_month']) + '/' + str(row['Date_day']) + '/' + str(row['Date_year']), str(row['Current Zip Code'])}
            val = self.__jaccard_similarity__(txt, train)
            pred_groupids.append(val)
            if row['GroupID'] == val:
                correct += 1
        accuracy = (correct / test.shape[0]) * 100
        print('Accuracy using Jaccard Similarity is: %.2f'% accuracy)

    def data_clean(self):
        self.__clean_sex__()
        self.__clean_names__()
        self.__clean_dates__()
        self.__clean_states__()
        self.df = self.df.fillna('')


def main():
    data = pd.read_csv('data/Patient Matching Data.csv')
    pm = PatientMatching(data)
    pm.data_clean()
    pm.model()
    pm.get_jaccard_similarity()


if __name__ == '__main__':
    main()
