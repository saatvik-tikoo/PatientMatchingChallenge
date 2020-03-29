from collections import defaultdict
import pandas as pd
from metaphone import doublemetaphone
import jellyfish
import states


class PatientMatchingCluster:
    def __init__(self, df, threshold):
        self.threshold = threshold
        self.df = df

    def __clean_sex__(self):
        # Cleans the geneder column
        for i, row in self.df.iterrows():
            if not pd.isna(row['Sex']):
                self.df.at[i, 'Sex'] = str(row['Sex']).upper()[0]
            else:
                self.df.at[i, 'Sex'] = 'U'

    def __hashing_names__(self):
        # Creates a metaphone for First Name and Last Name. We can also use soundex. (That code has been commented below)
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
        # Cleans First Name and Last Name column
        self.df['First Name'] = self.df['First Name'].str.lower()
        self.df['Last Name'] = self.df['Last Name'].str.lower()
        self.__hashing_names__()

    def __clean_dates__(self):
        # Cleans Date column
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
        # Cleans States column
        for i, row in self.df.iterrows():
            if not pd.isna(row['Current State']):
                if str(row['Current State']).lower() in states.us_states:
                    self.df.at[i, 'Current State'] = states.us_states[str(row['Current State']).lower()]
            else:
                self.df.at[i, 'Sex'] = ''

    def get_jaccard_similarity(self):
        """ This is where the magic happens
            Here we calculate the Jaccard Similarity between each pair of records and
            then group them based on a threshold value
        """

        # In further work we can use sets instead of Dataframes to make the computations faster
        # df = self.df.values.tolist()
        # df = [set(row) for row in df]
        cluster = defaultdict(set)
        cluster_number = 1
        columns_clustered = set()
        for i, row in self.df.iterrows():
            if i in columns_clustered:
                continue
            test_set = {str(row['First Name']), str(row['Last Name']), str(row['Sex']), str(row['Sex']), str(row['Current State']),
                        str(row['Date_month']) + '/' + str(row['Date_day']) + '/' + str(row['Date_year']), str(row['Current Zip Code'])}

            for j, row_inner in self.df.iterrows():
                if j in columns_clustered:
                    continue
                actual_set = {str(row_inner['First Name']), str(row_inner['Last Name']), str(row_inner['Sex']),
                              str(row_inner['Date_month']) + '/' + str(row_inner['Date_day']) + '/' + str(row_inner['Date_year']),
                              str(row_inner['Current Zip Code']), str(row_inner['Current State'])}
                numerator = len(test_set.intersection(actual_set))
                denominator = len(test_set.union(actual_set))
                similarity = float(numerator) / denominator
                if similarity > self.threshold:
                    if i not in columns_clustered and j not in columns_clustered:
                        cluster[cluster_number].add(i)
                        cluster[cluster_number].add(j)
                        columns_clustered.add(i)
                        columns_clustered.add(j)
                        cluster_number += 1
                    elif i not in columns_clustered:
                        for k, v in cluster.items():
                            if j in v:
                                cluster[k].add(i)
                                columns_clustered.add(i)
                                break
                    elif j not in columns_clustered:
                        for k, v in cluster.items():
                            if i in v:
                                cluster[k].add(j)
                                columns_clustered.add(j)
                                break
        return cluster

    def data_clean(self):
        # For cleaning the data
        self.__clean_sex__()
        self.__clean_names__()
        self.__clean_dates__()
        self.__clean_states__()
        self.df = self.df.fillna('')

    def print_groups(self, cluster):
        # This is for printing the groups and the records in each group
        for k, v in cluster.items():
            print('In group: {} indexes are: {}'.format(k, v))

    def get_accuracy(self, calculated_cluster):
        # Calculate the accuracy of our algorithm
        actual_clusters = defaultdict(set)
        for i, row in self.df.iterrows():
            actual_clusters[row['GroupID']].add(i)

        incorrect = 0
        for k, v in calculated_cluster.items():
            pairing_list = dict()
            for dp in v:
                for k_a, v_a in actual_clusters.items():
                    if dp in v_a:
                        if k_a not in pairing_list:
                            pairing_list[k_a] = 0
                        pairing_list[k_a] += 1
            if len(pairing_list) > 1:
                incorrect += (len(v) - max(pairing_list.values()))
        print('\nAccuracy of the Clusters is: %.2f' % ((1 - incorrect / len(self.df)) * 100))


def main():
    data = pd.read_csv('data/Patient Matching Data.csv')
    pm = PatientMatchingCluster(data, threshold=0.5)
    pm.data_clean()
    cluster = pm.get_jaccard_similarity()
    pm.print_groups(cluster)
    pm.get_accuracy(cluster)


if __name__ == '__main__':
    main()
