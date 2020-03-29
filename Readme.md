#### Team Name:
## Da VINCE
Members:
- Saatvik Tikoo
- Kurian KS

## Patient Matching

### Set up instructions

#### What programming language is this in?
-We used Python for this project. 

#### How to set up your project to verify your work?
#### Initial Requirements: 
- The system should have Python 3.6,
- Clone the code from the git repository: 
- After Git clone go to the directory and run **pip install -r requirements.txt**. This will install all the dependencies

#### How can we replicate your steps to achieve your challenge?
1. Our code is in _ patientMatching_cluster.py _. To run the code do **python patientMatching_cluster.py**
2. This will generate Groups and puts each record in an appropriate group.

In order to test another file, simply change the data file in the data folder.

### Proof of Concept Steps
There are two steps in our approach to the problem:
1. Data Cleaning and Normalization: This step is used as a data pre-processing step and is used to clean the records. We are cleaning the following fields:
    1. Gender: Converting all the strings to lower and converting the null fields to 'Unknowns'. After this, we converted all the fields to M: Male, F: Female, U: Unknown.
    2. Names: Used doublemetaphone() to get similar sounding words together. Before this, we convert all the strings to lower case.
    3. Dates: We are assuming that all the date fields are in the format of mm/dd/yyyy. Here we are splitting the given string by '/' and check if we have 3 integral values of which the 3rd one should be of 4 digits. If none of these exists then it is given a Null value.
    4. States:  Here we are converting all the strings to lower case and then using a dictionary to assign abbreviations to all the States. We are only using US States.

The data fields we are using are: ['First Name', 'Sex', 'Last Name', 'Date of Birth', 'Current Zip Code', 'Current State']

2. Clustering: We are using Jaccard's similarity score to find similarities between the two records. The time complexity of doing it is going to be O(N^2), where N = Total Number of records in our data file. 

##### For further work: We can surely make the complexity better by using Sets instead of data frames. _

Also, one thing to note here is we have used a _ threshold _ value. This is a hyperparameter and needs tuning for new data. We tuned it based on the data that was provided to us.

**Accuracy of our clustering algorithm is: 96.02%**

There is another file in the code-base called _ patientMatch_classify.py _. We first worked on this file. Earlier we thought this was a classification problem and hence used Random Forest Classifier to get an accuracy of 92.06%. But later we realized this was a clustering problem hence we moved to the code discussed above. We have kept the code, in case it is required.

### Contact info
Email: tikoo@usc.edu

### Devpost
https://devpost.com/software/patient-matching-challenge