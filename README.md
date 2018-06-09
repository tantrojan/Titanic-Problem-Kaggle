# Titanic Problem

In this challenge from Kaggle, participants were asked to complete the analysis of what sorts of people were likely to survive. In particular, participants were asked to apply the tools of machine learning to predict which passengers survived the tragedy.
# Approach to Solving the Problem
  - Reading the Datasets Provided
  - Bringing out important features by **Visualizing the Data**
  - **Cleaning** the dataset
  - **Normalizing** the dataset and creating features
  - **Fitting** the features into the Classifier
  - **Predicting** the results using the Trained Classifier
### About the Dataset
##### **Dataset Parameters :**
###
| Attribute | Description |
| ------ | ------ |
| PassengerId| Used for indexing |
| Survived | Whether the passenger survived or not |
| Pclass | Class of the ticket |
| Name | Name of the passenger |
| Sex | Gender male or female |
| Age | Age of the Passenger |
| SibSp | Number of Siblings of the passenger |
| Parch | Number of Children and Parents of the passenger |
| Ticket |Ticket number|
| Fare | Ticket price |
| Cabin | Cabin number |
| Embarked | Place of Embarkment |

## Visualization
Grouping the passengers in each **Category** and visualizing the count of Passengers *Survived* and *Died*
#### Gender
  - Male
  - Female
<img src="./Visualize/Images2/gender.png" />

#### Ticket Class
  - (1,2,3)
<img src="./Visualize/Images2/class.png" />

#### Age
  - (0 to 16) 
  - (17 to 32)
  - (33 to 48)
  - (49 to 64)
<img src="./Visualize/Images2/age.png" />

#### Title
  - Mr
  - Master
  - Miss
  - Mrs
  - Rare
<img src="./Visualize/Images2/title.png" />

#### Fare
  - (0 to 8) 
  - (9 to 11)
  - (12 to 22)
  - (22 to 40)
  - (41 to 513)
<img src="./Visualize/Images2/fare.png" />

#### Embarkment
  - (S,Q,C)
<img src="./Visualize/Images2/embarkment.png" />

#### Family
  - (1,2,3,4,5,6,7,8,9,10,11)
<img src="./Visualize/Images2/family.png" />

## Data Processing 
The dataset provided needs to be cleaned and normalized in order to feed them into the classifier.
  - **Age** is mapped into **Age-Groups**
  - **Gender** is mapped as { 'male' : 1 , 'female' : 0 }
  - **Fare** is mapped into **Fare-Groups**
  - **Embarkment** is mapped as { 'S' : 0 , 'Q' : 1 , 'C' : 2 }
 
