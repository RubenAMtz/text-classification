

```javascript
%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
```


    <IPython.core.display.Javascript object>



# Text classification with NLTK and Scikit-Learn
## Summary: 
### We have collected data from two different categories: Hotels and Travel & Health and medical, cleaned the text and transformed it to a suitable representation, in this case we implemented a Bag of Words model.  Both categories contain an imbalanced amount of classes so we have to evaluate them using Precision-Recall curves and Avg. Precision. We  test different classifiers and finally train the models using a Naive Bayes model (as it perfomes the best compared to rest). We reach an average precision of 95%


```python
# from standard python library
import re, string 
import os
import urllib.request
import warnings
warnings.filterwarnings('ignore')

# third party libraries: BeautifulSoup, Numpy, Pandas, NLTK, Sklearn
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import sklearn
from sklearn.model_selection import cross_val_score, cross_val_predict
from joblib import load, dump
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn import ensemble, naive_bayes, svm, tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve, average_precision_score, f1_score
from sklearn import preprocessing

%matplotlib inline
import matplotlib.pyplot as plt
```

## We are thinking of testing different classifiers with our datasets, we will create a class to handle all the tests at once and return the best classifier.


```python
class TestClassifiers(object):

    def __init__(self, X, y, classifiers=None, scoring="accuracy"):
        
        self.X = X
        self.y = y
        
        if classifiers is None:
            self.classifiers = []
        self.classifiers = classifiers
        
        self.scoring = scoring
        
        # attribute asigned in call_fit method
        self.scores = []
        self.preds = []

    
    def cv_on_classifiers(self, folds=5):
        """ Apply cross validation on all classifiers
        """

        for classifier in self.classifiers:
            print("Cross-Validation using {} Classifier".format(type(classifier).__name__))
            score = cross_val_score(classifier, self.X, self.y, scoring=self.scoring, cv=folds)
            print("Scores:", score, "Mean score:", np.mean(score))
            self.scores.append(np.mean(score))
    
    def cv_pred_on_classifiers(self, folds=5, method='predict'):
        """ Apply cross validation predict on all classifiers
        """

        for classifier in self.classifiers:
            print("Predictions using Cross-Validation with {} Classifier".format(type(classifier).__name__))
            preds = cross_val_predict(classifier, self.X, self.y, cv=folds, n_jobs=5, method='predict')
            self.preds.append(preds)


    def fit_on_classifiers(self, prefix="", dump_=True):
        """ Apply fit method to all classifiers
        """

        for classifier in self.classifiers:
            print("Fitting {} Classifier".format(type(classifier).__name__))
            classifier.fit(self.X, self.y)
            print("Finished training classifier ", str(classifier))
            
            if dump_:
                dump(classifier, "../models/{}_{}.joblib".format(type(classifier).__name__, prefix))
    

    def best_classifier(self):
        """ Returns the classifier with the max score
        """
        print("Best classifier: \n")
        max_ = self.scores.index(np.max(self.scores))
        return self.classifiers[max_]


    def load_classifiers(self):
        """ Load classifiers from file
        """
        for classifier in self.classifiers:
            classifier = load("models/{}.joblib".format(type(classifier).__name__))

```

## Let's leave this class aside for a second, we will come back to it after downloading and processing our data. We will focus on only two categories out of the 5, "Health and medical" and "Hotels and travel":


```python
def parse_html(url):
    response = urllib.request.urlopen(url)
    html_doc = response.read()
    return BeautifulSoup(html_doc, 'html.parser')

def explore_and_download(url):
    # Fetch the html file: http://mlg.ucd.ie/modules/yalp/
    soup = parse_html(url)
    all_reviews = []
    all_stars = []
    # links at home page
    for i, link in enumerate(soup.find_all('a')):
        #Category: Automotive
        #Category: Bars
        #Category: Health and medical
        #Category: Hotels and travel
        #Category: Restaurant
        if i == 2 or i == 3: # 'Health & Medical' and 'Hotels & Travel'
            
            soup = parse_html(url+link.get('href'))
            # links level 2 (businesses)
            for link_2 in soup.find_all('a'):
                # List of 122 businesses in the category Health and medical
                # List of all 89 businesses in the category Hotels and travel
                print(link_2.get('href'))
                # access link & get reviews
                soup_2 = parse_html(url + link_2.get('href'))
                # find the scores:
                for p in soup_2.find_all('img'):
                    try: #the first entry has no attribute 'alt'
                        all_stars.append(p['alt'])
                    except:
                        pass
                # find the reviews:
                for p in soup_2.find_all('p'):
                    if p['class'][0] == 'text':
                        all_reviews.append(p.get_text())

            # concatenate and save the results
            results = np.concatenate((np.array(all_stars).reshape(-1,1), np.array(all_reviews).reshape(-1,1)), axis=1)
            pd.DataFrame(results, columns=['rate','review']).to_csv('../csv/' + link.get_text()[10:] + '.csv', index=False)
```

## Let's see what our Health and medical dataset looks like:


```python
base_url = "http://mlg.ucd.ie/modules/yalp/"

if not os.path.isfile('../csv/Health and medical.csv'):
    explore_and_download(base_url)
    
health = pd.read_csv('../csv/Health and medical.csv')
health
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rate</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5-star</td>
      <td>I have so many good things to say about this p...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5-star</td>
      <td>I found them to be highly skilled and an exper...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5-star</td>
      <td>Where do I even begin? This office has been so...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5-star</td>
      <td>I went in because I had toothache and needed a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5-star</td>
      <td>Found a new dental office. This place is amazi...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5-star</td>
      <td>Dr. Carlos is always on top of things, I've ne...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5-star</td>
      <td>Dr. Carlos and the staff were very friendly. T...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5-star</td>
      <td>Love these guys! Had a chip in my tooth and no...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5-star</td>
      <td>I just found this office in Scottsdale and the...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5-star</td>
      <td>Dr. Mandap has been my dentist for many years ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5-star</td>
      <td>A bit of a drive on the freeways but this plac...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1-star</td>
      <td>Wow. Do not order from here. I ordered an item...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1-star</td>
      <td>I was searching google for diabetic shoes for ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5-star</td>
      <td>Am I reviewing the same company as everyone el...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1-star</td>
      <td>This company is terrible! My mom purchased a p...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1-star</td>
      <td>Horrible Experience. I ordered an item and req...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1-star</td>
      <td>Ordered item in April 2015. Never arrived. The...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1-star</td>
      <td>Terrible online service! I ordered one item an...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1-star</td>
      <td>I want to let everyone know to not make purcha...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1-star</td>
      <td>Horrible customer service ... Ordered an item ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1-star</td>
      <td>I purchased a calf sleeve from this company to...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1-star</td>
      <td>The website descriptions and advertising is pu...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1-star</td>
      <td>Do not deal with this company. I ordered some ...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1-star</td>
      <td>they had the best price around for the item i ...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1-star</td>
      <td>I should have looked at the reviews before ord...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1-star</td>
      <td>DON"T BUY FROM THIS COMPANY. my order never ar...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1-star</td>
      <td>There are only two positive reviews (read: shi...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1-star</td>
      <td>Take your business else where. Notice how the ...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5-star</td>
      <td>I found ActiveForever after a family incident....</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1-star</td>
      <td>Ordered a product online and paid for for expe...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1420</th>
      <td>5-star</td>
      <td>Yuliana, the owner, is professional, knowledge...</td>
    </tr>
    <tr>
      <th>1421</th>
      <td>1-star</td>
      <td>Everything was great until they burned my cupi...</td>
    </tr>
    <tr>
      <th>1422</th>
      <td>5-star</td>
      <td>Let me start by saying that I'm used to paying...</td>
    </tr>
    <tr>
      <th>1423</th>
      <td>5-star</td>
      <td>Yuliya is the best at her job. I highly recomm...</td>
    </tr>
    <tr>
      <th>1424</th>
      <td>5-star</td>
      <td>I have had about 4 session of hair removal and...</td>
    </tr>
    <tr>
      <th>1425</th>
      <td>5-star</td>
      <td>Just had my facial/microderm treatment. It was...</td>
    </tr>
    <tr>
      <th>1426</th>
      <td>5-star</td>
      <td>This is the best laser hair removal in Toronto...</td>
    </tr>
    <tr>
      <th>1427</th>
      <td>5-star</td>
      <td>I love it here ! Yulia makes the process of la...</td>
    </tr>
    <tr>
      <th>1428</th>
      <td>5-star</td>
      <td>This place is one of Toronto's best-kept, imma...</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>5-star</td>
      <td>Yuilya is so kind and professional. She genuin...</td>
    </tr>
    <tr>
      <th>1430</th>
      <td>5-star</td>
      <td>The customer service is exceptional. I loved t...</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>5-star</td>
      <td>Yuliya is great! She's incredibly sweet and ha...</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>5-star</td>
      <td>I have never had any kind if laser treatments ...</td>
    </tr>
    <tr>
      <th>1433</th>
      <td>5-star</td>
      <td>YS Canadian Laser Spa is wonderful. Yuliya is ...</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>5-star</td>
      <td>Yuliya is amazing! I've had multiple sessions ...</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>5-star</td>
      <td>Yuliya is such a nice person and I felt so inc...</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>5-star</td>
      <td>I went for the first time to do laser hair rem...</td>
    </tr>
    <tr>
      <th>1437</th>
      <td>5-star</td>
      <td>I had a wonderful spa experience at Canadian L...</td>
    </tr>
    <tr>
      <th>1438</th>
      <td>5-star</td>
      <td>A wonderful treatment, makes you leave feeling...</td>
    </tr>
    <tr>
      <th>1439</th>
      <td>4-star</td>
      <td>I did a lot of research for different dental o...</td>
    </tr>
    <tr>
      <th>1440</th>
      <td>5-star</td>
      <td>I came from the US to do my teeth here. Wonder...</td>
    </tr>
    <tr>
      <th>1441</th>
      <td>5-star</td>
      <td>The staff is amazing, obviously customer servi...</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>5-star</td>
      <td>Gelareh did such an amazing cleaning for me th...</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>5-star</td>
      <td>Friendly staff that is both very professional ...</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>5-star</td>
      <td>Always a great place to get all your supplemen...</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>1-star</td>
      <td>WAY OVERPRICED. Go to Healthy Planet, or other...</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>4-star</td>
      <td>I'm surprised at the negative reviews. I go he...</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>1-star</td>
      <td>I've came in to this store twice now to purcha...</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>1-star</td>
      <td>Most definitely not impressed with this place....</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>2-star</td>
      <td>If a place like this can afford the rent on th...</td>
    </tr>
  </tbody>
</table>
<p>1450 rows × 2 columns</p>
</div>



# Now let's take a look at Hotels and travel:


```python
if not os.path.isfile('../csv/Hotels and travel.csv'):
    explore_and_download(base_url)

hotels = pd.read_csv('../csv/Hotels and travel.csv')
hotels
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rate</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5-star</td>
      <td>I have so many good things to say about this p...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5-star</td>
      <td>I found them to be highly skilled and an exper...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5-star</td>
      <td>Where do I even begin? This office has been so...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5-star</td>
      <td>I went in because I had toothache and needed a...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5-star</td>
      <td>Found a new dental office. This place is amazi...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5-star</td>
      <td>Dr. Carlos is always on top of things, I've ne...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5-star</td>
      <td>Dr. Carlos and the staff were very friendly. T...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5-star</td>
      <td>Love these guys! Had a chip in my tooth and no...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5-star</td>
      <td>I just found this office in Scottsdale and the...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5-star</td>
      <td>Dr. Mandap has been my dentist for many years ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5-star</td>
      <td>A bit of a drive on the freeways but this plac...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1-star</td>
      <td>Wow. Do not order from here. I ordered an item...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1-star</td>
      <td>I was searching google for diabetic shoes for ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5-star</td>
      <td>Am I reviewing the same company as everyone el...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1-star</td>
      <td>This company is terrible! My mom purchased a p...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1-star</td>
      <td>Horrible Experience. I ordered an item and req...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1-star</td>
      <td>Ordered item in April 2015. Never arrived. The...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1-star</td>
      <td>Terrible online service! I ordered one item an...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1-star</td>
      <td>I want to let everyone know to not make purcha...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1-star</td>
      <td>Horrible customer service ... Ordered an item ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1-star</td>
      <td>I purchased a calf sleeve from this company to...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1-star</td>
      <td>The website descriptions and advertising is pu...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1-star</td>
      <td>Do not deal with this company. I ordered some ...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1-star</td>
      <td>they had the best price around for the item i ...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1-star</td>
      <td>I should have looked at the reviews before ord...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1-star</td>
      <td>DON"T BUY FROM THIS COMPANY. my order never ar...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1-star</td>
      <td>There are only two positive reviews (read: shi...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1-star</td>
      <td>Take your business else where. Notice how the ...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5-star</td>
      <td>I found ActiveForever after a family incident....</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1-star</td>
      <td>Ordered a product online and paid for for expe...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2850</th>
      <td>5-star</td>
      <td>Classic old hotel with all the right upgrades ...</td>
    </tr>
    <tr>
      <th>2851</th>
      <td>5-star</td>
      <td>The Ritz in Montreal is beautiful. It is locat...</td>
    </tr>
    <tr>
      <th>2852</th>
      <td>5-star</td>
      <td>For my 300th review on Yelp....I decided to wr...</td>
    </tr>
    <tr>
      <th>2853</th>
      <td>5-star</td>
      <td>The staff at this amazing hotel go above and b...</td>
    </tr>
    <tr>
      <th>2854</th>
      <td>5-star</td>
      <td>I recently attended an executive conference at...</td>
    </tr>
    <tr>
      <th>2855</th>
      <td>5-star</td>
      <td>I have been to many Ritz all over the world an...</td>
    </tr>
    <tr>
      <th>2856</th>
      <td>5-star</td>
      <td>The Ritz was a perfect place to stay in Montre...</td>
    </tr>
    <tr>
      <th>2857</th>
      <td>3-star</td>
      <td>I brought my mum here for her 70th birthday a ...</td>
    </tr>
    <tr>
      <th>2858</th>
      <td>5-star</td>
      <td>This is a review only on the meeting room, not...</td>
    </tr>
    <tr>
      <th>2859</th>
      <td>4-star</td>
      <td>Oh the Ritz darling, what a fabulous experienc...</td>
    </tr>
    <tr>
      <th>2860</th>
      <td>4-star</td>
      <td>*This review is for my High Tea experience onl...</td>
    </tr>
    <tr>
      <th>2861</th>
      <td>5-star</td>
      <td>Beautiful hotel with great amenities. They acc...</td>
    </tr>
    <tr>
      <th>2862</th>
      <td>5-star</td>
      <td>Very nice medium sized luxury 5 star hotel wit...</td>
    </tr>
    <tr>
      <th>2863</th>
      <td>5-star</td>
      <td>As a frequent business traveler staying in top...</td>
    </tr>
    <tr>
      <th>2864</th>
      <td>5-star</td>
      <td>Beautiful hotel, perfect location (food, shopp...</td>
    </tr>
    <tr>
      <th>2865</th>
      <td>5-star</td>
      <td>After learning about the hotel's new renovatio...</td>
    </tr>
    <tr>
      <th>2866</th>
      <td>5-star</td>
      <td>Thé royal dans le "Palm court", le foyer consu...</td>
    </tr>
    <tr>
      <th>2867</th>
      <td>5-star</td>
      <td>Simply fantastic! One of the best, if not the ...</td>
    </tr>
    <tr>
      <th>2868</th>
      <td>5-star</td>
      <td>Perfection in every way! Beautiful rooms, grea...</td>
    </tr>
    <tr>
      <th>2869</th>
      <td>5-star</td>
      <td>This is the standard of Ritz Carlton. The tech...</td>
    </tr>
    <tr>
      <th>2870</th>
      <td>5-star</td>
      <td>This was the most amazing hotel experience I'v...</td>
    </tr>
    <tr>
      <th>2871</th>
      <td>5-star</td>
      <td>Recently renovated. Beautiful rooms with high ...</td>
    </tr>
    <tr>
      <th>2872</th>
      <td>5-star</td>
      <td>Hey, it's the Ritz, what can I say? This hotel...</td>
    </tr>
    <tr>
      <th>2873</th>
      <td>5-star</td>
      <td>Fabulous service. We stayed for 4 nights to ce...</td>
    </tr>
    <tr>
      <th>2874</th>
      <td>4-star</td>
      <td>The Ritz on Sherbrooke really does not look li...</td>
    </tr>
    <tr>
      <th>2875</th>
      <td>4-star</td>
      <td>Always wanted to try out a proper British high...</td>
    </tr>
    <tr>
      <th>2876</th>
      <td>5-star</td>
      <td>I have traveled to some of the most beautiful ...</td>
    </tr>
    <tr>
      <th>2877</th>
      <td>5-star</td>
      <td>Simon the concierge and victor the front desk ...</td>
    </tr>
    <tr>
      <th>2878</th>
      <td>5-star</td>
      <td>What I like about this hotel: great location; ...</td>
    </tr>
    <tr>
      <th>2879</th>
      <td>5-star</td>
      <td>OMG!!!! After being closed for 4 years for a r...</td>
    </tr>
  </tbody>
</table>
<p>2880 rows × 2 columns</p>
</div>



# Now let's classify the stars rating into two groups, 1-3 stars: negative and 4-5: positive


```python
if not os.path.isfile('../csv/Health and medical (classified).csv'):
    classify_stars('../csv/Health and medical.csv', '../csv/Health and medical (classified).csv')
if not os.path.isfile('../csv/Hotels and travel (classified).csv'):
    classify_stars('../csv/Hotels and travel.csv', '../csv/Hotels and travel (classified).csv')

health = pd.read_csv('../csv/Hotels and travel (classified).csv', index_col=0)
hotels = pd.read_csv('../csv/Health and medical (classified).csv', index_col=0)
health, hotels
```




    (        rate                                             review     class
     0     5-star  I have so many good things to say about this p...  positive
     1     5-star  I found them to be highly skilled and an exper...  positive
     2     5-star  Where do I even begin? This office has been so...  positive
     3     5-star  I went in because I had toothache and needed a...  positive
     4     5-star  Found a new dental office. This place is amazi...  positive
     5     5-star  Dr. Carlos is always on top of things, I've ne...  positive
     6     5-star  Dr. Carlos and the staff were very friendly. T...  positive
     7     5-star  Love these guys! Had a chip in my tooth and no...  positive
     8     5-star  I just found this office in Scottsdale and the...  positive
     9     5-star  Dr. Mandap has been my dentist for many years ...  positive
     10    5-star  A bit of a drive on the freeways but this plac...  positive
     11    1-star  Wow. Do not order from here. I ordered an item...  negative
     12    1-star  I was searching google for diabetic shoes for ...  negative
     13    5-star  Am I reviewing the same company as everyone el...  positive
     14    1-star  This company is terrible! My mom purchased a p...  negative
     15    1-star  Horrible Experience. I ordered an item and req...  negative
     16    1-star  Ordered item in April 2015. Never arrived. The...  negative
     17    1-star  Terrible online service! I ordered one item an...  negative
     18    1-star  I want to let everyone know to not make purcha...  negative
     19    1-star  Horrible customer service ... Ordered an item ...  negative
     20    1-star  I purchased a calf sleeve from this company to...  negative
     21    1-star  The website descriptions and advertising is pu...  negative
     22    1-star  Do not deal with this company. I ordered some ...  negative
     23    1-star  they had the best price around for the item i ...  negative
     24    1-star  I should have looked at the reviews before ord...  negative
     25    1-star  DON"T BUY FROM THIS COMPANY. my order never ar...  negative
     26    1-star  There are only two positive reviews (read: shi...  negative
     27    1-star  Take your business else where. Notice how the ...  negative
     28    5-star  I found ActiveForever after a family incident....  positive
     29    1-star  Ordered a product online and paid for for expe...  negative
     ...      ...                                                ...       ...
     2850  5-star  Classic old hotel with all the right upgrades ...  positive
     2851  5-star  The Ritz in Montreal is beautiful. It is locat...  positive
     2852  5-star  For my 300th review on Yelp....I decided to wr...  positive
     2853  5-star  The staff at this amazing hotel go above and b...  positive
     2854  5-star  I recently attended an executive conference at...  positive
     2855  5-star  I have been to many Ritz all over the world an...  positive
     2856  5-star  The Ritz was a perfect place to stay in Montre...  positive
     2857  3-star  I brought my mum here for her 70th birthday a ...  negative
     2858  5-star  This is a review only on the meeting room, not...  positive
     2859  4-star  Oh the Ritz darling, what a fabulous experienc...  positive
     2860  4-star  *This review is for my High Tea experience onl...  positive
     2861  5-star  Beautiful hotel with great amenities. They acc...  positive
     2862  5-star  Very nice medium sized luxury 5 star hotel wit...  positive
     2863  5-star  As a frequent business traveler staying in top...  positive
     2864  5-star  Beautiful hotel, perfect location (food, shopp...  positive
     2865  5-star  After learning about the hotel's new renovatio...  positive
     2866  5-star  Thé royal dans le "Palm court", le foyer consu...  positive
     2867  5-star  Simply fantastic! One of the best, if not the ...  positive
     2868  5-star  Perfection in every way! Beautiful rooms, grea...  positive
     2869  5-star  This is the standard of Ritz Carlton. The tech...  positive
     2870  5-star  This was the most amazing hotel experience I'v...  positive
     2871  5-star  Recently renovated. Beautiful rooms with high ...  positive
     2872  5-star  Hey, it's the Ritz, what can I say? This hotel...  positive
     2873  5-star  Fabulous service. We stayed for 4 nights to ce...  positive
     2874  4-star  The Ritz on Sherbrooke really does not look li...  positive
     2875  4-star  Always wanted to try out a proper British high...  positive
     2876  5-star  I have traveled to some of the most beautiful ...  positive
     2877  5-star  Simon the concierge and victor the front desk ...  positive
     2878  5-star  What I like about this hotel: great location; ...  positive
     2879  5-star  OMG!!!! After being closed for 4 years for a r...  positive
     
     [2880 rows x 3 columns],
             rate                                             review     class
     0     5-star  I have so many good things to say about this p...  positive
     1     5-star  I found them to be highly skilled and an exper...  positive
     2     5-star  Where do I even begin? This office has been so...  positive
     3     5-star  I went in because I had toothache and needed a...  positive
     4     5-star  Found a new dental office. This place is amazi...  positive
     5     5-star  Dr. Carlos is always on top of things, I've ne...  positive
     6     5-star  Dr. Carlos and the staff were very friendly. T...  positive
     7     5-star  Love these guys! Had a chip in my tooth and no...  positive
     8     5-star  I just found this office in Scottsdale and the...  positive
     9     5-star  Dr. Mandap has been my dentist for many years ...  positive
     10    5-star  A bit of a drive on the freeways but this plac...  positive
     11    1-star  Wow. Do not order from here. I ordered an item...  negative
     12    1-star  I was searching google for diabetic shoes for ...  negative
     13    5-star  Am I reviewing the same company as everyone el...  positive
     14    1-star  This company is terrible! My mom purchased a p...  negative
     15    1-star  Horrible Experience. I ordered an item and req...  negative
     16    1-star  Ordered item in April 2015. Never arrived. The...  negative
     17    1-star  Terrible online service! I ordered one item an...  negative
     18    1-star  I want to let everyone know to not make purcha...  negative
     19    1-star  Horrible customer service ... Ordered an item ...  negative
     20    1-star  I purchased a calf sleeve from this company to...  negative
     21    1-star  The website descriptions and advertising is pu...  negative
     22    1-star  Do not deal with this company. I ordered some ...  negative
     23    1-star  they had the best price around for the item i ...  negative
     24    1-star  I should have looked at the reviews before ord...  negative
     25    1-star  DON"T BUY FROM THIS COMPANY. my order never ar...  negative
     26    1-star  There are only two positive reviews (read: shi...  negative
     27    1-star  Take your business else where. Notice how the ...  negative
     28    5-star  I found ActiveForever after a family incident....  positive
     29    1-star  Ordered a product online and paid for for expe...  negative
     ...      ...                                                ...       ...
     1420  5-star  Yuliana, the owner, is professional, knowledge...  positive
     1421  1-star  Everything was great until they burned my cupi...  negative
     1422  5-star  Let me start by saying that I'm used to paying...  positive
     1423  5-star  Yuliya is the best at her job. I highly recomm...  positive
     1424  5-star  I have had about 4 session of hair removal and...  positive
     1425  5-star  Just had my facial/microderm treatment. It was...  positive
     1426  5-star  This is the best laser hair removal in Toronto...  positive
     1427  5-star  I love it here ! Yulia makes the process of la...  positive
     1428  5-star  This place is one of Toronto's best-kept, imma...  positive
     1429  5-star  Yuilya is so kind and professional. She genuin...  positive
     1430  5-star  The customer service is exceptional. I loved t...  positive
     1431  5-star  Yuliya is great! She's incredibly sweet and ha...  positive
     1432  5-star  I have never had any kind if laser treatments ...  positive
     1433  5-star  YS Canadian Laser Spa is wonderful. Yuliya is ...  positive
     1434  5-star  Yuliya is amazing! I've had multiple sessions ...  positive
     1435  5-star  Yuliya is such a nice person and I felt so inc...  positive
     1436  5-star  I went for the first time to do laser hair rem...  positive
     1437  5-star  I had a wonderful spa experience at Canadian L...  positive
     1438  5-star  A wonderful treatment, makes you leave feeling...  positive
     1439  4-star  I did a lot of research for different dental o...  positive
     1440  5-star  I came from the US to do my teeth here. Wonder...  positive
     1441  5-star  The staff is amazing, obviously customer servi...  positive
     1442  5-star  Gelareh did such an amazing cleaning for me th...  positive
     1443  5-star  Friendly staff that is both very professional ...  positive
     1444  5-star  Always a great place to get all your supplemen...  positive
     1445  1-star  WAY OVERPRICED. Go to Healthy Planet, or other...  negative
     1446  4-star  I'm surprised at the negative reviews. I go he...  positive
     1447  1-star  I've came in to this store twice now to purcha...  negative
     1448  1-star  Most definitely not impressed with this place....  negative
     1449  2-star  If a place like this can afford the rent on th...  negative
     
     [1450 rows x 3 columns])



# Let's look at the classes distribution:


```python
hotels.groupby('class')['class'].count()
```




    class
    negative     407
    positive    1043
    Name: class, dtype: int64




```python
health.groupby('class')['class'].count()
```




    class
    negative    1044
    positive    1836
    Name: class, dtype: int64



# ATENTION: Our classes are not evenly distributed, meaning that accuracy might not be a great indicator of a good performance, instead we will measure, precision and recall.


## When to Use ROC vs. Precision-Recall Curves?
### Generally, the use of ROC curves and precision-recall curves are as follows:

##### * ROC curves should be used when there are roughly equal numbers of observations for each class. 

#### * Precision-Recall curves should be used when there is a moderate to large class imbalance.

##### The reason for this recommendation is that ROC curves present an optimistic picture of the model on datasets with a class imbalance.


## As our machine learning techniques do not know how to process raw text, we need to transform it into a suitable representation. We will use the Bag of  Word model.

## We define a set of methods that together will create a pipeline of transformations:


```python
def filter_words(raw_comments):
    """ Returns a filtered list
        Removes numbers and symbols from each comment
    """
    table = str.maketrans('', '', string.punctuation + string.digits)
    stripped = [comment.translate(table) for comment in raw_comments]
    return stripped

def lowercase_all(comments):
    """ Returns a list of lowercased words
    """
    return [comment.lower() for comment in comments]

def split_words(comments):
    """ Splits each comment as a list of strings, where each string is a word
    """
    return [re.split(r'\W+', comment) for comment in comments]

def flatten_words(comments):
    """ Returns a list of strings, where each string is a word
    """
    return [word for comment in comments for word in comment]

def remove_stopwords(words):
    """ Returns a list filter out stop words
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in words if not word in stop_words]

def lemmatize(words):
    """ Lemma conversion
    """
    lem = WordNetLemmatizer()
    return [lem.lemmatize(word) for word in words]

def stemming(words):
    """ Linguistic normalization
    """
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]

```

## As we will use each word as features and eventually we want to test each classifier on a dataset other than the one it was trained on, we need to define a common set of features:


```python
# take both files and concatenate them together:
if not os.path.isfile('../csv/concat_categories.csv'):    
    cat_1 = pd.read_csv('../csv/Hotels and travel (classified).csv')
    cat_2 = pd.read_csv('../csv/Health and medical (classified).csv')
    cats = pd.concat([cat_1, cat_2], ignore_index=True)
    cats.to_csv('../csv/concat_categories.csv')
```

## Now, let's start our transformation pipeline for each of the files, we will use the concatenated files to define a vocabulary:


```python
def create_vocabulary(path_read):
    
    data = pd.read_csv(path_read)
    # strip reviews and place them in a single list
    raw_comments = list(data['review'])

    # Create vocabulary
    filtered = filter_words(raw_comments)
    splitted = split_words(filtered)
    flattened = flatten_words(splitted)
    words = lowercase_all(flattened)
    words = remove_stopwords(words)
    words = lemmatize(words)
    words = stemming(words)
    return sorted(list(set(words)))


def document_term_matrix(path_read, path_write):
   
    # call and create vocabulary
    vocabulary =  create_vocabulary('../csv/concat_categories.csv')

    data = pd.read_csv(path_read)
    # strip reviews and place them in a single list
    raw_comments = list(data['review'])

    # Document-Term Matrix
    # Count frequency of words per comment
    filtered = filter_words(raw_comments)
    comments_as_words = lowercase_all(filtered)
    comments_as_words = split_words(comments_as_words)
    
    # allocate memory for efficiency, a matrix of 
    # rows = number of comments in the file and 
    # columns = number of words in vocabulary
    dtm = np.zeros((len(list(data['review'])), len(vocabulary)), dtype=np.int8, order='F')
    
    # for each comment in all the comments in the file
    for i, comment in enumerate(comments_as_words):
        # continue pipeline per comment
        cleaned_comment = remove_stopwords(comment)
        comment_lem = lemmatize(cleaned_comment)
        comment_stem = stemming(comment_lem)
        # count frequency of words in comment 
        uniques_count = pd.DataFrame(np.array(comment_stem).reshape(-1,1), columns=['words']).groupby(['words'])['words'].count()
        
        # assign the frequency of each word to the dtm matrix:
        for word in uniques_count.index:
            dtm[i, vocabulary.index(word)] = uniques_count[word]
    # save results    
    pd.DataFrame(dtm, columns=vocabulary).to_csv(path_write)

```


```python
if not os.path.isfile('../csv/Hotels and travel dtm.csv'):
    document_term_matrix( '../csv/Hotels and travel (classified).csv', '../csv/Hotels and travel dtm.csv')

if not os.path.isfile('../csv/Health and medical dtm.csv'):
    document_term_matrix( '../csv/Health and medical (classified).csv', '../csv/Health and medical dtm.csv')
```

## It is important to notice that this strategy produces a sparse matrix, which is not very ideal for memory purposes.


```python
pd.read_csv('../csv/Hotels and travel dtm.csv', index_col=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 1</th>
      <th>aa</th>
      <th>aaa</th>
      <th>aan</th>
      <th>aat</th>
      <th>abalo</th>
      <th>abbey</th>
      <th>abbrevi</th>
      <th>abdomen</th>
      <th>abdomin</th>
      <th>...</th>
      <th>écrit</th>
      <th>égaux</th>
      <th>élaboré</th>
      <th>était</th>
      <th>étant</th>
      <th>été</th>
      <th>évident</th>
      <th>ête</th>
      <th>être</th>
      <th>über</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2850</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2851</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2852</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2853</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2854</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2855</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2856</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2857</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2858</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2859</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2860</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2861</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2862</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2863</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2864</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2865</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2866</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2867</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2868</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2869</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2870</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2871</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2872</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2873</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2874</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2875</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2876</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2877</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2878</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2879</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2880 rows × 11197 columns</p>
</div>



## Our data is finally ready for training, we will be using our first defined class to test a set of three different classifiers, we expect naive bayes to do better than the rest because of the sparsity of the data:


```python
def model_building(path_read_x, path_read_y, model_name):

    x_data = pd.read_csv(path_read_x, index_col=0)
    y_data = pd.read_csv(path_read_y, index_col=0)
    y_data = y_data['class']

    X_train, X_test_val, y_train, y_test_val = train_test_split(x_data, y_data,  test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val,  test_size=0.3, random_state=1)
    
    # configure models and leave default values as is:
    naive = naive_bayes.MultinomialNB()
    sgd = SGDClassifier(n_jobs=4)
    svmc = tree.DecisionTreeClassifier()
    
    clfs = [naive, sgd, svmc]
    # we will score the cross validation with roc_auc:
    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
    test_clfs = TestClassifiers(X_train, y_train, classifiers=clfs, scoring='roc_auc')
    test_clfs.cv_on_classifiers(folds=5)
    best_classifier = test_clfs.best_classifier()
    
    
    # get predictions with best_classifier
    test_clf = TestClassifiers(X_train, y_train, classifiers=[best_classifier])
    test_clf.cv_pred_on_classifiers(method='predict_proba')
    print(best_classifier)
    
    #curve needs binary/numeric values, transform the labels to 0 and 1
    le = preprocessing.LabelEncoder()
    le.fit(['positive', 'negative'])
    y_pred = le.transform(test_clf.preds[0])
    y_train = le.transform(np.array(y_train))
    
    #print(np.array(y_train))
    precision, recall, thresholds = precision_recall_curve(y_train, y_pred)
    #fpr, tpr, thresholds = roc_curve(np.array(y_train_trans), np.array(y_pred_trans), pos_label=1)
    #area = auc(fpr, tpr)
    f1 = f1_score(y_train, y_pred )
    # calculate precision-recall AUC
    auc_ = auc(recall, precision)
    # calculate average precision score
    ap = average_precision_score(y_train, y_pred, pos_label=1)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_, ap))
    
    plt.figure() 
    lw = 2
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='AUC (area = %0.2f)' % auc_)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (training data)',  fontsize=20)
    plt.legend(loc="lower right")
    
    return best_classifier
```

## Test a Naive Bayes classifier, a SGD classifier and a Decision Tree classifier using 5 fold cross-validation, then return the best of them over the Hotels and Travel category


```python
if not os.path.isfile('../models/MultinomialNB_model_ht.joblib'):
    best_classifier_1 = model_building('../csv/Hotels and travel dtm.csv', '../csv/Hotels and travel (classified).csv', 'model_ht')
```

    Cross-Validation using MultinomialNB Classifier
    Scores: [0.9316489  0.93571807 0.9454425  0.96894542 0.95606783] Mean score: 0.947564544410182
    Cross-Validation using SGDClassifier Classifier
    Scores: [0.92921437 0.94174616 0.93449921 0.95720721 0.9540673 ] Mean score: 0.9433468486251748
    Cross-Validation using DecisionTreeClassifier Classifier
    Scores: [0.73783393 0.75919449 0.77325119 0.77629836 0.79460784] Mean score: 0.7682371613619147
    Best classifier: 
    
    Predictions using Cross-Validation with MultinomialNB Classifier
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    f1=0.918 auc=0.942 ap=0.890
    


![png](output_29_1.png)


## The best classifier was the Naive Bayes, with an avg. precision of 89%

## We proceed to do the same for the Health and medical category:


```python
if not os.path.isfile('../models/MultinomialNB_model_hm.joblib'):
    best_classifier_2 = model_building('../csv/Health and medical dtm.csv', '../csv/Health and medical (classified).csv', 'model_hm')
```

    Cross-Validation using MultinomialNB Classifier
    Scores: [0.97476568 0.95650084 0.95445806 0.91606585 0.97116078] Mean score: 0.954590242730113
    Cross-Validation using SGDClassifier Classifier
    Scores: [0.95085316 0.9513939  0.94688777 0.92093247 0.94766883] Mean score: 0.9435472242249461
    Cross-Validation using DecisionTreeClassifier Classifier
    Scores: [0.73317712 0.72632781 0.74194905 0.77319154 0.75606825] Mean score: 0.7461427541456381
    Best classifier: 
    
    Predictions using Cross-Validation with MultinomialNB Classifier
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    f1=0.942 auc=0.951 ap=0.904
    


![png](output_31_1.png)


## Same as before, Naive Bayes is the best with avg. precision of 90.4%
## These results are looking promissing.

## Let's train the models on the training data and run them on the test data:


```python
x_data = pd.read_csv('../csv/Hotels and travel dtm.csv', index_col=0)
y_data = pd.read_csv('../csv/Hotels and travel (classified).csv', index_col=0)
y_data = y_data['class']

X_train, X_test_val, y_train, y_test_val = train_test_split(x_data, y_data,  test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val,  test_size=0.3, random_state=1)

#fit best classifier with filtered data
best_classifier_1.fit(X_train, y_train)
#produce some predictions
y_pred = best_classifier_1.predict(X_test)


#curve needs binary/numeric values, transform the labels to 0 and 1
le = preprocessing.LabelEncoder()
le.fit(['positive', 'negative'])
y_pred = le.transform(y_pred)
y_test = le.transform(y_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
f1 = f1_score(y_test, y_pred )

# calculate precision-recall AUC
auc_ = auc(recall, precision)

# calculate average precision score
ap = average_precision_score(y_test, y_pred, pos_label=1)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_, ap))
    

plt.figure() 
lw = 2
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.plot(recall, precision, color='darkorange',
         lw=lw, label='AUC (area = %0.2f)' % auc_)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for HOTELS AND TRAVEL (test_set)',  fontsize=20)
plt.legend(loc="lower right")

```

    f1=0.930 auc=0.949 ap=0.902
    




    <matplotlib.legend.Legend at 0x2321f78c240>




![png](output_33_2.png)


## We seem to be able to generalize from our training data, as we are getting about the same values in avg. precision = 90.2%

## We run the same analysis on the health and medical data


```python
x_data = pd.read_csv('../csv/Health and medical dtm.csv', index_col=0)
y_data = pd.read_csv('../csv/Health and medical (classified).csv', index_col=0)
y_data = y_data['class']

X_train, X_test_val, y_train, y_test_val = train_test_split(x_data, y_data,  test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val,  test_size=0.3, random_state=1)

#fit best classifier with filtered data
best_classifier_2.fit(X_train, y_train)
#produce some predictions
y_pred = best_classifier_2.predict(X_test)

#encode the predictions
le = preprocessing.LabelEncoder()
le.fit(['positive', 'negative'])
y_test = le.transform(y_test)
y_pred = le.transform(y_pred)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
f1 = f1_score(y_test, y_pred )

# calculate precision-recall AUC
auc_ = auc(recall, precision)

# calculate average precision score
ap = average_precision_score(y_test, y_pred, pos_label=1)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_, ap))
    

plt.figure() 
lw = 2
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.plot(recall, precision, color='darkorange',
         lw=lw, label='AUC (area = %0.2f)' % auc_)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for HEALTH AND MEDICAL (test_set)' , fontsize=20)
plt.legend(loc="lower right")
```

    f1=0.945 auc=0.953 ap=0.908
    




    <matplotlib.legend.Legend at 0x2321f7b00f0>




![png](output_35_2.png)


## The results are really good, we are reaching an average precision of over 90% in both categories. For a dataset with such imbalance this is good news.

## Let's see if our learning transfer from one category to the other.


```python
x_data = pd.read_csv('../csv/Health and medical dtm.csv', index_col=0)
y_data = pd.read_csv('../csv/Health and medical (classified).csv', index_col=0)
y_data = y_data['class']

X_train, X_test_val, y_train, y_test_val = train_test_split(x_data, y_data,  test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val,  test_size=0.3, random_state=1)

#fit best classifier with the whole training set (HEALTH AND MEDICAL)
best_classifier_2.fit(X_train, y_train)
dump(best_classifier_2, "../models/Naive_model_hm.joblib")

#produce some predictions using the other category as input
x_data = pd.read_csv('../csv/Hotels and travel dtm.csv', index_col=0)
y_data = pd.read_csv('../csv/Hotels and travel (classified).csv', index_col=0)
y_data = y_data['class']

y_pred = best_classifier_2.predict(x_data)

#encode the predictions
le = preprocessing.LabelEncoder()
le.fit(['positive', 'negative'])
y_test_ = le.transform(y_data)
y_pred_ = le.transform(y_pred)


precision, recall, thresholds = precision_recall_curve(y_test_, y_pred_)
f1 = f1_score(y_test_, y_pred_ )

# calculate precision-recall AUC
auc_ = auc(recall, precision)

# calculate average precision score
ap = average_precision_score(y_test_, y_pred_, pos_label=1)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_, ap))
    
plt.figure() 
lw = 2
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.plot(recall, precision, color='darkorange',
         lw=lw, label='AUC (area = %0.2f)' % auc_)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-Curve HOTELS AND TRAVEL (with a model trained over HEALTH AND MEDICAL)', fontsize=20)
plt.legend(loc="lower right")
```

    f1=0.933 auc=0.948 ap=0.900
    




    <matplotlib.legend.Legend at 0x232243e2358>




![png](output_37_2.png)



```python
print(classification_report(y_data, y_pred))
```

                  precision    recall  f1-score   support
    
        negative       0.91      0.84      0.87      1044
        positive       0.91      0.96      0.93      1836
    
       micro avg       0.91      0.91      0.91      2880
       macro avg       0.91      0.90      0.90      2880
    weighted avg       0.91      0.91      0.91      2880
    
    

## Great news, our model "best_classifier_2" it is generalizing well over data it has never seen before. Let's see if the same applies for "best_classifier_1":


```python
x_data = pd.read_csv('../csv/Hotels and travel dtm.csv', index_col=0)
y_data = pd.read_csv('../csv/Hotels and travel (classified).csv', index_col=0)
y_data = y_data['class']

X_train, X_test_val, y_train, y_test_val = train_test_split(x_data, y_data,  test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val,  test_size=0.3, random_state=1)

#fit best classifier with the whole training set (HOTELS AND TRAVEL)
best_classifier_1.fit(X_train, y_train)
dump(best_classifier_1, "../models/Naive_model_ht.joblib")

#produce some predictions using the other category as input
x_data = pd.read_csv('../csv/Health and medical dtm.csv', index_col=0)
y_data = pd.read_csv('../csv/Health and medical (classified).csv', index_col=0)
y_data = y_data['class']

y_pred = best_classifier_1.predict(x_data)

#encode the predictions
le = preprocessing.LabelEncoder()
le.fit(['positive', 'negative'])
y_test_ = le.transform(y_data)
y_pred_ = le.transform(y_pred)

precision, recall, thresholds = precision_recall_curve(y_test_, y_pred_)
f1 = f1_score(y_test_, y_pred_ )

# calculate precision-recall AUC
auc_ = auc(recall, precision)

# calculate average precision score
ap = average_precision_score(y_test_, y_pred_, pos_label=1)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_, ap))
    
plt.figure() 
lw = 2
fig = plt.gcf()
fig.set_size_inches(12, 10)
plt.plot(recall, precision, color='darkorange',
         lw=lw, label='AUC (area = %0.2f)' % auc_)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ROC HEALTH AND MEDICAL (with a model trained over HOTELS AND TRAVEL)', fontsize=20)
plt.legend(loc="lower right")
```

    f1=0.973 auc=0.978 ap=0.957
    




    <matplotlib.legend.Legend at 0x232854acfd0>




![png](output_40_2.png)



```python
print(classification_report(y_data, y_pred))
```

                  precision    recall  f1-score   support
    
        negative       0.96      0.89      0.93       407
        positive       0.96      0.99      0.97      1043
    
       micro avg       0.96      0.96      0.96      1450
       macro avg       0.96      0.94      0.95      1450
    weighted avg       0.96      0.96      0.96      1450
    
    

## Even better results, definetly this is due the amount of data, as the Hotels and Travel category contains twice more samples than the Health and Medical category. I would recommend then to deploy the model trained over Hotel and Travel.
