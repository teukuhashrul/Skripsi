# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:39:54 2019

@author: TEUKU HASHRUL
 """
 
 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
dataset = pd.read_csv('hasileksperimen1.csv')  


year_meanRating = dataset[['year', 'metascore']].groupby('year').count()
plt.bar(year_meanRating.index , year_meanRating['metascore'],linewidth = 1.2,edgecolor='black',alpha=0.5)
plt.xlabel('Year',fontweight='bold')
plt.ylabel('Rating',fontweight='bold')
plt.title('Mean Film Rating Every Year ', fontweight='bold')




#function for scatter plot
def scatterplot(xVar , yVar, xName , yName): 
    plt.scatter(xVar , yVar)
    plt.ylabel(yName)
    plt.xlabel(xName)
    plt.title("Scatter plot perbandingan {} dan {}".format(xName , yName), fontweight = "bold")
    # only choose 1 , show() to the console , savefig save to the folder 
    plt.show()
    #plt.savefig("scatterp-"+xName+"-"+yName+".jpg")
    plt.clf()


scatterplot(dataset['votes'] , dataset['revenue'] , 'Votes' , 'Revenue')
scatterplot(dataset['rating'] , dataset['revenue']  , 'Rating' , 'Revenue' )
scatterplot(dataset['metascore'] , dataset['revenue'] , 'Metascore' , 'Revenue')
scatterplot(dataset['year'] , dataset['revenue'] , 'Year' , 'Revenue')
scatterplot(dataset['runtime'] , dataset['revenue'] , 'Runtime' , 'Revenue')
# analisis selera peminat dan reviewer

normTest = dataset[['votes', 'metascore', 'rating']] 
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
votesNorm = mm_scaler.fit_transform( normTest[['votes']]) 
metascoreNorm = mm_scaler.fit_transform(normTest[['metascore']])
ratingNorm = mm_scaler.fit_transform(normTest[['rating']])
scatterplot(votesNorm , metascoreNorm, 'Votes','Metascore')
scatterplot(votesNorm , ratingNorm, 'Votes','Rating')
scatterplot(metascoreNorm , ratingNorm, 'Metascore','Rating')
dataset.to_csv('hasileksperimen2.csv')
 


# =============================================================================
# 
# =============================================================================
    
# use pearson correlation to find correlation between other feature and revenue 
from scipy.stats.stats import pearsonr 
pcor_runtime = pearsonr(dataset['runtime'] , dataset['revenue'])   
pcor_rating  = pearsonr(dataset['rating'] , dataset['revenue'])
pcor_votes   = pearsonr(dataset['votes'] , dataset['revenue'])
pcor_metascore = pearsonr(dataset['metascore'] , dataset['revenue'])
pcor_year    = pearsonr(dataset['year'], dataset['revenue']) 

sc_runtime = pcor_runtime[0]

pcor_dataset = pd.DataFrame({"name":['Runtime' , 'Rating' , 'Votes' , 'Metascore' , 'Year'],
                             "score":[pcor_runtime[0],pcor_rating[0],pcor_votes[0],pcor_metascore[0],pcor_year[0]]})
# sort by score 
pcor_dataset = pcor_dataset.sort_values('score')

def barchart(xVar , yVar , xName , yName , title): 
    plt.bar(xVar , yVar,linewidth = 1.2,edgecolor='black',alpha=0.5)
    plt.xlabel(xName,fontweight='bold')
    plt.ylabel(yName,fontweight='bold')
    plt.title(title , fontweight='bold')
    #set axis if needed
    plt.ylim(0,1)
    
    #plt.savefig("barchart-"+title+".jpg") 
    
    #draw line 
    plt.axhline(y=0.0, color='r' , linestyle='-')
    plt.show()
    plt.clf

barchart(pcor_dataset['name'] , pcor_dataset['score'] , "Features" , "Pearson Score" , "Barchart Pearson Correlation Revenue with Other Features Comparison")
    
    
# =============================================================================
# Linear Regression Example  
# =============================================================================


#barchart(datasetfeatures['features'] , datasetfeatures['score'], 'Features' , 'Accuracy Score' , 'Barchart comparing linear Regression score')   
x= dataset[['votes']]
y = dataset[['revenue']]



x_arr = np.array(x.values)
y_arr = np.array(y.values)

xtrain , xtest , ytrain , ytest = train_test_split(x_arr,  y_arr , test_size =0.2, random_state =False)

linearModel = LinearRegression() 
linearModel.fit(xtrain , ytrain)

linearModel.intercept_
linearModel.coef_
ypred = linearModel.predict(xtest) 

#ARRAY OF SQUARED ERROR
se = [pow(ypred - ytest,2 )]
nd = np.array(se) 
nd = nd.ravel()

plt.title("Linear Regression Revenue Prediction Squared Error \n using votes")
plt.boxplot(np.array(nd), vert = False)
plt.xlabel("Squared Errors (Ytrue - Ypred) \n"
           +"Q1 :"+ str(np.quantile(nd,0.25))+ " \n"
          +"Q2 :"+ str(np.quantile(nd,0.5)) + " \n"
          +"Q3 :"+ str(np.quantile(nd,0.75)))

se_selected = nd[nd<20000]

# =============================================================================
# SCORE Linear Regression 
# =============================================================================
score  = linearModel.score(xtest , ytest)

rmse_linear = np.sqrt(mean_squared_error(ytest,ypred))
r2_linear = r2_score(ytest,ypred)
print("rmse linear :" + str(rmse_linear))
print("r2 linear :"+ str(r2_linear))

votestest = xtest.ravel()
revenuetest = ytest.ravel() 
revenuepred = ypred.ravel()
linearPredComparison = []
for i in range(0, len(ytest)):
    listNow = []
    listNow.append(votestest[i])
    listNow.append(revenuetest[i]) 
    listNow.append(revenuepred[i]) 
    
    linearPredComparison.append(listNow)
    
#intercept 
intercept = linearModel.intercept_[0]
coefficient = linearModel.coef_[0][0]
#
print("linear pred function using votes > REVENUE(votes) = " + str(intercept) + " + " + str(coefficient) + "(Votes)" )
#extends the x axis 
plt.figure(figsize=(8,3))
plt.grid(zorder=3)
plt.scatter(xtest , ytest, zorder=3)

plt.ylabel('revenue')
plt.xlabel('votes')

plt.title("Scatter plot perbandingan {} dan {}".format('votes' , 'revenue'))
# only choose 1 , show() to the console , savefig save to the folder 
plt.plot(xtest, ypred, color='red', linewidth=1)
plt.show()

#plt.savefig("linearregressionplot")
plt.clf()


# =============================================================================
#   Polynomial Regression 
# =============================================================================
import operator
# transforming the data to include another axis
xtest = xtest
xtrain = xtrain
ytrain = ytrain
ytest = ytest



degree_num =2
polynomial_features= PolynomialFeatures(degree=degree_num)

x_poly_train = polynomial_features.fit_transform(np.array(xtrain))
x_poly_test = polynomial_features.fit_transform(np.array(xtest)) 


model = LinearRegression()
model.fit(x_poly_train, ytrain)
y_poly_pred = model.predict(x_poly_test)

#SQUARED ERROR
se = [pow(y_poly_pred - ytest, 2)]

nd_se = np.array(se)
nd_se = nd_se.ravel()

se_poly_selected = nd_se[nd_se < 20000]


# boxplot se polynom
plt.title("Polynomial Regression Revenue Prediction Squared Error \n using votes")
plt.boxplot(nd_se, vert = False)
plt.xlabel("Squared Errors (Ytrue - Ypred) \n"
           +"Q1 :"+ str(np.quantile(nd_se,0.25))+ " \n"
          +"Q2 :"+ str(np.quantile(nd_se,0.5)) + " \n"
          +"Q3 :"+ str(np.quantile(nd_se,0.75)))

# comparison SE linear and polynom
q2_linear_se =  "{:.2f} ".format( np.quantile(nd , 0.5) )
q2_poly_se   =  "{:.2f} ".format( np.quantile(nd_se, 0.5) )

fig,ax = plt.subplots() 
plt.title('Perbandingan Squared Error Prediksi Revenue menggunakan Votes')
plt.ylabel('Regresi')
plt.xlabel('Distribusi nilai SE')
ax.set_yticklabels(['linear \n (Q2:'+ q2_linear_se+')', 'polynomial \n (Q2:'+ q2_poly_se +')'])
ax.boxplot([se_selected, se_poly_selected] , vert=False) 


# print all the coefficient :  if degree is2 there is 3 coefficient a + bx + cx2
model.coef_
arrofcoef = model.coef_
rmse_polynom = np.sqrt(mean_squared_error(ytest,y_poly_pred))
r2_polynom = r2_score(ytest,y_poly_pred)
error = model.intercept_

#extends the x axis 
plt.figure(figsize=(7,3))

#draw grid  
plt.grid(zorder=0)
plt.scatter(x, y, s=10 , zorder=3)
plt.title("Scatter plot perbandingan {} dan {}".format('votes' , 'revenue'))
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()


# =============================================================================
# experiment detecting outlier 
# =============================================================================

from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters = 2 , random_state = 0)
kmeans_model.fit(dataset[['revenue']])




# append labales of to the data
y_pred = kmeans_model.predict(dataset[['revenue']])

dataset['labels'] = y_pred


# ambil yang outlier 
posoutlier = dataset[dataset['labels'] == 1] 
notoutlier = dataset[dataset['labels'] == 0]


# get all centroid centers 
centroid = pd.DataFrame(kmeans_model.cluster_centers_)
#draw grid  
plt.grid(zorder=0)
plt.scatter(notoutlier['votes'] , notoutlier['revenue'] , marker ='s' , s =50 , c='lightgreen', label ='cluster1')
plt.scatter(posoutlier['votes'] , posoutlier['revenue'] , marker = 'v' , s =50 , c = 'red' , label ='cluster2') 
plt.scatter(centroid[0] , centroid[1] , c = 'yellow' , s =75 , edgecolors='black' )
plt.legend(scatterpoints = 1)

plt.show()
plt.clf()



# =============================================================================
# create bar chart 
# =============================================================================
