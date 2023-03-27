#DataBases:
#pandas.read_sql(sql_querry,connection, index_cols=None, columns=None)
#import pymysql
#con=pymysql.connect(host='localhost',user='root',password=pw,databse='employees')
#employees=pd.read_sql('select * from employees.employees',con)
#con.close()

#Pandas Input and Output Methods:
#pandas.read_pickle(filepath) or DataFrame.to_pickle(path)
#pandas.read_json()
#pandas.read_table(filepath)

def func(*args):
    print(args)

func(1,2,3,4)

def key_word(**kwargs):
    print(kwargs)

key_word(name='yogesh', age=20)

#insert(index,element_to_insert)
list_1.insert(0,100)
list_1

####Numpy:

#numpy.array(object,dtype,...)
#for eg: numpy.array(object, np.longdouble)
matrix=np.array([[5,8,9],[10,11,12]])
matrix

#numpy.randon.rand(row,cols)
#numpy.random.randint(low,high,shape,dtype)
z=np.random.randn(3, 3,3)
z

matrix1 = np.random.randint(1,10, (3, 3))
matrix1
#matrix1.shape

#Array Manipulation function:
array_2d.reshape(2,6)

#Flatten: convert array in 1d array
np.ndarray.flatten(array_2d)

#numpy.mean(array,axis)
#numpy.amin(array,axis)
#numpy.amax(array,axis)

#IO Functions
#window filepath: 'C:\Desktop\tfl-daily-cycle-hires.txt'
x=np.loadtxt('tfl-daily-cycle-hires.txt',delimiter=',', usecols=(1), skiprows=1)
x

rn=np.random.randint(3,10,(2,2))
rn

#pandas----------------------------------------------------------------------------

# write to a csv file without an index
#bank_df.to_csv('sample_output_index.csv', index = True)

countries.set_index('Rank')

countries.insert(0,column='age',value=[18,28,28,29,30,31,25,26,28,35])
countries

countries.drop(labels=['age'],axis=1, inplace=True)
countires

countries1.sort_values(by='Rank')

countries1.iloc[0:5,0:3]

countries.loc['India']

#between
countries1[countries1['Rank'].between(1,5)]

# Drop duplicates
#bank_df.drop_duplicates(subset = ["Last Name"], inplace = True)

#where
countries1.where(countries1['Rank']<4)

#Feature Engineering and handling missing datasets:
# We can also indicate which columns we want to drop NaN from
#employee_df.dropna(how = 'any', inplace = True, subset = ['MonthlyIncome', 'PercentSalaryHike'])

countries1.where(countries1['Rank']<4).dropna(how='any')

countries1["Rank"]=countries1["Rank"].astype("float64")
countries1.info()

countries1.set_index(keys=['Date','Rank'])

countries1['Region'].unique() #or countries1['Region'].isnull()

countries1.groupby('Rank')['Population']

countries.loc[5,'Country / Dependency']

countries.iloc[2,4]

countries['Region'].isin(['Asia','America'])

bank_all_df = pd.concat([bank1_df, bank2_df], ignore_index = True)
bank_all_df

# We can perform concatenation and also use multi-indexing dataframe as follows:

# Let's merge all data on 'Bank Client ID'
#bank_all_df = pd.merge(bank_all_df, bank_salary_df, on = 'Bank Client ID')
#bank_all_df

bank_all_df = pd.concat([bank1_df, bank2_df], keys = ["Customers Group 1", "Customers Group 2"])
bank_all_df

countries.rename(columns={'Country / Dependency':'Country'})

countries.drop(labels=['Date'],axis=1)

countries['Date']=countries['Date'].astype("datetime64")
pd.to_datetime(countries['Date'],format='%d/%m/%Y')


date_1 = dt.date(2020, 3, 22)
date_2 = dt.date(2020, 4, 22)
date_3 = dt.date(2020, 5, 22)
dates_list = [date_1, date_2, date_3]
dates_index = pd.DatetimeIndex(dates_list)
sales = [50, 55, 60]
sales = pd.Series(data = sales, index = dates_index) # Series constructor method
sales

# you can also define a range of dates as follows:
my_days = pd.date_range(start = "2020-01-01", end = "2020-04-01", freq = "M")
my_days

# you can offset (shift) all dates by days or month as follows
#avocado_df.index = avocado_df.index + pd.DateOffset(months = 12, days = 30)

# Once you have the index set to DateTime, this unlocks its power by performing aggregation
# Aggregating the data by year (A = annual)
#avocado_df.resample(rule = 'A').mean()

# Aggregating the data by month (M = Month)
#avocado_df.resample(rule='M').mean()

# You can obtain the maximum value for each Quarter end as follows: 
#avocado_df.resample(rule='Q').max()

#countries[['Region','Population']].groupby(by='Region').agg([sum,max])
countries[['Region','Population']].groupby(by='Region').sum()

#pandas.pivot_table(data,values,aggfunc,index,columns,fill_vale,...)
pd.pivot_table(data=countries,index='Country / Dependency', values='Population',columns='Region',aggfunc=sum)

df['salary']=df['salary'].fillna(value=df['salary'].mean())
df

emp=pd.read_excel("employees_hr.xls",sheet_name='employees')
emp

#join:
emp.join(dept.set_index('id'), on='dept_id') # join on dept_id because dept_id common in both sheet

#pandas.merge(leftdataframe,rightdataframe,left_on(key to join on from df),right_on(key to join on from df),how='inner' or etc,..)
pd.merge(left=emp,right=dept,left_on='dept_id',right_on='id',how='inner')

--Matplotlib-------------------------------------------------------------------------------------------------------------------------------------------
#add_axes(), figsize=(width,height), react=[position from left,position from bottom,width,height]
fig=plt.figure(figsize=(5,5))
axes1=fig.add_axes(rect=[0,0,1,1])
axes1.set_ylim(0,10)
#twinx()
axes2=axes1.twinx()
axes2.set_ylim(0,5)
axes1.set_title('Graph')
axes1.set_xlabel('x')
axes1.set_ylabel('y')

#plt.subplots(nrows,ncols,squeeze=True,...)
plt.subplots(1,2,tight_layout=True)

fig,ax=plt.subplots(1,2)
ax[0].set_title("first")


fig,(ax1,ax2)=plt.subplots(1,2)
ax1.set_title('first')
ax2.set_title('second')

fig,[[ax1,ax2],[ax3,ax4]]=plt.subplots(2,2,tight_layout=True)
ax1.set_title('first')
ax2.set_title('second')
ax3.set_title('third')
ax4.set_title('fourth')

#df_g['Population']=df_g['Population'].str.replace(',','').astype("float64")
#df_g.sort_values(by='Population',inplace=True)
#plot area
fig=plt.figure(figsize=(5,5))
#add axes
axes=fig.add_axes(rect=[0,0,1,1])
axes.set_title('Top 10 Population Country')
axes.set_xlabel('Region')
axes.set_ylabel('Total Population')

#bar plot, #plt.bar(x(x-cordinate),height,width,bottom,align,data)
axes.bar(x=df_g['Region'], height=df_g['Population'],color='red',edgecolor='black')
#horizontal bar plot-> plt.barh(y(y-cordinate,width))
plt.show()

#legend(handles)
fig=plt.figure()
ax=fig.add_axes(rect=[0,0,1,1])
line1, = ax.plot([1, 2, 3], label='label1')
line2, = ax.plot([1, 2, 3], label='label2')
ax.legend(handles=[line1, line2])

tf1['Day']=tf1['Day'].astype('datetime64')

tf1_y=tf1_y.groupby(by='Year').sum().reset_index()

#Important:
#Plot, plt.plot(x,y)
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.plot(tf1_y['Year'],tf1_y['Number of Bicycle Hires'], tf1_y['Number of Bicycle Hires']*2,color='b',alpha=0.8,marker='o',markersize=10,linestyle=(0,(1,10)))
#or axes.plot('Year','Number of Bicycle Hires',data=tf1_y.head(10))
plt.show()

#display hide columnns or all columns
pd.set_option('display.max_columns',None)
players.head(5)

#plt.scatter(x,y,marker,edgecolors,linewidths,alpha,...)
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.set_title('Overall ability wages')
axes.set_xlabel('overall ability')
axes.set_ylabel('wages')

#scatter
axes.scatter(x=players['overall'],y=players['wage_eur'],s=players['value_eur']/150000)

#Histogram
#plt.hist(x,bins,range,density,weights,cumlative,histtype,align,orientation,label,data,...)
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
axes.set_xlabel('Number of Players')
axes.set_ylabel('Overall Score(bins)')
plt.xticks([0,25,50,75,100],[0,25,50,75,100]) #or plt.xticks([0,25,50,75,100],[0,25,50,75,100],rotation='vertical')

#histogram
axes.hist(players['overall'], bins=[0,25,50,75,100],edgecolor='black',orientation='vertical')

#pie chart:
labels=['India','Argentina','Australia','America','England']
myexplode=[0.2,0,0,0]
plt.pie(x=players['height_cm'].head(5),labels=labels)
plt.legend()
plt.show()

#BoxPlot:
#plt.boxplot(x,notch,labels,positions,bootstrap,data,...)
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])

#boxplot
axes.boxplot(players['overall'],labels=['overall score'],showfliers=False)
plt.show()

#BoxPlot:
#plt.boxplot(x,notch,labels,positions,bootstrap,data,...)
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])

#boxplot
axes.boxplot(data,labels=['inter','dortmund'],showfliers=False)
plt.show()

#vionlinplot:
#plt.violinplot(datasets,vert,width,data,...)
fig=plt.figure()
axes=fig.add_axes([0,0,1,1])
#violinplot
axes.violinplot(data)
axes.boxplot(data)
plt.xticks([1,2],['inter','dortmund'])
plt.show()

#note: side by side bar chart:
X = ['Group A','Group B','Group C','Group D']
Ygirls = [10,20,20,40]
Zboys = [20,30,25,30]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Girls')
plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Boys')
  
plt.xticks(X_axis, X)
plt.xlabel("Groups")
plt.ylabel("Number of Students")
plt.title("Number of Students in each group")
plt.legend()
plt.show()

#Q:What is the purpose of a violin plot?
#ans: A violin plot is a hybrid of a box plot and a kernel density plot, which shows peaks in the data. It is used to visualize the distribution of numerical data. 
#Unlike a box plot that can only show summary statistics, violin plots depict summary statistics and the density of each variable.

--Seaborn-------------------------------------------------------------------

#categroical plot:
#sns.barplot(x,y,data,order,hue,saturation,...)
fig=plt.figure(figsize=(8,8))
sns.barplot(x=countries['Region'],y=countries['Population'],ci=None,estimator=sum,hue=countries['Region'],dodge=False) # or sns.barplot(x='Region',y='Population',data=countries,ci=None)
plt.legend(loc='center')
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,tight_layout=True,figsize=(8,8))
sns.barplot(x='Region',y='Population',data=countries,hue='Region',ax=ax1,ci=None,dodge=False)

#countplot:
#sns.countplot(x,y,hue_order,data,color,orient,dodge,ax,...)
sns.countplot(x=countries['Region'])

#boxplot
plt.figure(figsize=(8,8))
sns.boxplot(y=players['overall'],x=players['league_name'])

#violinplot
fig=plt.figure(figsize=(8,8))
sns.violinplot(y=players['overall'],x=players['league_name'])

#strip plot:
fig=plt.figure(figsize=(8,8))
sns.stripplot(y=players['overall'],x=players['league_name'],color='black')
sns.boxplot(y=players['overall'],x=players['league_name'])

#pointplot
#sns.pointplot()
sns.pointplot(x='overall',hue='preferred_foot',y='league_name',data=players.head(100))

#relational Plot:
#line Plot
fig=plt.figure(figsize=(10,10))
sns.lineplot(x='Date',y='Close',data=stocks,hue='Symbol',linestyle='--')

#relplot()
sns.relplot(x='Date',y='Close',data=stocks,row='Symbol',kind='line',height=5,aspect=1)

#scatterplot:
sns.get_dataset_names()

#scatterplot
fig=plt.figure(figsize=(7,7))

sns.scatterplot(x='body_mass_g',y='flipper_length_mm',data=penguins,hue='island')

#Distribution Plots:
#sns.rugplot(x,yaxis,ax,data,....)
fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0,0,1,1])
ax.set_ylim(165,235)
sns.rugplot(x='body_mass_g',y='flipper_length_mm',data=penguins)
sns.scatterplot(x='body_mass_g',y='flipper_length_mm',data=penguins)

#hisplot:
sns.histplot(x='body_mass_g',data=penguins,hue='sex')

sns.histplot(x='body_mass_g',data=penguins,y='sex')

#kde(kernal density estimation:)
sns.kdeplot(x='body_mass_g',y='flipper_length_mm',data=penguins,hue='island')

sns.kdeplot(x='body_mass_g',y='flipper_length_mm',data=penguins,hue='island',fill=True)
sns.scatterplot(x='body_mass_g',y='flipper_length_mm',data=penguins)

#ecdfplot:
sns.ecdfplot(x='body_mass_g',data=penguins,hue='sex')
plt.grid(True)

#displot:
sns.displot(x='body_mass_g',data=penguins,col='island')

sns.displot(x='body_mass_g',data=penguins,col='island',kind='kde')

#Regression Plot:
#correlation coefficeint is measure of correlation and is between is -1 and 1
penguins.corr()

#sns.regplot(x,y,data,...)
fig=plt.figure(figsize=(7,7))

sns.regplot(x='flipper_length_mm',y='bill_depth_mm',data=penguins)

#lmplot
sns.lmplot(x='flipper_length_mm',y='body_mass_g',data=penguins,ci=None,height=5,aspect=1.5,row='island',col='sex')

#Matrix Plot:
#heatmap:
#sns.heatmap(player_matrix,annot=True,cmap='Y1GnBu')

#clustermap
#sns.clustermap(player_matrix,annot=True,cmap='Y1GnBu')

Multi Plot Grids:
#FacetGrid:
g=sns.FacetGrid(data=penguins,col='island',row='species')
g.map_dataframe(sns.histplot,x='body_mass_g')

g=sns.FacetGrid(data=penguins,col='island',row='species')
g.map_dataframe(sns.scatterplot,x='body_mass_g',y='flipper_length_mm')

#pairplot:
sns.pairplot(penguins)

sns.pairplot(penguins,kind='kde',corner=True,hue='island')

#PairGrid:
g=sns.PairGrid(penguins)
g.map(sns.histplot)

g=sns.PairGrid(penguins)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)

#jointplot:
sns.jointplot(data=penguins,x='bill_length_mm',y='body_mass_g',kind='reg')

sns.jointplot(data=penguins,x='bill_length_mm',y='body_mass_g',kind='reg').plot_joint(sns.kdeplot)

#JointGrid
g=sns.JointGrid(data=penguins,x='bill_length_mm',y='body_mass_g')
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.rugplot,height=0.15)

#or
g=sns.JointGrid()
sns.scatterplot(data=penguins,x='bill_length_mm',y='body_mass_g',ax=g.ax_joint)
sns.histplot(data=penguins,x='body_mass_g',ax=g.ax_marg_x)
sns.kdeplot(data=penguins,y='flipper_length_mm',ax=g.ax_marg_y)

#set style and theme
#sns.set_style('whitegrid')
#sns.set_theme(style='ticks',palette='pastel')
-------------------------------------------------------------------------------------------------------
Practice:
  
df_newww.interpolate()

arr1.argmax() # return index of maxium number in array

np.sqrt(ar1) #or np.exp(ar1) , np.max(ar1), np.sin(ar1), np.log(ar1),

#Indexing by loc, df.loc[rowname,colname]
df.loc['B','W':'Y']

x=np.arange(0,18,4)
y=x**2
fig=plt.figure(figsize=(8,4)) # or fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(8,2))
ax=fig.add_axes([0,0,1,1])
ax.set_title('parabola equation')
ax.set_xlabel('x-value')
ax.set_ylabel('y=x2')
ax.plot(x,y,'b',label='x vs y')
ax.plot(y,x,'r',label='y vs x')
ax.legend(loc=5) #or ax.legend(loc='upper right')
plt.show()

#to save figure in image
fig.savefig('mypicture.png',dpi=200)

#Plot Appearance:
x=np.random.rand(5)
y=np.log(x)
fig=plt.figure()
ax1=fig.add_axes([0,0,1,1])
ax1.hist(x,bins=30)
ax1.plot(x,y,color='r',linewidth=1,alpha=0.5,linestyle='--',marker='o',markersize=10,markerfacecolor='blue',markeredgewidth=2,markeredgecolor='black')

#pairplot make every combination of numerical columns comparision:
sns.pairplot(tips,hue='sex',palette='coolwarm') #hue take categorical column

#rugplot take single colum like distplot:
sns.rugplot(tips['total_bill']) # below figure show 10-20 bills frequency high.

#countplot count no of occurences:
sns.countplot(x='sex',data=tips)

sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker') #below dots are outliers.

sns.violinplot(x='day',y='total_bill',data=tips,hue='smoker',split=True)

tc=tips.corr()

sns.heatmap(tc,annot=True,cmap='coolwarm') #tell about correlation, use where we visualize correlation.

fp=flights.pivot_table(index='month',columns='year',values='passengers')
fp

sns.heatmap(fp,cmap='coolwarm',linecolor='white',linewidth=1)

#clustermap:
sns.clustermap(fp,cmap='coolwarm') #cluster information to try to show row and column similar to each other.

#lmplot used for plot very simple linear model plot:
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','v'],scatter_kws={'s':100})
plt.title("Important Graph")

sns.lmplot(x='total_bill',y='tip',data=tips,col='sex',row='time',hue='sex',aspect=0.4,height=8)

# 95% confidence interval in t and normal:

#in t-statistics:


import numpy as np
import scipy.stats as st
  
# define sample data
gfg_data = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 
            3, 4, 4, 5, 5, 5, 6, 7, 8, 10]
  
# create 90% confidence interval
st.t.interval(alpha=0.90, df=len(gfg_data)-1,
              loc=np.mean(gfg_data),
              scale=st.sem(gfg_data))

#in normal distributioin:

import numpy as np
import scipy.stats as st
  
# define sample data
gfg_data = np.random.randint(5, 10, 100)
  
# create 90% confidence interval
# for population mean weight
st.norm.interval(alpha=0.90,
                 loc=np.mean(gfg_data),
                 scale=st.sem(gfg_data))

----statistics----------------------------------------------------------------------------

#confidence interval for two difference of two means dependent sample formula= d+or-tn-1,@/2*sd(standard deviation)/root of n.
#for independent samples:

#Standard deviation is the spread of a group of numbers from the mean. The variance measures the average degree to which each point differs from the mean. While standard deviation is the square root of the variance, variance is the average of all data points within a group.

#

#Normal Distrubution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
  
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-20, 20, 0.01)
  
# Calculating mean and standard deviation
mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)
  
plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import statistics
  
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-20, 20, 0.01)
  
# Calculating mean and standard deviation
mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)
  
plt.plot(x_axis, t.pdf(x_axis, 4)) # 4 is degree of freedom.
plt.show()

#calculate p value:
#scipy.stats.t.sf(abs(t_score), df=degree_of_freedom
#Parameters:
#t_score: It signifies the t-score
#degree_of_freedom: It signifies the degrees of freedom

import scipy.stats
  
# find p-value for two-tailed test
scipy.stats.t.sf(abs(1.36), df=33)*2  

#Example 1: Independent Two Sample t-Test in Pandas

import pandas as pd
from scipy.stats import ttest_ind

#create pandas DataFrame
df = pd.DataFrame({'method': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                              'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
                   'score': [71, 72, 72, 75, 78, 81, 82, 83, 89, 91, 80, 81, 81,
                             84, 88, 88, 89, 90, 90, 91]})

#view first five rows of DataFrame
#df.head()

#  method  score
#0      A     71
#1      A     72
#2      A     72
#3      A     75
#4      A     78

#define samples
group1 = df[df['method']=='A']
group2 = df[df['method']=='B']

#perform independent two sample t-test
ttest_ind(group1['score'], group2['score'])

#Regression analysis singe variable:

import numpy as np
import matplotlib.pyplot as plt
  
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)
  
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
  
    # predicted response vector
    y_pred = b[0] + b[1]*x
  
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
  
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
  
    # function to show plot
    plt.show()
  
def main():
    # observations / data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
  
    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
  
    # plotting regression line
    plot_regression_line(x, y, b)
  
if __name__ == "__main__":
    main()
    
    #Multiple linear regression:

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
  
# load the boston dataset
boston = datasets.load_boston(return_X_y=False)
  
# defining feature matrix(X) and response vector(y)
X = boston.data
y = boston.target
  
# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)
  
# create linear regression object
reg = linear_model.LinearRegression()
  
# train the model using the training sets
reg.fit(X_train, y_train)
  
# regression coefficients
print('Coefficients: ', reg.coef_)
  
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))
  
# plot for residual error
  
## setting plot style
plt.style.use('fivethirtyeight')
  
## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  
## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
  
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
## plotting legend
plt.legend(loc = 'upper right')
  
## plot title
plt.title("Residual errors")
  
## method call for showing the plot
plt.show()

#F-Statistics:

import numpy as np
import scipy.stats
  
# Create data
group1 = [0.28, 0.2, 0.26, 0.28, 0.5]
group2 = [0.2, 0.23, 0.26, 0.21, 0.23]
  
# converting the list to array
x = np.array(group1)
y = np.array(group2)
  
# calculate variance of each group
print(np.var(group1), np.var(group2))
  
def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = x.size-1
    dun = y.size-1
    p_value = 1-scipy.stats.f.cdf(f, nun, dun)
    return f, p_value
  
# perform F-test
f_test(x, y)

#Distribution:

#a distribution is a function that shows the possoble values for a varaiable and how often they occur.

#Normal Distribution-> Gaussian Distribution, mean=median=mode, no skew.

#Standard Normal Distribution: z=x-u/standard deviation.
#standard variance= root(square of (x=u))/N

#standard error= sigma(standard deviation)/root n


#confidence interval for two difference of two means dependent sample formula= d+or-tn-1,@/2*sd(standard deviation)/root of n.
#for independent samples:

#Standard deviation is the spread of a group of numbers from the mean. The variance measures the average degree to which each point differs from the mean. While standard deviation is the square root of the variance, variance is the average of all data points within a group.

#

#Population vs sample:
#Population is All interest N items,(population is hard to contsruct and hard to observe)
#sample is subset of population , n items(sample is easy to oberve and construct)

# sample is less time consuming and less costly
# sample is two characterstics:
# Randomness :random sample is collected when each member of the sample is chosen from the population strictly by chance.
#Representatives:sample is subset of population that accurately reflects the members of entire population.

#ratio has true zero and interval does not have true zero

#Pareto chart:
# Add cumulative percentage column
#df["cum_percentage"] = round(df["error"].cumsum()/df["error"].sum()*100,2)

#Plotting:
#fig, ax = plt.subplots(figsize=(22,10))

# Plot bars (i.e. frequencies)
#ax.bar(df.index, df["error"])
#ax.set_title("Pareto Chart")
#ax.set_xlabel("Medication Error")
#ax.set_ylabel("Frequency");

# Second y axis (i.e. cumulative percentage)
#ax2 = ax.twinx()
#ax2.plot(df.index, df["cum_percentage"], color="red", marker="D", ms=7)
#ax2.axhline(80, color="orange", linestyle="dashed")
#ax2.yaxis.set_major_formatter(PercentFormatter())
#ax2.set_ylabel("Cumulative Percentage");
    
  
  #visulaization of date:
#frequency distribution table	A table that represents the frequency of each variable.
#frequency	Measures the occurrence of a variable.

#In bar chart we use frequency and in In pie chart we use relative frequency(% of total frequency)(in %)

#Pareto diagram	A special type of bar chart, where frequencies are shown in descending order. There is an additional line on the chart, showing the cumulative frequency.

#cummalitve frequency is sum of relatove frequency(frequecny/total frequency).
#pareto diagram show how subtotals change with each additional category.

#histogram:
#histogram	A type of bar chart that represents numerical data. It is divided into intervals (or bins) that are not overlapping and span from the first observation to the last. The intervals (bins) are adjacent - where one stops, the other starts.
#bins (histogram)	The intervals that are represented in a histogram.
#difference between histogram and bar :
# Bar chart plot with frequency,no bins there but in histogram it plot witg relative frequency with bins(with equal and unequal intervals).
#

#Cross table and scatter plot:
#categorical date(represent with cross table),side and side bar chart

#scatter plot are used when we are representing two numerical variables.
#outlier are data points that go against logic of whole datasets.

#Central tendency;
#mean,median ands mode:
#U=mean(average),easily effected by outlier.to avoid this problem we use median(middle number in ordered dataset).
#mode is value that occur more often.

#Measure of symmetry(skewness):
#skewness indicate whether data is is concentrated on one side
#three cases:
#1) if mean>median->Postive skew, right skweness means outlier tends to right side and vice versa.
#2)mean=median=mode-> Zero skew,symmetrical distribution.
#3)mean<median->negative skew. Left skew because outlier belon on left side of distributiuon.

#Measuring how data is spread out:
#1 Variance: it is measure dispersion(dispersion is not negative) of set of data points around their mean. It is second degree calculation.
# * our sample variance has rightfully corrected upwards,in order to reflect higher potential variability.

#2 Standard variance: square root of population or sample variance.
#* coeeficent of variation= variance/mean.

#Co-variance:
#the two variables are correlated and the main statistics to measure this correlation is called co-variance.
#characterstics of co-variance:
#if co-varince>0, two variable move together, co-varince<0, two variable move oppsoite, co-varince=0,two variables are indepedent.
#formula= (x-x̅)*(y-ȳ)/n-1

#if covariance will very high or low like 33467 or 0.000002, in that case we will use correlation coffecient(>-1 and <1)
#formula: 

#Practical Example:
#Note trick to get data type is take mean and mean tell data is categorical or numerical. for eg: mean of id is 2279 and mean of price $23456, mean of price is important thats why it is numerical.
#Id is categorical data type. it is qualitative and nomianl.
#Age: it is quantative and ratio. Age is descret as well as contious variable. histogram
#Price:is numerical and can be discret and contious
#gender is catergorial and nominal.
#State:categorical variable. nominal use pie chart
#Relation between age and price by scatter plot.


#Pandas makes it very easy to find the correlation coefficient! We can simply call the .corr() method on the dataframe of interest.

# Getting the Pearson Correlation Coefficient
#correlation = df.corr()
#print(correlation.loc['History', 'English'])

#Numpy:
#Similarly, Numpy makes it easy to calculate the correlation matrix between different variables.
#The library has a function named .corrcoef()
#corr = np.corrcoef(df['History'], df['English'])
#print(corr)

#Scipy:
#import scipy.stats as stats
#We can use the scipy.stats.pearsonr()
#r = stats.pearsonr(df['History'], df['English'])
#print(r)

#o/p:# Returns: (0.9309116476981856, 0.0)
#We can see that this returns a tuple of values:

#r: Pearson’s correlation coefficient
#p-value: long-tailed p-value




