
'''Task 1'''
https://www.hackerrank.com/challenges/py-if-else/problem

Given an integer, , n perform the following conditional actions:

If  n is odd, print Weird
If  n is even and in the inclusive range of  2 to 5 , print Not Weird
If  n is even and in the inclusive range of  6 to 20, print Weird
If  n is even and greater than 20 , print Not Weird

Input Format

A single line containing a positive integer, n .

Constraints
1<=n<=100
Output Format

Print Weird if the number is weird. Otherwise, print Not Weird.
-----------
import sys

N = int(input().strip())
if N % 2 == 1 :
    print ('Weird')
else:
    if N <= 5 :
        print ("Not Weird")
    elif N <= 20:
        print ("Weird")
    else:
        print ("Not Weird")
'''Task 2'''
https://www.hackerrank.com/challenges/python-arithmetic-operators/problem

The provided code stub reads two integers from STDIN,  a and b . Add code to print three lines where:

1.The first line contains the sum of the two numbers.
2.The second line contains the difference of the two numbers (first - second).
3.The third line contains the product of the two numbers.
-------
a=int(input())
b=int(input())

print (a+b)
print (a-b)
print (a*b)
-------------------------
a = 3
b = 5

for i in [a+b, a-b, a*b]: print(i)
-----------------------------------------------
'''Task 3'''kod isleyir inpu edende error cixir
https://www.hackerrank.com/challenges/python-division/problem?isFullScreen=true

The provided code stub reads two integers,  and , from STDIN.

Add logic to print two lines. The first line should contain the result of integer division,  a//b .
 The second line should contain the result of float division,  a/b.

No rounding or formatting is necessary.
#I)
a=input()
b=input()
x=a/b
y=float(a)/b
print (x)
print (y)
#2) 
a = input()
b = input()

print (a//b)
print (a/b)
--------------------------------------------
'''Task 4'''
https://www.hackerrank.com/challenges/python-loops/problem
The provided code stub reads and integer,n, from STDIN. For all non-negative integers i<n, print i^^2.

n=int(input())
for i in range(n):
    print (i**2)


---------------------------
'''Task5'''
https://www.hackerrank.com/challenges/write-a-function/problem?isFullScreen=true
An extra day is added to the calendar almost every four years as February 29, and the day is called a leap day. 
It corrects the calendar for the fact that our planet takes approximately 365.25 days to orbit the sun.
 A leap year contains a leap day.

In the Gregorian calendar, three conditions are used to identify leap years:

The year can be evenly divided by 4, is a leap year, unless:
The year can be evenly divided by 100, it is NOT a leap year, unless:
The year is also evenly divisible by 400. Then it is a leap year.
This means that in the Gregorian calendar, the years 2000 and 2400 are leap years, while 1800, 1900, 2100, 2200, 2300 and 2500 are NOT leap years.

Task

Given a year, determine whether it is a leap year. If it is a leap year, return the Boolean True, otherwise return False.

Note that the code stub provided reads from STDIN and passes arguments to the is_leap function. It is only necessary to complete the is_leap function.

def is_leap(year):
    leap = False
    
    if year % 4 == 0:
        leap = True
        if year % 100 == 0:
            leap = False
            if year % 400 == 0:
                leap = True
    return leap

--------------------------------
'''Task6'''
https://www.hackerrank.com/challenges/python-print/problem?isFullScreen=true
The included code stub will read an integer,n, from STDIN.

Without using any string methods, try to print the following:
123...n

Note that "" represents the consecutive values in between.

for i in range(1,int(input())+1):
    print(i,sep='',end='')

------------------------------
'''Task7'''
https://www.hackerrank.com/challenges/find-second-maximum-number-in-a-list/problem?isFullScreen=true
Given the participants score sheet for your University Sports Day, you are required to find the runner-up score.
 You are given n scores. Store them in a list and find the score of the runner-up.

Input Format

The first line contains n. The second line contains an array  A[] of n  integers each separated by a space.


Output Format

Print the runner-up score.

n=int(input())
a=list(map(int, input().split()))
a=list(set(a))
a.sort()
print(a[len(a)-2])
--------------
'Task8'
https://www.hackerrank.com/challenges/nested-list/problem?isFullScreen=true
Given the names and grades for each student in a class of  N students,
 store them in a nested list and print the name(s) of any student(s) having the second lowest grade.

xs = [(input(), float(input())) for _ in range(int(input()))]
min_mark = min(x[1] for x in xs)
xs = [x for x in xs if x[1] > min_mark]
min2_mark = min(x[1] for x in xs)
xs = sorted([x[0] for x in xs if x[1] == min2_mark])
for x in xs:
    print(x)
-------------------
'Task9'
https://www.hackerrank.com/challenges/finding-the-percentage/problem?isFullScreen=true
The provided code stub will read in a dictionary containing key/value pairs of name:[marks] for a list of students.
 Print the average of the marks array for the student name provided, showing 2 places after the decimal.

Example
marks key:value pairs are
'alpha':[20,30,40]
'beta':[30,50,70]
query_name='beta'

The query_name is 'beta'. beta''s average score is .(30+50+70)/3=50.0

Input Format
(30+50+70)/3=50.0
The first line contains the integer n , the number of students records. 
The next  n lines contain the names and marks obtained by a student, each value separated by a space.
 The final line contains query_name, the name of a student to query.
Output Format

Print one line: The average of the marks obtained by the particular student correct to 2 decimal places.
-----------
data = {}
for _ in range(int(input())):
    name, *marks = input().split()
    data[name] = [float(m) for m in marks]
marks = data[input()]
print("%.2f" % (sum(marks)/len(marks)))
-----------------
'Task10'
https://www.hackerrank.com/challenges/swap-case/problem?isFullScreen=true
You are given a string and your task is to swap cases. In other words, convert all lowercase letters to uppercase letters and vice versa.

For Example:

Www.HackerRank.com → wWW.hACKERrANK.COM
Pythonist 2 → pYTHONIST 2  
Function Description

Complete the swap_case function in the editor below.

swap_case has the following parameters:

*string s: the string to modify
Returns

*string: the modified string
Input Format

A single line containing a string s.

print(input().swapcase())
----------------------------------------------

2. Kaggle Project
Verilmiş datasetlərdən istənilən birini seçərək üzərində tələb olunan analizləri edin:
a. Ümumi descriptive analiz
b. Ən az 1 plot və 1 pivot table
c. Regression prediction və ya Classification


4. Mobil telefonların qiymət aralıqlarının təyin olunması

import pandas as pd
df = pd.read_csv('test.csv')
df1=pd.read_csv('train.csv')
df.info()
df1.info()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



battery_power: Total energy a battery can store in one time measured in mAh

blue: Has bluetooth or not

clock_speed: speed at which microprocessor executes instructions

dual_sim: Has dual sim support or not

fc: Front Camera mega pixels

four_g: Has 4G or not

int_memory: Internal Memory in Gigabytes

m_dep: Mobile Depth in cm

mobile_wt: Weight of mobile phone

n_cores: Number of cores of processor

pc: Primary Camera mega pixels

px_height: Pixel Resolution Height

px_width: Pixel Resolution Width

ram: Random Access Memory in Mega Bytes

sc_h: Screen Height of mobile in cm

sc_w: Screen Width of mobile in cm

talk_time: longest time that a single battery charge will last when you are

three_g: Has 3G or not

touch_screen: Has touch screen or not

wifi: Has wifi or not

price_range: This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) 
and 3(very high cost).

'''
--------------
'''
'correlation between features'
'Following heatmap shows correlation values between features.'
import seaborn as sns
import matplotlib.pyplot as plt
corr=df.corr()
fig = plt.figure(figsize=(15,12))
r = sns.heatmap(corr, cmap='Purples')
r.set_title("Correlation ")
------------------------------------------
'price range correlation'
corr.sort_values(by=["price_range"],ascending=False).iloc[0].sort_values(ascending=False)
----------------
f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
--------------------
'touchscreen-ram and price range '
# Show each observation with a scatterplot
sns.stripplot(x="touch_screen", y="ram", hue="price_range",
              data=df, dodge=True, jitter=True,
              alpha=.25, zorder=1)

# Show the conditional means

# Improve the legend 
handles, labels = ax.get_legend_handles_labels()
ax.legend( title="Price Range",
          handletextpad=0, columnspacing=1,
          loc="best", ncol=2, frameon=True)
--------------------------------------
'4g ram price'
f, ax = plt.subplots(figsize=(10, 10))
ax=sns.swarmplot(x="four_g", y="ram", hue="price_range",
              palette="Dark2", data=df)
ax=sns.set(style="darkgrid")
-------------------------------
g = sns.FacetGrid(df, col="dual_sim", hue="price_range", palette="Set1",height=5
                   )
g = (g.map(sns.distplot, "ram").add_legend())
----

'train test split'

y = df["price_range"].values
x_data=df1.drop(["price_range"],axis=1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)

from yellowbrick.target import ClassBalance
visualizer = ClassBalance(labels=[0, 1, 2,3])
visualizer.fit(y_train, y_test)
visualizer.poof()
--------------

FIRST MODEL
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(x_train,y_train)
print("train accuracy:",svm.score(x_train,y_train))
print("test accuracy:",svm.score(x_test,y_test))

train accuracy: 0.91
test accuracy: 0.84

------
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
accuracy_list_train = []
k=np.arange(1,21,1)
for each in k:
    x_new = SelectKBest(f_classif, k=each).fit_transform(x_train, y_train)
    svm.fit(x_new,y_train)
    accuracy_list_train.append(svm.score(x_new,y_train))   
    
plt.plot(k,accuracy_list_train,color="green",label="train")
plt.xlabel("k values")
plt.ylabel("train accuracy")
plt.legend()
plt.show()
-------------------------------------
df.describe()
df.isnull().sum()
sns.pointplot(y='ram',x='price_range',data=df)

sns.pointplot(x='price_range',y='battery_power',data=df)

sns.boxplot(x='price_range',y='battery_power',data=df)

sns.pointplot(x='price_range',y='int_memory',data=df)

-----
categorical_col = ['blue','dual_sim','four_g','three_g','touch_screen','price_range']

for i in categorical_col:
  sns.countplot(df[i])
  plt.xlabel(i)
  plt.show()
  -----
  for i in df.drop(df[categorical_col],axis=1):
       fig = plt.figure(figsize=(9,8))
       plt.hist(df[i],color='purple',bins=10)
       plt.xlabel(i)
       plt.show()
       --------------
       labels = ["3G-supported",'Not supported']
values = df['three_g'].value_counts().values
fig1, ax1 = plt.subplots()
colors = ['red', 'blue']
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90,colors=colors)
plt.show()
------
plt.figure(figsize=(10,6))
df['fc'].hist(alpha=0.5,color='blue',label='Front camera')
df['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
----
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123,stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
lr = LogisticRegression(penalty='l2',C=0.1)
lr.fit(x_train,y_train)




y_test_pred = lr.predict(x_test)
y_train_pred = lr.predict(x_train)

lr_acc=accuracy_score(y_test_pred,y_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))

'train acc' 64.26666
'test acc' 63.0

'''DT'''
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

y_test_pred3 = dtc.predict(x_test)
y_train_pred3=dtc.predict(x_train)


print("Train Set Accuracy:"+str(accuracy_score(y_train_pred3,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred3,y_test)*100))
print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred3,y_test))
print("\nClassification Report:\n%s"%classification_report(y_test_pred3,y_test))
'Train Acc' 100.0
'Test Acc' 79.80000000000001

# hyper parameter tuning

grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(x_train, y_train)
grid_search.best_params_
dtc = grid_search.best_estimator_
y_predi=dtc.predict(x_test)
dtc_train_acc = accuracy_score(y_train, dtc.predict(x_train))
dtc_test_acc = accuracy_score(y_test, y_predi)

print(f"Training Accuracy of SVC Model is {dtc_train_acc}")
print(f"Test Accuracy of SVC Model is {dtc_test_acc}")

92
83





















