
*** '''Task 1'''
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



















