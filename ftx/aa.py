import os
from random import randint
from time import sleep
import pandas as pd
from datetime import date
import csv


def days_between(y1, m1, date1, y2, m2, date2):
    d0 = date(y1, m1, date1)
    d1 = date(y2, m2, date2)
    return abs((d1 - d0).days)


header = ['refereeAccountId', 'totalSize', '',
          'Previous Change Date', '', 'Current Date', '', 'Activity']
data = []


today = date.today()
current_time = today.strftime("%Y-%m-%d")


# from referrals read data
file = open('referrals.csv')
referrals_file = csv.reader(file)
referrals_csv = []
for row in referrals_file:
    if len(row) == 0:
        continue
    referrals_csv.append(row)
file.close()
# print(referrals_csv)

# from update csv read data
file = open('update.csv')
update_file = csv.reader(file)
update_csv = []
for row in update_file:
    if len(row) == 0:
        continue
    update_csv.append(row)
file.close()
# print(update_csv)


result_row = []
for i in range(len(referrals_csv)):
    if i == 0:
        continue
    print(referrals_csv[i])
    check = 0
    refereeAccountId = referrals_csv[i][0]
    totalSize = referrals_csv[i][1]
    for j in range(len(update_csv)):
        if(referrals_csv[i][0] == update_csv[j][0]):
            check = 1
            update_time = update_csv[j][3]
    if(check == 0):
        update_time = current_time

    current = current_time
    y1 = int(current_time[0:4])
    m1 = int(current_time[5:7])
    date1 = int(current_time[8:10])

    d2 = update_time
    y2 = int(d2[0:4])
    print(y2)
    m2 = int(d2[5:7])
    date2 = int(d2[8:10])
    c = days_between(y1, m1, date1, y2, m2, date2)
    if c <= 7:
        if c == 0:
            activity = "Active Today"
        else:
            activity = "Active => Active last %s days before" % c
    elif c > 7 and c <= 14:
        activity = "Sometimes Active => Active last %s days before" % c
    else:
        activity = "Inactive => Active last %s days before" % c

    lst = [refereeAccountId, totalSize, ' ',
           update_time, ' ', current, ' ', activity]
    result_row.append(lst)
    print(lst)
    print()

result = open('update.csv', 'w', encoding='UTF8')
writer = csv.writer(result)
writer.writerow(header)
for row in result_row:
    writer.writerow(row)
