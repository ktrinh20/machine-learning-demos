"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Purpose:    Data Exploration

Note:       See full explanation at 
            https://www.kaggle.com/code/ktrinh/sdss-classification-with-random-forests-99-2/notebook

@author:    Kevin Trinh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pieChart(sdss_df):
    '''Plot a pie chart for label count.'''
    label_counts = sdss_df['class'].value_counts()
    colors = ['skyblue', 'red', 'gold']
    fig1, ax1 = plt.subplots()
    ax1.pie(label_counts, labels=['Galaxy', 'Stars', 'Quasars'],
            autopct='%1.2f%%', startangle=45, colors=colors)
    ax1.axis('equal')
    plt.title('SDSS Object Classes')
    plt.show()

def distribution(sdss_df, axes, feature, row):
    '''Plot the distribution of a space object w.r.t. a given feature.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    for i in range(len(labels)):
        label = labels[i]
        ax = sns.distplot(sdss_df.loc[sdss_df['class']==label, feature], 
                          kde=False, bins=30, ax=axes[row, i], color=colors[i])
        ax.set_title(label)
        if (i == 0):
            ax.set(ylabel='Count')
            
def equitorial(sdss_df, row):
    '''Plot equitorial coordinates of observations.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    label = labels[row]
    sns.lmplot(x='ra', y='dec', data=sdss_df.loc[sdss_df['class']==label],
               hue='class', palette=[colors[row]], scatter_kws={'s': 2}, 
               fit_reg=False, height=4, aspect=2)
    plt.ylabel('dec')
    plt.title('Equitorial coordinates')
    

def main():

    # read in SDSS data
    filepath = 'SDSS_data.csv'
    sdss_df = pd.read_csv(filepath, encoding='utf-8')

    # define lists of relevant features
    geo = ['ra', 'dec']
    nonugriv = ['redshift', 'plate', 'mjd', 'fiberid']
    ugriv = ['u', 'g', 'r', 'i', 'z']

    # plot pie chart of label count
    pieChart(sdss_df)

    # plot equitorial coordinates of observations
    for row in range(3):
        equitorial(sdss_df, row)
        plt.show()
    
    # plot the distribution of non-geo and non-ugriv features
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 14))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(nonugriv)):
        feat = nonugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()
        
    # plot the distribution of ugriv features
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(ugriv)):
        feat = ugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()

main()