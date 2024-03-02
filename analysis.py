#!/usr/bin/env python3.8
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import errno
import os
# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz


# Ukol 1: nacteni dat
def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load the locally stored accident statistics file, create 'date' column and change datatypes of some other columns to save some memory.

    Arguments:
        filename (str)              - path to the accident statistics file
        verbose (bool, optional)    - if is set, print the orginal size and new size of the loaded data to the standard output (default False)

    Return value:
        pd.DataFrame    - resized accidents data with 'date' column
    """
    df = pd.read_pickle(filename, compression='gzip')
    # get original size of data
    orig_size = df.memory_usage(deep=True).sum()

    df.rename(columns={'p2a' : 'date'}, inplace=True)

    df = df.astype({'p36': 'category', 'date': 'datetime64', 'weekday(p2a)': 'category', 'p2b': 'category', 'p6': 'category', 'p7': 'category', 'p8': 'category', 'p9': 'category', 
                    'p10': 'category', 'p11': 'category', 'p15': 'category', 'p16': 'category', 'p17': 'category', 'p18': 'category', 'p19': 'category', 'p20': 'category', 
                    'p21': 'category', 'p22': 'category', 'p23': 'category', 'p24': 'category', 'p27': 'category', 'p28': 'category', 'p35': 'category', 'p39': 'category', 
                    'p44': 'category', 'p45a': 'category', 'p47': 'category', 'p48a': 'category', 'p49': 'category', 'p50a': 'category', 'p50b': 'category', 'p51': 'category', 
                    'p52': 'category',  'p55a': 'category', 'p57': 'category', 'p58': 'category', 'h': 'category', 'i': 'category', 'j': 'category', 'k': 'category', 
                    'l': 'category', 'n': 'category', 'o': 'category', 'p': 'category', 'q': 'category', 'r': 'category', 's': 'category', 't': 'category', 'p5a': 'category'})
    
    if verbose:
        # print original and new size of data in MB to one decimal place
        print(f'orig_size={(orig_size/1048576):.1f} MB')
        print(f'new_size={(df.memory_usage(deep=True).sum()/1048576):.1f} MB')

    return df


# Ukol 2: následky nehod v jednotlivých regionech
def plot_conseq(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):
    """
    Create graph of accident consequences in individual regions, which are listed on the x-axis according to the number of accidents in descending order.

    Arguments:
        df (pd.DataFrame)               - accidents data
        fig_location (str, optional)    - address, where the figure is stored (default None - figure is not saved)
        show_figure (bool, optional)    - if is set, the figure is shown in the window (default False)
    """
    if fig_location and show_figure == False:
        # there is no need to do anything, if both of these arguments are not set
        return
    
    # get necessary records and sort them according to number of accidents in individual regions
    df = df.groupby('region').agg({'p13a' : 'sum', 'p13b' : 'sum', 'p13c' : 'sum', 'p14' : 'count'}).reset_index().sort_values(by='p14', ascending=False)

    sns.set_style('darkgrid', rc={'axes.facecolor' : '#E2D5BF'})

    fig, axes = plt.subplots(nrows=4, constrained_layout=True, figsize=(7,8), sharex=True)

    # prepare concrete color palette
    palette = sns.color_palette('blend:#F56767,#900000', len(df.index))

    # set and customize individual subplots
    for ax, col, header in zip(axes, ['p13a', 'p13b', 'p13c', 'p14'], ['Deaths', 'Heavily injured', 'Slightly injured', 'Total number of accidents']):
        ax.set_title(header, fontsize=13)

        # create bar graph and set color intensity of each bar according to its height
        sns.barplot(ax=ax, x='region', y=col, data=df, palette=__get_sorted_palette(palette, df[col]))
        ax.set_ylabel('number of people', style='italic')
        ax.label_outer()

    axes[-1].set_xlabel('region shortcut', style='italic')
    axes[-1].set_ylabel('number of accidents', style='italic')

    if fig_location:
        __save_image(fig, fig_location)

    if show_figure:
        plt.show()


def __get_sorted_palette(palette, col):
    """
    Rearrange palette according to the values in column.

    Arguments:
        palette     - original palette
        col         - column with values specifying the new rearranged palette

    Return value:
        np.ndarray  - new rearrange palette
    """
    order = col.argsort().argsort()  # get sorted indexes and then than idexes in the palette that correspond to them 
    return np.array(palette)[order]  # rearrange and return palette


def __save_image(fig, fig_location):
    """
    Save figure to the specified location.

    Arguments:
        fig             - image to be saved
        fig_location    - address, where the figure is stored
    """
    fig_location = os.path.normpath(fig_location)
    dir_name, fig_file = os.path.split(fig_location)

    # create directories
    if dir_name != '':
            try:
                os.makedirs(dir_name)
            except OSError as e:
                if(e.errno != errno.EEXIST):
                    raise

    fig.savefig(fig_location)


# Ukol3: příčina nehody a škoda
def plot_damage(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):
    """
    Show the number of accidents depending on the damage to the vehicles, which is divided into categories, for selected regions.
    Numbers of accidents are further divided according to the main cause of the accident.

    Arguments:
        df (pd.DataFrame)               - accidents data
        fig_location (str, optional)    - address, where the figure is stored (default None - figure is not saved)
        show_figure (bool, optional)    - if is set, the figure is shown in the window (default False)
    """
    if fig_location and show_figure == False:
        # there is no need to do anything, if both of these arguments are not set
        return

    # get necessary data for selected regions
    df = df[['region', 'p12', 'p53']][df['region'].isin(['PHA', 'STC', 'ULK', 'MSK'])]

    # divide records into the categories according to the main cause of accident
    df['p12'] = pd.cut(df['p12'], [100, 200, 300, 400, 500, 600, 615], labels=['not caused by the driver', 'excessive speed', 'incorrect overtaking', 
                       'not giving way', 'incorrect driving style', 'technical defect of the vehicle'], include_lowest=True)

    # divide records into the categories according to the vehicle damage
    df['p53'] = pd.cut(df['p53'], [0, 500, 2000, 5000, 10000, float('inf')], labels=['< 50', '50 - 200', '200 - 500', '500 - 1000', '> 1000'], right=False)

    sns.set_theme(style='darkgrid', rc={'axes.facecolor' : '#E2D5BF', 'xtick.labelsize' : 'small', 'ytick.labelsize' : 'small'})

    # create graphs and customize them
    g = sns.catplot(data=df, x='p53', kind='count', hue='p12', col='region', col_wrap=2, height=4.4)
    g.set(yscale='log')
    g.set_titles('{col_name}', size=13)
    g.set_axis_labels('Damage [thousand CZK]', 'Number of cases', style='italic')
    g.legend.set_title('Cause of the accident')
    g.tight_layout()

    if fig_location:
        __save_image(g, fig_location)

    if show_figure:
        plt.show()


# Ukol 4: povrch vozovky
def plot_surface(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Show dependence between the number of accidents, the month and road conditions.

    Arguments:
        df (pd.DataFrame)               - accidents data
        fig_location (str, optional)    - address, where the figure is stored (default None - figure is not saved)
        show_figure (bool, optional)    - if is set, the figure is shown in the window (default False)
    """
    if fig_location and show_figure == False:
        # there is no need to do anything, if both of these arguments are not set
        return

    # get necessary data for selected regions
    df = df[['region', 'date', 'p16', 'p1']][df['region'].isin(['PHA', 'STC', 'ULK', 'MSK'])]

    # get numbers of accidents with concrete road condition on specific dates
    df = pd.pivot_table(df, columns='p16', values='p1', index=['region', 'date'], fill_value=0, aggfunc='count')
    
    df.rename(columns={1 : 'dry and clean', 2 : 'dry and dirty', 3 : 'wet', 4 : 'muddy', 5 : 'icing and snow - sprinkled', 6 : 'icing and snow - not sprinkled', 
                       7 : 'spilled oil, diesel etc.', 8 : 'continuous snow layer, slush', 9 : 'sudden change of state', 0 : 'different condition'}, inplace=True)

    # recommended way - aggregation of data by regions and subsampling into months and convert data back to the stacked format
    df = df.groupby('region').resample('M', level=1).sum().melt(var_name='Road condition', value_name='count', ignore_index=False).reset_index()

    sns.set_theme(style='darkgrid', rc={'axes.facecolor' : '#E2D5BF', 'xtick.labelsize' : 'small', 'ytick.labelsize' : 'small'})
    
    # create graphs and customize them
    g = sns.relplot(data=df, x='date', y='count', kind='line', hue='Road condition', col='region', col_wrap=2, height=3.5, aspect=1.5)

    g.set_titles('{col_name}', size=13)
    g.set_axis_labels('Date of the accident', 'Number of accidents', style='italic')
    g.tight_layout()

    if fig_location:
        __save_image(g, fig_location)

    if show_figure:
        plt.show()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni ¨
    # funkce.
    df = get_dataframe("accidents.pkl.gz")
    plot_conseq(df, fig_location="01_nasledky.png", show_figure=True)
    plot_damage(df, "02_priciny.png", True)
    plot_surface(df, "03_stav.png", True)
