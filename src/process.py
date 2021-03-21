import os
import pandas as pd


def folder_setup(city):
    '''setup folders for each city
    Args:
        city: city
    Returns:
    '''
    if not os.path.exists('dataset/'+city):
        os.mkdir('dataset/'+city)
        os.mkdir('dataset/'+city+'/process/')
        os.mkdir('dataset/'+city+'/emb/')
        os.mkdir('dataset/'+city+'/feature/')
        os.mkdir('dataset/'+city+'/result/')
        os.mkdir('dataset/'+city+'/defense/')


def data_process(city, cicnt):
    ''' process the raw dataset
    Args:
        city: city
        cicnt: ?
    Returns:
        checkin: processed check-in data
        friends: friends list (asymetric) [u1, u2]
    '''

    checkin = pd.read_csv('dataset/'+city+'_'+str(cicnt)+'.checkin')
    friends = pd.read_csv('dataset/'+city+'_'+str(cicnt)+'.friends')

    return checkin, friends


# add by zhixin
def data_process_gowalla():
    ''' process the raw dataset
    Args:
        city: city
        cicnt: ?
    Returns:
        checkin: processed check-in data
        friends: friends list (asymetric) [u1, u2]
    '''

    toy_rows = 99999
    checkin = pd.read_csv('gowalla/gowalla_checkins.csv', nrows=toy_rows)
    friends = pd.read_csv('gowalla/gowalla_friendship.csv', nrows=toy_rows)
    checkin = checkin.rename(columns={"userid": "uid", "placeid": "locid"})
    friends = friends.rename(columns={"userid1": "u1", "userid2": "u2"})

    return checkin, friends
