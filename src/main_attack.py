import sys
from process import folder_setup, data_process
from process import data_process_gowalla
from emb import ul_graph_build, para_ul_random_walk, emb_train
from predict import feature_construct, unsuper_friends_predict
from datetime import datetime 

# # city = sys.argv[1]  # ny la london
# # cicnt = int(sys.argv[2])  # 20 30
# city, cicnt = "la", 20
# folder_setup(city)
# checkin, friends = data_process(city, cicnt)

# add by zhixin
city, cicnt = "gowalla", 20
folder_setup(city)
checkin, friends = data_process_gowalla()
print "Check-in data:", checkin.shape, "\n", checkin[0: 3]
print "Friends data:", friends.shape, "\n", friends[0: 3]

ul_graph, lu_graph = ul_graph_build(checkin, 'locid')
print "user-location Graph:", ul_graph.shape

model_name = str(cicnt) + '_locid'
print(model_name)

walk_len, walk_times = 100, 20  # maximal 100 walk_len, 20 walk_times

st = datetime.now()
print 'Start walking'
para_ul_random_walk(city, model_name, checkin.uid.unique(), ul_graph, lu_graph, walk_len, walk_times)
print 'walk done'
ed = datetime.now()
print "Total seconds for working:", (ed-st).seconds

st = datetime.now()
print 'emb training'
emb_train(city, model_name)
print 'emb training done'
ed = datetime.now()
print "Total seconds for training:", (ed-st).seconds

feature_construct(city, model_name, friends)
unsuper_friends_predict(city, model_name)
