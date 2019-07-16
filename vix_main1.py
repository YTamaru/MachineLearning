import vix_util as util
import DataPrepare
import importlib
from bokeh.charts import Bar, output_notebook, show
import pdb
import os,sys
from keras.models import load_model
#pdb.set_trace()

importlib.reload(util)

stock_data = util.prepare_stock_data("vixcurrent.csv",300)

#type(stock_data.ix)<class 'pandas.core.indexing._IXIndexer'>


"""
stock_dataの中身：
              open   close    high     low    volume    code     ema_12  \
date
2016-03-07  10.996  13.189  13.189  10.996     385.0  603027  13.189000
2016-03-08  14.505  14.505  14.505  14.505     219.0  603027  13.391462
2016-03-09  15.960  15.960  15.960  15.960     206.0  603027  13.786621
2016-03-10  17.555  17.555  17.555  17.555     375.0  603027  14.366372
.
.
.
"""
dp = DataPrepare.DataPrepare(stock_data.ix[1:].copy())

trainX, trainY, date = dp.create_dataset(dp.train)

"""
trainX = (159,20,19) : <class 'numpy'>
trainY = (159,3) : [[0,0,1],[1,0,0],....[0,1,0]] <class 'numpy'>
date = (159) <class 'list'>
"""



output_notebook()

show_data = stock_data.copy()
#show_data.redex()
#show_data.index = map(lambda x: str(x).split(' 00:00:00')[0], list(show_data.index))
p = Bar(show_data, values="close", title="stock example", legend="")

nb_epoch = 200
models_path = "vix_models"
filepath = "trainX_Y_model_" + "epoch_" + str(nb_epoch)
filepath = os.path.join(models_path,filepath + ".h5")
if os.path.exists(filepath):
	model = load_model(filepath)
	print("load_model : ",filepath)
else:
	batch_size = 1
	time_step = dp.time_step
	labels_len = len(dp.labels)
	model = util.CreateModel(batch_size, time_step, labels_len)
	model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)


testX, testY, date= dp.create_dataset(dp.test)
predictY = model.predict(testX, batch_size=batch_size)
evauation = model.evaluate(testX, testY, batch_size=batch_size)
print(evaluation)



# pdb.set_trace()

# win = 0
# for i in range(trans_predictY.shape[0]):
#     print(testY[i], trans_predictY[i])
#     if (testY[i] == trans_predictY[i]).all():
#         win = win + 1
# win = win / trans_predictY.shape[0]
# print(win)

# stock_data = util.prepare_stock_data("600810")

# dp = DataPrepare.DataPrepare(stock_data.ix[1:].copy())

# trainX, trainY, date = dp.create_dataset(dp.train)
# testX, testY, date= dp.create_dataset(dp.test)
# model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=300, verbose=1)




# stock_data = util.prepare_stock_data("002715")
# dp_002715 = DataPrepare.DataPrepare(stock_data.ix[1:].copy())
# test_002715 = list(dp_002715.create_dataset(dp_002715.test))
# train_002715 = list(dp_002715.create_dataset(dp_002715.train))
# predictY = model.predict(test_002715[0], batch_size=batch_size)
# trans_predictY = []
# for item in predictY:
#     trans_predictY.append(util.trans(item))
# trans_predictY = numpy.array(trans_predictY)
# util.result_display(dp_002715.test.index[test_002715[2]],  trans_predictY)


# model.evaluate(test_002715[0], test_002715[1], batch_size=batch_size)

# model.fit(train_002715[0], train_002715[1], batch_size=batch_size, nb_epoch=300, verbose=1)

# util.proba_display(dp_002715.test.index[test_002715[2]], model.predict(test_002715[0], batch_size=batch_size))

# test = numpy.array([32, 14, 8, 323])
# [i], = numpy.where(test == max(test))
# print(i)
