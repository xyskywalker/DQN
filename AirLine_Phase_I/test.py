import AirLine_Phase_I.DataReader as DR

reader = DR.DataReader(filename='DATA_20170705.xlsx' , env_d=59)
print(reader.read())