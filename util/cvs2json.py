import csv
import json
import pandas as pd

def cvs2json(path):
    with open(path+".csv",'r',encoding="UTF-8-sig") as f:
        reader = csv.reader(f)
        fieldnames = next(reader)#获取数据的第一列，作为后续要转为字典的键名 生成器，next方法获取
        # print(fieldnames)
        csv_reader = csv.DictReader(f,fieldnames=fieldnames) #self._fieldnames = fieldnames # list of keys for the dict 以list的形式存放键名
        for row in csv_reader:
            d={}
            for k,v in row.items():
                d[k]=v
            print(d)
            with open(path+".json", "a") as f:
                f.write("{\"paragraphs\":[")
                json.dump(d, f)
                f.write("]},")

            print("加载入文件完成...")
def xlsx_to_csv_pd(path):
    data_xls = pd.read_excel(path+'.xlsx', index_col=0)
    data_xls.to_csv(path+'.csv', encoding='UTF-8-sig')


if __name__ == '__main__':

    # xlsx_to_csv_pd('RQ4_train')
    # xlsx_to_csv_pd('RQ4_test')

    cvs2json("RQ4_train")
    cvs2json("RQ4_test")