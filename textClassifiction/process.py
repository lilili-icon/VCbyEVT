import pickle
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

# # with open("THUCNews/data/vocab.txt", "r",encoding="utf-8") as f:  # 打开文件
# #     data = f.read()  # 读取文件
# #
# # f = open('THUCNews/data/vocab.text', 'wb',0)
# # #对象写入文件
# # pickle.dump(data,f)
# # f.close()
# g = open('THUCNews/data/embedding_SougouNews.npz', 'rb')
# e=pickle.load(g)
# print(e)