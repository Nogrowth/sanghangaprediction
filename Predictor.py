"""
DB에서 학습에 필요한 데이터를 불러와서 세팅하는 파일

해야할 것
상한가 / 비상한가 분류 1:1 정도로 일단 해보자
상한가 데이터를 찾아서 당일과 이전 10일치 OHLCV 를 불러온다. (total 11일치)
random data에 대해 똑같이 한다.

one-hot-encoding 으로 상한가/비상한가를 구분한다.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import random
import tensorflow as tf
from datetime import date, datetime

# for random forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # 정확도 함수

# for save model on pickle fmt
import pickle
import joblib

class Predictor:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:as6114@localhost:3306/stock_data", encoding='utf-8')
        self.engine_datasets = create_engine("mysql+pymysql://root:as6114@localhost:3306/datasets", encoding='utf-8')
        # self.n = 상한가 이전 n일의 data 분석
        self.n = 10

        # for 문 위해서 종목명 list 받아오기
        sql = "SELECT TABLE_NAME FROM information_schema.tables WHERE table_schema='stock_data'"
        self.stock_list = self.engine.execute(sql).fetchall()
        self.stock_list.remove(('stock_list', ))

        # data setting 용 column set
        self.col = ['o', 'h', 'l', 'c', 'v']
        for i in range(1, self.n):
            for mkr in self.col[0:5]:
                self.col.append(mkr + f'+{i}')
        self.col.append('upper_limit_tomorrow')

    def data_collection(self, end_date):
        """
        특정 날짜에 이전 n일치 ohclv column (total 5n개) + 상한가 여부 판단하는 column 1개 모으기
        """
        dataset = pd.DataFrame([], columns=self.col)
        end_date = '2022-02-17'
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        for name in self.stock_list:
        # date 날짜의 Change 및 이전 n일간의 ohlcv 모으기
            name = name[0]
            sql = f'select `Date`, `Open`, `High`, `Low`, `Close`, `Volume` from `{name}` where `Date` < "{end_date}" order by `Date` desc limit {self.n}'
            ohlcv_10 = pd.read_sql(sql, self.engine)
            ohlcv_10.sort_values('Date', inplace=True)
            ohlcv_10.drop('Date', axis=1, inplace=True)
            if ohlcv_10.size == 5 * self.n:
                flat_data = ohlcv_10.values.reshape(5*self.n, )
                sql = f'select `Change` from `{name}` where `Date`="{end_date}"'
                change = self.engine.execute(sql).fetchone()[0]
                if change > 0.295:
                    flat_data = np.append(flat_data, np.array([1]))
                else:
                    flat_data = np.append(flat_data, np.array([0]))
                dataset.loc[len(dataset)] = flat_data
        dataset.to_sql(f'{end_date}_10ohlcv_data', self.engine_datasets, index=False, if_exists='replace')
        print(f'{end_date}의 상한가여부 및 이전 10일치 ohlcv data를 dataset DB에 저장하였습니다. ')


    def mlp_model(self):
        sql = 'select * from `dataset_limit`'
        dataset_1 = pd.read_sql(sql, self.engine_datasets)
        sql = 'select * from `dataset_not_limit`'
        dataset_2 = pd.read_sql(sql, self.engine_datasets)

        dataset = pd.concat([dataset_1, dataset_2])

        # randomization
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        # 독립변수, 종속변수 분리
        col = ['o', 'h', 'l', 'c', 'v']
        for i in range(1, self.n):
            for j in col[0:5]:
                col.append(j + f'+{i}')

        dataset_x = dataset[col]
        dataset_y = dataset[['upper limit next day']]

        # dataset_y 의 dtype을 category로 변경
        dataset_y['upper limit next day'] = dataset_y['upper limit next day'].astype('category')

        # train / test data 분리
        train_x = dataset_x[0:int(0.2 * len(dataset_x))]
        test_x = dataset_x[int(0.2 * len(dataset_x)):]
        train_y = dataset_y[0:int(0.2 * len(dataset_y))]
        test_y = dataset_y[int(0.2 * len(dataset_y)):]

        # 모델 생성
        X = tf.keras.layers.Input(shape=[50])

        H = tf.keras.layers.Dense(10)(X)
        H = tf.keras.layers.BatchNormalization()(H)
        tf.keras.layers.Activation('swish')(H)

        H = tf.keras.layers.Dense(10)(H)
        H = tf.keras.layers.BatchNormalization()(H)
        tf.keras.layers.Activation('swish')(H)

        Y = tf.keras.layers.Dense(1, activation='softmax')(H)
        model = tf.keras.models.Model(X, Y)
        model.compile(loss='categorical_crossentropy', metrics='accuracy')

        # 모델 학습
        model.fit(train_x, train_y, epochs=1000)

    def random_forest(self, target_date):
        sql = f'select * from `{target_date}_10ohlcv_data`'
        dataset = pd.read_sql(sql, self.engine_datasets)

        # randomization
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        dataset_x = dataset.drop('upper_limit_tomorrow', axis=1)
        dataset_y = dataset[['upper_limit_tomorrow']]

        # dataset_y 의 dtype을 category로 변경
        dataset_y['upper_limit_tomorrow'] = dataset_y['upper_limit_tomorrow'].astype('category')

        # train / test data 분리
        train_x = dataset_x[0:int(0.2 * len(dataset_x))]
        test_x = dataset_x[int(0.2 * len(dataset_x)):]
        train_y = dataset_y[0:int(0.2 * len(dataset_y))]
        test_y = dataset_y[int(0.2 * len(dataset_y)):]

        # 모델 생성
        clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
        clf.fit(train_x, train_y)

        # 정확도 확인
        predict_train = clf.predict(train_x)
        print('모델의 정확도를 출력합니다.')
        print(accuracy_score(train_y, predict_train))

        predict_test = clf.predict(test_x)
        print('모델의 정확도를 출력합니다.')
        print(accuracy_score(test_y, predict_test))

        # 모델 저장
        joblib.dump(clf, './random_forest_model.pkl')

    def prediction(self, prediction_date):
        """
        model 을 바탕으로 내일 상한가 여부를 예측하여 DB에 저장
        """
        today = datetime.today().date()
        prediction = pd.DataFrame([], columns=['Name', 'Prediction'])
        for idx, name in enumerate(self.stock_list):
            name = name[0]
            # 오늘날짜 기준 데이터
            sql = f"SELECT `Date`, `Open`, `Low`, `High`, `Close`, `Volume` FROM `{name}` WHERE `Date` <= '{today}' " \
                  f"ORDER BY `Date` desc LIMIT 10"
            real_data = pd.read_sql(sql, self.engine)
            real_data.sort_values('Date', inplace=True)
            real_data.drop('Date', axis=1, inplace=True)

            # model 불러오기
            loaded_model = joblib.load('./random_forest_model.pkl')

            if real_data.size == 5 * self.n:
                real_data_flat = real_data.values.reshape(1, 5 * self.n)
                predict = loaded_model.predict(real_data_flat)
                df_predict_result = pd.DataFrame({'Name': name, 'Prediction': predict})
                prediction = pd.concat([prediction, df_predict_result])
                print(f'{name} 종목의 예측 결과를 저장했습니다. ({idx}/{len(self.stock_list)})')
        prediction.to_sql(f'predict_{today}', self.engine_datasets, if_exists='replace', index=False)

if __name__ == "__main__":
    p = Predictor()
    p.data_collection('2022-02-17')
    p.random_forest('2022-02-17')
    p.prediction()










