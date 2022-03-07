"""
DB에서 학습에 필요한 데이터를 불러와서 세팅하는 파일

해야할 것
상한가 / 비상한가 분류 1:1 정도로 일단 해보자
상한가 데이터를 찾아서 당일과 이전 10일치 OHLCV 를 불러온다. (total 11일치)
random data에 대해 똑같이 한다.

one-hot-encoding 으로 상한가/비상한가를 구분한다.
"""
import pandas as pd
from sqlalchemy import create_engine
import random
import tensorflow as tf

class Datasetter:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:as6114@localhost:3306/stock_data", encoding='utf-8')
        self.engine_datasets = create_engine("mysql+pymysql://root:as6114@localhost:3306/datasets", encoding='utf-8')
        # self.n = 상한가 이전 n일의 data 분석
        self.n = 10

        # for 문 위해서 종목명 list 받아오기
        sql = "SELECT TABLE_NAME FROM information_schema.tables WHERE table_schema='stock_data'"
        self.stock_list = self.engine.execute(sql).fetchall()
        self.stock_list.remove(('stock_list', ))

    def sanghanga_collection(self):
        col_x = ['o', 'h', 'l', 'c', 'v']
        for i in range(1, self.n):
            for j in col_x[0:5]:
                col_x.append(j + f'+{i}')

        # 독립변수 데이터 SET 설정
        x = pd.DataFrame([], columns=col_x)

        for idx, i in enumerate(self.stock_list):
            name = i[0]
            sql = f'select `Date`, `Open`, High, Low, Close, Volume, `Change` from `{name}` where `Change` > 0.29'
            data = pd.read_sql(sql, self.engine)
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change'])
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            # name 종목의 상한가 날짜의 dataframe

            for date in df.Date:
                sql = f"select Date, Open, High, Low, Close, Volume, `Change` from `{name}` where Date < '{date}' order by `Date` desc limit {self.n}"
                prev_n_data_for_date = pd.read_sql(sql, self.engine).sort_values('Date').reset_index(drop=True)
                # 특정 날짜에 대한 위로 n개의 data 들 => 이게 독립변수
                prev_n_data_for_date = prev_n_data_for_date[['Open', 'High', 'Low', 'Close', 'Volume']]
                if prev_n_data_for_date.size == 5*self.n:
                    flat_data = prev_n_data_for_date.values.reshape(1, 5*self.n)
                    df_flat = pd.DataFrame(flat_data, columns=col_x)
                    x = pd.concat([x, df_flat])
            print(f'dataset_x collecting... ({idx}/{len(self.stock_list)})')
        x['upper limit next day'] = 1
        x.to_sql('dataset_limit', self.engine_datasets, if_exists='replace', index=False)
        self.len_x = len(x)

        # 앞에 이름없는 index 지우기
        # randomization 하기

    def control_collection(self):
        """
        sanahanga_collection method 에서 모은 m개의 데이터와 학습 시 대조할 m개의 데이터를 랜덤하게 뽑기
        """
        col_y = ['o', 'h', 'l', 'c', 'v']
        for i in range(1, self.n):
            for j in col_y[0:5]:
                col_y.append(j + f'+{i}')

        # 독립변수 데이터 SET 설정
        y = pd.DataFrame([], columns=col_y)
        while len(y) <= self.len_x:
            # stock_list 에서 random 하게 한 원소 뽑아서
            name = random.sample(self.stock_list, 1)[0][0]
            # 해당 종목에서 random 하게 한 줄 고르고 change 가 0.29 미만이면 그 행 포함 10개 row 를 가져오자.
            sql = f"select `Date`, `Open`, `Low`, `High`, `Close`, `Volume`, `Change` FROM `{name}` ORDER BY RAND() LIMIT 1"
            rand_row = self.engine.execute(sql).fetchone()
            # change 가 0.295 미만일 때, 즉 상한가가 아닌 데이터만 모아서
            # 이 날짜 포함 10개 row를 가져와서 y에 추가해준다.
            # 언제까지? y의 길이가 x와 동등해질 때 까지
            if (rand_row[6]) and (rand_row[6] < 0.295):
                rand_date = rand_row[0]
                sql = f"SELECT `Date`, `Open`, `Low`, `High`, `Close`, `Volume` FROM `{name}` WHERE `Date` <= '{rand_date}' ORDER BY `Date` desc LIMIT {self.n}"
                rand_n_row = pd.read_sql(sql, self.engine)
                rand_n_row.sort_values('Date', inplace=True)
                rand_n_row.drop('Date', axis=1, inplace=True)
                if len(rand_n_row) == self.n:
                    rand_n_row_flat = rand_n_row.values.reshape(1, 5*self.n)
                    df_flat = pd.DataFrame(rand_n_row_flat, columns=col_y)
                    y = pd.concat([y, df_flat])
            print(f'dataset_y collecting... ({len(y)}/{self.len_x})')
        y['upper limit next day'] = 0
        y.to_sql('dataset_not_limit', self.engine_datasets, if_exists='replace', index=False)

    def learning_model(self):
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

        dataset_x = dataset[[col]]
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


if __name__ == "__main__":
    ds = Datasetter()
    # ds.sanghanga_collection()
    # ds.control_collection()










