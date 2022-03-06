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

class Datasetter:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:as6114@localhost:3306/stock_data", encoding='utf-8')

    def sanghanga_collection(self):
        sql = 'select `Open`, High, Low, Close, Volume, `Change` from 유앤아이 where `Change` > 0.29'
        data = pd.read_sql(self.engine, sql)
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change'])
        df['Name'] = '유앤아이'
        df = df[['Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']]
        # 유앤아이 종목의 상한가 날짜의 dataframe 생성

        # 독립변수 데이터 SET 설정
        x = np.array([[]], dtype='object')

        for date in df.Date:
            sql = f"select Date, Open, High, Low, Close, Volume, `Change` from 유앤아이 where Date < '{date}' order by `Date` desc limit 10"
            prev_10_data_for_date = pd.DataFrame(self.engine.execute(sql).fetchall()).sort_values('Date').reset_index(drop=True)
            # 특정 날짜에 대한 위로 10개의 data 들 => 이게 독립변수








